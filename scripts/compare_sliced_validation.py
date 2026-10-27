"""Compare whole-image and SAHI-inspired sliced inference on a fixed validation subset."""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms
from ultralytics import YOLO


def starts(length, size, overlap):
    if length <= size:
        return [0]
    step = max(1, int(size * (1 - overlap)))
    return sorted(set(list(range(0, max(1, length - size + 1), step)) + [length - size]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    if args.count <= 0 or args.tile_size <= 0 or not 0 <= args.overlap < 1:
        parser.error("count and tile size must be positive; overlap must be in [0, 1)")
    if not 0 < args.confidence < 1 or not 0 < args.iou <= 1:
        parser.error("confidence must be in (0, 1), and IoU must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA GPU is required")
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(config["path"])
    images = sorted(
        p
        for p in (root / config["val"]).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    random.Random(args.seed).shuffle(images)
    images = images[: args.count]
    if not images:
        raise RuntimeError("No validation images found")
    names = config["names"]
    names = (
        {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    )
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in names.items()],
        "info": {},
    }
    detections = {"whole": [], "sliced": []}
    elapsed = {"whole": 0.0, "sliced": 0.0}
    model = YOLO(args.weights)
    annotation_id = 1
    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read: {image_path}")
        height, width = image.shape[:2]
        gt["images"].append(
            {"id": index, "file_name": image_path.name, "height": height, "width": width}
        )
        label = root / "labels" / "val" / (image_path.stem + ".txt")
        if label.exists():
            for line in label.read_text().splitlines():
                cls, x, y, w, h = map(float, line.split())
                box = [(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]
                gt["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": index,
                        "category_id": int(cls),
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        options = {
            "device": 0,
            "imgsz": args.imgsz,
            "conf": args.confidence,
            "iou": args.iou,
            "max_det": 300,
            "verbose": False,
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        whole = model.predict(image, **options)[0].boxes.data.cpu().numpy()
        torch.cuda.synchronize()
        whole_time = time.perf_counter() - started
        elapsed["whole"] += whole_time
        crops, offsets = [], []
        for y in starts(height, args.tile_size, args.overlap):
            for x in starts(width, args.tile_size, args.overlap):
                crops.append(
                    image[y : min(y + args.tile_size, height), x : min(x + args.tile_size, width)]
                )
                offsets.append((x, y))
        started = time.perf_counter()
        pieces = [whole.copy()]
        for result, (x, y) in zip(model.predict(crops, batch=8, **options), offsets, strict=True):
            boxes = result.boxes.data.cpu().numpy().copy()
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            pieces.append(boxes)
        combined = np.concatenate(pieces, axis=0)
        if len(combined):
            combined[:, [0, 2]] = combined[:, [0, 2]].clip(0, width)
            combined[:, [1, 3]] = combined[:, [1, 3]].clip(0, height)
            tensors = torch.from_numpy(combined)
            selected = batched_nms(tensors[:, :4], tensors[:, 4], tensors[:, 5], args.iou)[:300]
            combined = combined[selected.numpy()]
        torch.cuda.synchronize()
        elapsed["sliced"] += time.perf_counter() - started + whole_time
        for key, boxes in [("whole", whole), ("sliced", combined)]:
            for x1, y1, x2, y2, score, cls in boxes:
                detections[key].append(
                    {
                        "image_id": index,
                        "category_id": int(cls),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
        if index % 20 == 0:
            print(f"Processed {index}/{len(images)}", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_subset.json").write_text(json.dumps([p.name for p in images], indent=2))
    (output / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    report = {
        "settings": vars(args),
        "images": len(images),
        "ground_truth_boxes": len(gt["annotations"]),
    }
    for mode in ("whole", "sliced"):
        (output / f"{mode}_detections.json").write_text(json.dumps(detections[mode]))
        if not detections[mode]:
            report[mode] = {"error": "No predictions"}
            continue
        evaluator = COCOeval(coco, coco.loadRes(detections[mode]), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        per_class = {}
        precision = evaluator.eval["precision"]
        for class_index, class_id in enumerate(evaluator.params.catIds):
            values = precision[0, :, class_index, 0, 2]
            values = values[values >= 0]
            if len(values):
                per_class[names[class_id]] = float(values.mean())
        report[mode] = {
            "mAP50_95": float(evaluator.stats[0]),
            "mAP50": float(evaluator.stats[1]),
            "AP_small": float(evaluator.stats[3]),
            "seconds_per_image": elapsed[mode] / len(images),
            "per_class_AP50": per_class,
        }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
