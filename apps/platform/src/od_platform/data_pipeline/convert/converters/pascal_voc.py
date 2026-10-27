# apps/platform/src/od_platform/data_pipeline/convert/converters/pascal_voc.py
"""Pascal VOC → YOLO。每图一个 .xml;<size> 给宽高,<bndbox> 是绝对坐标。
能力:detect(分割多边形 VOC 没有,注册时就声明了)。
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.converters._geometry import normalize_box
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter


@register_converter(AnnotationFormat.PASCAL_VOC, supported_tasks=(Task.DETECT,))
def convert_pascal_voc(input_dir: Path, output_labels_dir: Path,
                       options: ConvertOptions) -> list[str]:
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到任何 .xml 标注文件")

    # classes 三态:None=按数据自带;[...]=白名单(顺序即 class_id);[]=全排除
    if options.classes is None:
        found: set[str] = set()
        for index, xf in enumerate(xml_files, 1):
            found.update(o.findtext("name", "").strip() for o in ET.parse(xf).getroot().iter("object"))
            if options.on_progress: options.on_progress(index, len(xml_files)*2)
        names = sorted(found)
        classes = [n for n in names if n]
    else:
        classes = list(options.classes)
    class_to_idx = {n: i for i, n in enumerate(classes)}

    for index, xf in enumerate(xml_files, 1):
        root = ET.parse(xf).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"{xf}: 缺少 size 节点")
        W = float(size.findtext("width", ""))
        H = float(size.findtext("height", ""))
        lines: list[str] = []
        for obj in root.iter("object"):
            name = obj.findtext("name", "").strip()
            if name not in class_to_idx:
                continue                                # 白名单之外的类:静默丢弃
            bb = obj.find("bndbox")
            if bb is None:
                raise ValueError(f"{xf}: 类别 {name!r} 缺少 bndbox 节点")
            cx, cy, bw, bh = normalize_box(
                float(bb.findtext("xmin", "")), float(bb.findtext("ymin", "")),
                float(bb.findtext("xmax", "")), float(bb.findtext("ymax", "")), W, H)
            lines.append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (output_labels_dir / f"{xf.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if options.on_progress:
            offset = len(xml_files) if options.classes is None else 0
            options.on_progress(offset+index, offset+len(xml_files))
    return classes
