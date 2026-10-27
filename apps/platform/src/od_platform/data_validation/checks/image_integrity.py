"""完整解码所有图像,避免只读取文件头而漏掉截断像素数据。"""
from PIL import Image

from od_platform.common.progress import Progress
from od_platform.data_validation.registry import CheckSeverity, CheckResult, check


@check("image_integrity")
def validate_image_integrity(ctx):
    files = [file for split in ctx.snapshot.splits for file in ctx.snapshot.images_per_split[split]]
    errors = []; progress = Progress("完整解码图像", ctx.on_progress); progress(0, len(files))
    for index, file in enumerate(files, 1):
        try:
            with Image.open(file) as image: image.verify()
            with Image.open(file) as image: image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"image": str(file), "error": str(error)})
        progress(index, len(files))
    return CheckResult("image_integrity", CheckSeverity.ERROR if errors else CheckSeverity.PASS,
                       f"{len(files)} 张图像完整解码,异常 {len(errors)}", {"count": len(errors), "examples": errors[:20]})
