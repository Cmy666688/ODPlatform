"""数据集台账。训练现场继续使用原有 JSON 审计,迁移后保留既有实验。"""
from django.db import models


class Dataset(models.Model):
    id = models.CharField(primary_key=True, max_length=32)
    name = models.CharField("名称", max_length=80)
    source_name = models.CharField("上传来源", max_length=255, blank=True)
    status = models.CharField("状态", max_length=16, default="importing")
    phase = models.CharField("阶段", max_length=80, default="准备导入")
    progress = models.FloatField(default=0)
    counts = models.JSONField(default=dict)
    fingerprint = models.CharField(max_length=64, blank=True)
    report = models.JSONField(default=dict)
    error = models.TextField(blank=True)
    pid = models.IntegerField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "数据集"
        verbose_name_plural = "数据集"

    def __str__(self): return self.name
