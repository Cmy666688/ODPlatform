from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("od_web", "0001_initial")]
    operations = [migrations.AddField(model_name="dataset", name="task",
                  field=models.CharField(default="semantic_segment", max_length=32, verbose_name="任务类型"))]
