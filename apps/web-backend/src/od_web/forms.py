from django import forms


class DatasetUploadForm(forms.Form):
    name = forms.CharField(max_length=80)
    mode = forms.ChoiceField(choices=[("archive", "压缩包"), ("folder", "文件夹")])
