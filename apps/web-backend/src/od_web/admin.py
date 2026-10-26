from django.contrib import admin

from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ["name", "id", "status", "source_name", "created_at"]
    search_fields = ["name", "id", "source_name"]
    list_filter = ["status"]
    readonly_fields = [field.name for field in Dataset._meta.fields]

    def has_add_permission(self, request): return False
    def has_delete_permission(self, request, obj=None): return False
