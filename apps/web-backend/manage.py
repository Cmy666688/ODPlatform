"""Django 迁移、系统检查与管理员账号管理入口。"""
import sys

from od_web.app import configure_environment


if __name__ == "__main__":
    configure_environment()
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
