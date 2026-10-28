#!/usr/bin/env bash
# =============================================================================
# ODPlatform 公网部署初始化脚本(服务器端一次性执行)
#
# 目标环境: Oracle Cloud Always Free(ARM64)+ Ubuntu 24.04,默认用户 ubuntu。
# 用法: 把本脚本上传到服务器后执行  bash bootstrap.sh
#
# 完成内容:
#   1. 安装 Python 3.12 venv / git / Caddy / ufw
#   2. 克隆代码到 ~/ODPlatform 并安装 platform + web-backend(CPU 版依赖)
#   3. 生成站点环境配置 /etc/odp/od-web.env(主机名与 CSRF 可信源)
#   4. 注册并启动 systemd 服务 od-web(127.0.0.1:8000,崩溃自动重启)
#   5. 配置 Caddy 监听 80 端口反代到 8000,并启用 ufw 防火墙
#
# 私有仓库: 用 REPO_URL 携带只读 token,例如
#   REPO_URL=https://<token>@github.com/Cmy666688/ODPlatform.git bash bootstrap.sh
#
# 注意: Oracle 控制台的【子网安全列表】还需手动放行 TCP 80(脚本管不到云平台侧)。
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Cmy666688/ODPlatform.git}"
APP_DIR="$HOME/ODPlatform"
ENV_FILE="/etc/odp/od-web.env"

echo "==> 安装系统依赖"
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y universe || true
sudo apt-get install -y python3-venv python3-dev git ufw caddy curl

SERVER_IP="${SERVER_IP:-$(curl -fsSL --max-time 5 ifconfig.me || hostname -I | awk '{print $1}')}"
echo "==> 公网 IP: $SERVER_IP"

echo "==> 拉取代码"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> 创建虚拟环境并安装依赖(约 2GB,首次较慢)"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ./apps/platform -e ./apps/web-backend
.venv/bin/odp-init

echo "==> 生成站点环境配置"
sudo install -d -m 750 /etc/odp
if [ ! -f "$ENV_FILE" ]; then
  printf 'ODP_ALLOWED_HOSTS=%s\nODP_CSRF_ORIGINS=http://%s\n' \
    "$SERVER_IP" "$SERVER_IP" | sudo tee "$ENV_FILE" > /dev/null
  sudo chmod 600 "$ENV_FILE"
else
  echo "    $ENV_FILE 已存在,保留既有配置"
fi

echo "==> 注册 systemd 服务"
sudo tee /etc/systemd/system/od-web.service > /dev/null <<UNIT
[Unit]
Description=ODPlatform Web 面板
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/.venv/bin/python -m od_web.app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now od-web

echo "==> 配置 Caddy(80 端口反代,无域名走纯 HTTP)"
sudo tee /etc/caddy/Caddyfile > /dev/null <<'CADDY'
:80 {
    encode gzip
    reverse_proxy 127.0.0.1:8000
    header {
        X-Content-Type-Options nosniff
        Referrer-Policy no-referrer
    }
}
CADDY
sudo systemctl reload caddy || sudo systemctl restart caddy

echo "==> 启用防火墙(仅放行 22/80)"
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw --force enable

echo ""
echo "============================================================"
echo "  部署完成: http://$SERVER_IP"
echo "  下一步: 在本机运行 scripts/deploy/sync_to_server.ps1 同步数据与密钥"
echo "  提醒: Oracle 控制台子网安全列表需手动放行 TCP 80 入站"
echo "============================================================"
