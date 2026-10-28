# =============================================================================
# ODPlatform 公网快速隧道(Cloudflare Quick Tunnel)
#
# 作用: 把本机 127.0.0.1:8000 的 Django 面板映射成 https 公网地址,无需域名、
#       无需注册 Cloudflare 账号、无需服务器。地址形如
#       https://<随机词>.trycloudflare.com
#
# 安全: 隧道只映射本机端口,不启用访问口令;地址本身即公网入口。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts/deploy/start_tunnel.ps1
#   可选 -Port 8000 指定本地端口
#
# 注意:
#   - 本机必须保持开机联网,网页才能访问;关闭窗口/重启后地址会变。
#   - 隧道只映射本机端口,不修改任何业务数据。
# =============================================================================
param(
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$metaDir = Join-Path $projectRoot '.odp-meta/web'
New-Item -ItemType Directory -Force -Path $metaDir | Out-Null

# ---- 隧道域名相关配置 ----
# Django 允许 `.trycloudflare.com` 形式的通配主机名;CSRF 可信源用通配子域,
# 这样每次随机分配的域名都不用改配置。
$env:ODP_ALLOWED_HOSTS = '.trycloudflare.com'
$env:ODP_CSRF_ORIGINS = 'https://*.trycloudflare.com'
$env:ODP_WEB_HOST = '127.0.0.1'
$env:ODP_WEB_PORT = "$Port"

function Test-Backend {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 10
        return $health.status -eq 'ok'
    } catch { return $false }
}

# ---- 后端 ----
if (-not (Test-Backend)) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "端口 $Port 已被其他服务占用,请先停止它再运行本脚本。"
    }
    Write-Host "==> 启动 Django 后端(:$Port)"
    Start-Process -FilePath (Get-Command python).Source -ArgumentList @('-m', 'od_web.app', '--port', "$Port") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $metaDir 'backend.out.log') `
        -RedirectStandardError (Join-Path $metaDir 'backend.err.log') | Out-Null
    for ($i = 0; $i -lt 40 -and -not (Test-Backend); $i++) { Start-Sleep -Milliseconds 400 }
    if (-not (Test-Backend)) { throw "后端启动失败,请查看 $metaDir\backend.err.log" }
}

# ---- 隧道 ----
$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) { throw "未找到 cloudflared,请先安装: winget install Cloudflare.cloudflared" }

$tunnelLog = Join-Path $metaDir 'tunnel.log'
if (Test-Path $tunnelLog) { Remove-Item $tunnelLog -Force }
Write-Host "==> 建立 Cloudflare 快速隧道"
Start-Process -FilePath $cloudflared `
    -ArgumentList @('tunnel', '--url', "http://127.0.0.1:$Port", '--no-autoupdate') `
    -WindowStyle Hidden `
    -RedirectStandardOutput $tunnelLog -RedirectStandardError (Join-Path $metaDir 'tunnel.err.log') | Out-Null

$publicUrl = $null
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    foreach ($file in @($tunnelLog, (Join-Path $metaDir 'tunnel.err.log'))) {
        if (Test-Path $file) {
            $match = Select-String -Path $file -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches -ErrorAction SilentlyContinue
            if ($match) { $publicUrl = $match.Matches[0].Value; break }
        }
    }
    if ($publicUrl) { break }
}
if (-not $publicUrl) { throw "未能获取隧道地址,请查看 $metaDir\tunnel.err.log" }

$publicUrl | Set-Content -Path (Join-Path $metaDir 'tunnel-url') -Encoding ascii -NoNewline
Write-Host ""
Write-Host "============================================================"
Write-Host "  公网地址: $publicUrl"
Write-Host "============================================================"
Write-Host "  关闭本机或隧道进程后地址失效;重新运行本脚本会分配新地址。"
