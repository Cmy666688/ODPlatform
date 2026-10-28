# =============================================================================
# ODPlatform 资产同步脚本(本机 Windows -> 服务器)
#
# 同步不进 Git 的运行资产:数据集 data/、模型 models/、DeepSeek 配置 .env、
# 远程 GPU 连接三件套(remote.json / remote_gpu_key / remote_known_hosts)。
# remote.json 中的本机 Windows 路径会在推送时改写为服务器 Linux 路径。
#
# 用法(PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts/deploy/sync_to_server.ps1 -Server <服务器IP>
#   可选: -User ubuntu(默认) -KeyFile <SSH私钥路径> -IncludeRuns(同步历史实验现场约3.3GB)
#
# 说明: data/(约6.8GB)走未压缩 tar 流(图片再压缩无收益、省 CPU),
#       家用宽带上行 30Mbps 约需 35-40 分钟,期间请保持网络稳定。
# =============================================================================
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [string]$User = "ubuntu",
    [string]$KeyFile = "",
    [switch]$IncludeRuns
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$target = "${User}@${Server}"
$sshOpts = @("-o", "ConnectTimeout=10")
if ($KeyFile) { $sshOpts += @("-i", $KeyFile) }

Write-Host "==> 测试 SSH 连接 $target"
ssh @sshOpts $target "echo ok" | Out-Null

Write-Host "==> 创建远端目录"
ssh @sshOpts $target "mkdir -p ~/ODPlatform/.odp-meta/web"

Write-Host "==> 同步 data/(约 6.8GB,耗时取决于上行带宽)"
tar -cf - -C "$root" data | ssh @sshOpts $target "tar -xf - -C ~/ODPlatform"

Write-Host "==> 同步 models/(约 255MB)"
tar -cf - -C "$root" models | ssh @sshOpts $target "tar -xf - -C ~/ODPlatform"

if ($IncludeRuns) {
    Write-Host "==> 同步 runs/(约 3.3GB 历史实验现场)"
    tar -cf - -C "$root" runs | ssh @sshOpts $target "tar -xf - -C ~/ODPlatform"
}

Write-Host "==> 同步 .env(DeepSeek 配置)"
scp @sshOpts "$root/.env" "${target}:~/ODPlatform/.env"

Write-Host "==> 同步远程 GPU 密钥与连接配置(路径已改写为 Linux)"
scp @sshOpts "$root/.odp-meta/web/remote_gpu_key" "${target}:~/ODPlatform/.odp-meta/web/remote_gpu_key"
scp @sshOpts "$root/.odp-meta/remote_known_hosts" "${target}:~/ODPlatform/.odp-meta/remote_known_hosts"
$remote = Get-Content "$root/.odp-meta/web/remote.json" -Raw | ConvertFrom-Json
$remote.key_file = "/home/$User/ODPlatform/.odp-meta/web/remote_gpu_key"
$remote.known_hosts = "/home/$User/ODPlatform/.odp-meta/remote_known_hosts"
($remote | ConvertTo-Json -Depth 5) | ssh @sshOpts $target "cat > ~/ODPlatform/.odp-meta/web/remote.json"
ssh @sshOpts $target "chmod 600 ~/ODPlatform/.odp-meta/web/remote_gpu_key"

Write-Host "==> 重启服务并验证"
ssh @sshOpts $target "sudo systemctl restart od-web; sleep 3; curl -fsS http://127.0.0.1:8000/api/health"

Write-Host ""
Write-Host "同步完成。面板地址: http://$Server"
