# build.ps1 - 皮皮贴图修改器 一键打包脚本
# 用法：在项目根目录下运行 .\build.ps1

# ★★★ 更新守则与白名单处理说明 ★★★
# 
# 打包架构：拆包设计（exe + _internal/ + app/）
# 目的：实现 iOA 白名单一劳永逸方案
# 
# 白名单处理流程：
# 1. 首次打包后，将整个 dist/PPTextureEditor/ 目录提供给管理员
# 2. 管理员需要将以下内容加入 iOA 白名单：
#    - 皮皮贴图修改器.exe（完整权限）
#    - _internal/ 目录（读写执行权限）
#    - app/ 目录（读写执行权限）
# 
# 更新优势：
# - 主 exe 文件哈希值固定，白名单不会失效
# - 未来版本更新只需替换 app/ 目录内容
# - 无需重新申请白名单，实现一劳永逸
#
# 注意事项：
# - 不要修改 exe 文件名（保持「皮皮贴图修改器.exe」）
# - 打包前确保已安装 opencv-python：pip install opencv-python
# - 打包完成后验证目录结构是否正确

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  皮皮贴图修改器 - 一键打包脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── 第一步：PyInstaller 打包 exe ──
Write-Host "[1/3] 正在打包 exe（PyInstaller）..." -ForegroundColor Green
pyinstaller "皮皮贴图修改器.spec" --noconfirm
if ($LASTEXITCODE -ne 0) {
    Write-Host "打包失败！请检查错误信息。" -ForegroundColor Red
    exit 1
}
Write-Host "exe 打包完成。" -ForegroundColor Green
Write-Host ""

# ── 第二步：复制 app/ 业务代码到 dist ──
Write-Host "[2/3] 正在复制业务代码到 dist/PPTextureEditor/app/ ..." -ForegroundColor Green
$distAppDir = "dist/PPTextureEditor/app"
if (Test-Path $distAppDir) {
    Remove-Item $distAppDir -Recurse -Force
}
Copy-Item -Path "app" -Destination $distAppDir -Recurse
Write-Host "业务代码复制完成。" -ForegroundColor Green
Write-Host ""

# ── 第三步：验证输出 ──
Write-Host "[3/3] 验证输出目录..." -ForegroundColor Green
$exePath = "dist/PPTextureEditor/皮皮贴图修改器.exe"
$appDir = "dist/PPTextureEditor/app"
$internalDir = "dist/PPTextureEditor/_internal"

$allOk = $true
if (Test-Path $exePath) {
    Write-Host "  ✅ exe 文件存在: $exePath" -ForegroundColor Green
} else {
    Write-Host "  ❌ exe 文件缺失: $exePath" -ForegroundColor Red
    $allOk = $false
}
if (Test-Path $appDir) {
    $appFiles = (Get-ChildItem $appDir -Recurse -File).Count
    Write-Host "  ✅ app/ 目录存在 ($appFiles 个文件)" -ForegroundColor Green
} else {
    Write-Host "  ❌ app/ 目录缺失" -ForegroundColor Red
    $allOk = $false
}
if (Test-Path $internalDir) {
    Write-Host "  ✅ _internal/ 目录存在" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  _internal/ 目录缺失（可能打包配置有问题）" -ForegroundColor Yellow
}

Write-Host ""
if ($allOk) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  打包完成！" -ForegroundColor Cyan
    Write-Host "  输出目录: dist/PPTextureEditor/" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
} else {
    Write-Host "打包过程中存在问题，请检查上方输出。" -ForegroundColor Red
}
