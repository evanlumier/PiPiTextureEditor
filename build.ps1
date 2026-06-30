# build.ps1 - 皮皮贴图修改器 一键打包脚本（强化版）
# 用法：
#   .\build.ps1                       # 默认自动小版本+1（patch+1）
#   .\build.ps1 -BumpPatch:$false     # 不递增版本号，沿用 version.py 里的当前值
#   .\build.ps1 -Version "v0.9.0"     # 手动指定版本号
#
# ★★★ 设计原则 ★★★
# 1. 任何关键步骤失败 → 立即 exit 1，绝不允许带病推进
# 2. 流程最开头就交互式询问 release notes，避免最后才发现忘写
# 3. release_notes.txt 用 UTF-8 无 BOM 写入，根治中文乱码
# 4. 本脚本只负责"准备产物"，绝不触碰 git/GitHub，由 release.ps1 推送
#
# 打包架构：拆包设计（exe + _internal/ + app/）
# - 主 exe 文件哈希值固定，iOA 白名单一劳永逸
# - 未来版本更新只需替换 app/ 目录内容

[CmdletBinding()]
param(
    [switch]$BumpPatch = $true,
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host ""
    Write-Host "❌ $msg" -ForegroundColor Red
    Write-Host "打包流程已中止。" -ForegroundColor Red
    exit 1
}

function Ok($msg)   { Write-Host "  ✅ $msg" -ForegroundColor Green }
function Info($msg) { Write-Host "  ℹ️  $msg" -ForegroundColor Gray }
function Warn($msg) { Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "🐶 皮皮贴图修改器 - 打包脚本" -ForegroundColor Magenta
Write-Host "   工作目录: $ProjectRoot" -ForegroundColor DarkGray

# ────────────────────────────────────────────────────────────────
# 第一步：决定本次版本号
# ────────────────────────────────────────────────────────────────
Write-Step "[1/8] 确定版本号"

$versionFile = Join-Path $ProjectRoot "app\version.py"
if (-not (Test-Path $versionFile)) { Fail "找不到版本文件: $versionFile" }

$versionFileContent = Get-Content $versionFile -Raw -Encoding UTF8
if ($versionFileContent -notmatch '__version__\s*=\s*"([^"]+)"') {
    Fail "无法从 version.py 解析版本号，请检查文件格式。"
}
$currentVersion = $matches[1]
Info "当前版本号: $currentVersion"

if ($Version -ne "") {
    # 手动指定，去掉前缀 v
    $newVersion = $Version.TrimStart('v', 'V')
    Info "手动指定版本号: $newVersion"
} elseif ($BumpPatch) {
    if ($currentVersion -notmatch '^(\d+)\.(\d+)\.(\d+)$') {
        Fail "当前版本号格式不是 x.y.z，无法自动递增: $currentVersion"
    }
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    $patch = [int]$matches[3] + 1
    $newVersion = "$major.$minor.$patch"
    Info "自动递增版本号: $currentVersion → $newVersion"
} else {
    $newVersion = $currentVersion
    Info "沿用当前版本号: $newVersion"
}

# 写回 version.py（仅当版本号变了）
if ($newVersion -ne $currentVersion) {
    $newContent = $versionFileContent -replace '__version__\s*=\s*"[^"]+"', "__version__ = `"$newVersion`""
    # UTF-8 无 BOM 写入
    [System.IO.File]::WriteAllText($versionFile, $newContent, (New-Object System.Text.UTF8Encoding $false))
    Ok "已更新 app/version.py → $newVersion"
} else {
    Info "版本号未变化，跳过写回。"
}

$VersionTag = "v$newVersion"   # 用于 release tag
$ZipName    = "PPTextureEditor_$VersionTag.zip"

# ────────────────────────────────────────────────────────────────
# 第二步：交互式收集 release notes（最先做，避免后面忘）
# ────────────────────────────────────────────────────────────────
Write-Step "[2/8] 收集本次版本更新内容（Release Notes）"

Write-Host ""
Write-Host "请输入本次版本（$VersionTag）的更新内容。" -ForegroundColor Yellow
Write-Host "支持多行，输入完成后单独一行键入 END 回车即可结束。" -ForegroundColor Yellow
Write-Host "（直接键入 END 表示放弃本次构建）" -ForegroundColor DarkYellow
Write-Host "------------------------------------------------" -ForegroundColor DarkGray

$lines = New-Object System.Collections.Generic.List[string]
# 第一行先放版本号
$lines.Add($VersionTag) | Out-Null
$lines.Add("") | Out-Null

$userLineCount = 0
while ($true) {
    $line = Read-Host
    if ($line -ceq "END") { break }
    $lines.Add($line) | Out-Null
    $userLineCount++
}

if ($userLineCount -eq 0) {
    Fail "未输入任何更新内容，已放弃本次构建。"
}

# 末尾加联系信息
$lines.Add("") | Out-Null
$lines.Add("有任何问题，请联系eyvanlu") | Out-Null

$notesContent = ($lines -join "`r`n")
$notesPath = Join-Path $ProjectRoot "release_notes.txt"

# 关键：UTF-8 无 BOM 写入，避免 PowerShell 默认编码导致 GitHub 显示乱码
[System.IO.File]::WriteAllText($notesPath, $notesContent, (New-Object System.Text.UTF8Encoding $false))
Ok "已写入 $notesPath（UTF-8 无 BOM）"

# 同步追加到 CHANGELOG（前置追加，最新版本在最上面）
$changelogPath = Join-Path $ProjectRoot "CHANGELOG"
$oldChangelog = ""
if (Test-Path $changelogPath) {
    $oldChangelog = Get-Content $changelogPath -Raw -Encoding UTF8
}
$newChangelog = $notesContent + "`r`n`r`n" + $oldChangelog
[System.IO.File]::WriteAllText($changelogPath, $newChangelog, (New-Object System.Text.UTF8Encoding $false))
Ok "已更新 CHANGELOG"

Write-Host "------------------------------------------------" -ForegroundColor DarkGray
Write-Host "[预览本次 Release Notes]" -ForegroundColor Gray
Write-Host $notesContent -ForegroundColor White
Write-Host "------------------------------------------------" -ForegroundColor DarkGray

# ────────────────────────────────────────────────────────────────
# 第三步：清理上次的 dist 残留
# ────────────────────────────────────────────────────────────────
Write-Step "[3/8] 清理上次构建残留"

$distRoot = Join-Path $ProjectRoot "dist"
$distApp  = Join-Path $distRoot   "PPTextureEditor"
if (Test-Path $distApp) {
    Remove-Item $distApp -Recurse -Force
    Ok "已清理 dist/PPTextureEditor/"
} else {
    Info "dist/PPTextureEditor/ 不存在，无需清理。"
}

# 旧 zip 也清掉，避免误传
$oldZip = Join-Path $ProjectRoot $ZipName
if (Test-Path $oldZip) {
    Remove-Item $oldZip -Force
    Info "删除旧 zip: $ZipName"
}

# ────────────────────────────────────────────────────────────────
# 第四步：PyInstaller 打包
# ────────────────────────────────────────────────────────────────
Write-Step "[4/8] PyInstaller 打包 exe"

pyinstaller "皮皮贴图修改器.spec" --noconfirm
if ($LASTEXITCODE -ne 0) { Fail "PyInstaller 打包失败（exit code $LASTEXITCODE）。" }
Ok "PyInstaller 打包完成。"

# ────────────────────────────────────────────────────────────────
# 第五步：复制 app/ 到 dist（拆包架构必要步骤）
# ────────────────────────────────────────────────────────────────
Write-Step "[5/8] 复制 app/ 业务代码到 dist"

$srcAppDir  = Join-Path $ProjectRoot "app"
$distAppDir = Join-Path $distApp     "app"
if (Test-Path $distAppDir) {
    Remove-Item $distAppDir -Recurse -Force
}
Copy-Item -Path $srcAppDir -Destination $distAppDir -Recurse

# 清理 dist/app 下的 __pycache__
Get-ChildItem $distAppDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
Ok "已复制并清理 __pycache__"

# 清理 dist 根目录可能残留的脏文件（error_log.txt 等）
$dirtyPatterns = @("error_log.txt", "*.log", "_update_in_progress.lock")
foreach ($pattern in $dirtyPatterns) {
    Get-ChildItem $distApp -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item $_.FullName -Force
        Info "清理脏文件: $($_.Name)"
    }
}

# ────────────────────────────────────────────────────────────────
# 第六步：硬性校验 dist 产物
# ────────────────────────────────────────────────────────────────
Write-Step "[6/8] 校验 dist 产物完整性"

$exePath     = Join-Path $distApp "皮皮贴图修改器.exe"
$internalDir = Join-Path $distApp "_internal"

if (-not (Test-Path $exePath))     { Fail "exe 文件缺失: $exePath" }
Ok "exe 文件存在: 皮皮贴图修改器.exe"

if (-not (Test-Path $internalDir)) { Fail "_internal/ 目录缺失，PyInstaller 配置可能有问题。" }
$internalCount = (Get-ChildItem $internalDir -Recurse -File).Count
Ok "_internal/ 目录存在（$internalCount 个文件）"

if (-not (Test-Path $distAppDir))  { Fail "app/ 目录缺失，请检查复制步骤。" }

# 对比源 app/ 与 dist/app/ 的 .py 文件数（忽略 __pycache__）
$srcPyCount  = (Get-ChildItem $srcAppDir  -Recurse -File -Filter "*.py" |
    Where-Object { $_.FullName -notmatch '__pycache__' }).Count
$distPyCount = (Get-ChildItem $distAppDir -Recurse -File -Filter "*.py" |
    Where-Object { $_.FullName -notmatch '__pycache__' }).Count
if ($srcPyCount -ne $distPyCount) {
    Fail "源 app/ 与 dist/app/ 的 .py 文件数不一致（源=$srcPyCount, dist=$distPyCount）"
}
Ok "app/ 目录完整（$distPyCount 个 .py 文件，与源一致）"

# 防呆：exe 必须叫"皮皮贴图修改器.exe"
$exeFiles = Get-ChildItem $distApp -Filter "*.exe"
if ($exeFiles.Count -ne 1 -or $exeFiles[0].Name -ne "皮皮贴图修改器.exe") {
    Fail "exe 文件名异常，必须保持「皮皮贴图修改器.exe」。当前: $($exeFiles.Name -join ', ')"
}
Ok "exe 文件名正确"

# ────────────────────────────────────────────────────────────────
# 第七步：打包成 zip
# ────────────────────────────────────────────────────────────────
Write-Step "[7/8] 压缩为 $ZipName"

$zipPath = Join-Path $ProjectRoot $ZipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

# 用 .NET 直接打 zip，比 Compress-Archive 快且兼容性好
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $distApp,
    $zipPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $true   # 包含 PPTextureEditor/ 顶层目录
)
$zipSize = (Get-Item $zipPath).Length
$zipSizeMB = [math]::Round($zipSize / 1MB, 2)
Ok "已生成 $ZipName（$zipSizeMB MB）"

# ────────────────────────────────────────────────────────────────
# 第八步：zip 内容回查（关键守门，防止昨天那种漏 app/ 的事）
# ────────────────────────────────────────────────────────────────
Write-Step "[8/8] zip 内容回查"

if ($zipSize -lt 100MB) {
    Fail "zip 体积异常偏小（$zipSizeMB MB），完整包应 ≥ 100MB，可能内容缺失。"
}
Ok "zip 体积正常（$zipSizeMB MB ≥ 100MB）"

$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $entries = $zip.Entries
    $entryCount = $entries.Count

    $hasExe      = $false
    $hasApp      = $false
    $hasInternal = $false
    $hasVersionPy = $false

    foreach ($e in $entries) {
        $n = $e.FullName -replace '\\', '/'
        if ($n -match '^PPTextureEditor/皮皮贴图修改器\.exe$')  { $hasExe = $true }
        if ($n -match '^PPTextureEditor/app/')                   { $hasApp = $true }
        if ($n -match '^PPTextureEditor/_internal/')             { $hasInternal = $true }
        if ($n -match '^PPTextureEditor/app/version\.py$')       { $hasVersionPy = $true }
    }

    Info "zip 包含 $entryCount 个 entry"
    if (-not $hasExe)      { Fail "zip 内未包含 PPTextureEditor/皮皮贴图修改器.exe" }
    Ok "zip 内 exe 存在"
    if (-not $hasInternal) { Fail "zip 内未包含 PPTextureEditor/_internal/" }
    Ok "zip 内 _internal/ 存在"
    if (-not $hasApp)      { Fail "zip 内未包含 PPTextureEditor/app/ ★ 这是上次踩过的坑 ★" }
    Ok "zip 内 app/ 存在"
    if (-not $hasVersionPy){ Fail "zip 内未包含 PPTextureEditor/app/version.py" }
    Ok "zip 内 version.py 存在"
} finally {
    $zip.Dispose()
}

# ────────────────────────────────────────────────────────────────
# 完成
# ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "  ✅ 打包流程全部完成，请人工核验！" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  版本号       : $VersionTag" -ForegroundColor White
Write-Host "  Zip 路径     : $ZipName ($zipSizeMB MB)" -ForegroundColor White
Write-Host "  Release Notes: release_notes.txt" -ForegroundColor White
Write-Host "  Dist 目录    : dist\PPTextureEditor\" -ForegroundColor White
Write-Host ""
Write-Host "  人工核验建议：" -ForegroundColor Yellow
Write-Host "    1. 打开 dist\PPTextureEditor\皮皮贴图修改器.exe，确认能正常启动" -ForegroundColor Yellow
Write-Host "    2. 检查 release_notes.txt 内容无误（无乱码、版本号正确）" -ForegroundColor Yellow
Write-Host "    3. 解压 $ZipName 到任意目录，确认 app/ 与 _internal/ 都在" -ForegroundColor Yellow
Write-Host ""
Write-Host "  核验通过后，请运行：" -ForegroundColor Cyan
Write-Host "    .\release.ps1 -Confirm" -ForegroundColor Cyan
Write-Host ""
