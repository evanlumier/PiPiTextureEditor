# release.ps1 - 皮皮贴图修改器 一键发布脚本
#
# 用法：
#   .\release.ps1                    # Dry-run：只打印将要执行的操作，不实际执行
#   .\release.ps1 -Confirm           # 实际执行：commit + push + 创建 GitHub Release
#
# ★★★ 设计原则 ★★★
# 1. 必须人工核验 build.ps1 产物之后，再运行此脚本
# 2. 必须显式传 -Confirm 才会真正动手，否则只 dry-run
# 3. 任何关键步骤失败 → 立即 exit 1
# 4. release notes 用 --notes-file 传递，避免命令行中文乱码

[CmdletBinding()]
param(
    [switch]$Confirm = $false
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
    Write-Host "发布流程已中止。" -ForegroundColor Red
    exit 1
}
function Ok($msg)   { Write-Host "  ✅ $msg" -ForegroundColor Green }
function Info($msg) { Write-Host "  ℹ️  $msg" -ForegroundColor Gray }
function Warn($msg) { Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "🚀 皮皮贴图修改器 - 发布脚本" -ForegroundColor Magenta
if (-not $Confirm) {
    Write-Host "   [模式] DRY-RUN（不会实际推送，加 -Confirm 才会执行）" -ForegroundColor Yellow
} else {
    Write-Host "   [模式] EXECUTE（将实际推送到 GitHub）" -ForegroundColor Red
}

# ────────────────────────────────────────────────────────────────
# 第一步：检查产物完整性
# ────────────────────────────────────────────────────────────────
Write-Step "[1/6] 检查发布产物"

# 1.1 版本号
$versionFile = Join-Path $ProjectRoot "app\version.py"
if (-not (Test-Path $versionFile)) { Fail "找不到 app\version.py" }
$versionContent = Get-Content $versionFile -Raw -Encoding UTF8
if ($versionContent -notmatch '__version__\s*=\s*"([^"]+)"') {
    Fail "无法从 version.py 解析版本号"
}
$version = $matches[1]
$VersionTag = "v$version"
$ZipName = "PPTextureEditor_$VersionTag.zip"
Ok "版本号: $VersionTag"

# 1.2 release notes
$notesPath = Join-Path $ProjectRoot "release_notes.txt"
if (-not (Test-Path $notesPath)) { Fail "找不到 release_notes.txt，请先运行 build.ps1" }
$notesSize = (Get-Item $notesPath).Length
if ($notesSize -lt 10) { Fail "release_notes.txt 内容过短（$notesSize 字节），可能未正确生成" }

# 检查 release notes 第一行是否匹配版本号（防止 build 之后又改了版本但 notes 没同步）
$firstLine = (Get-Content $notesPath -TotalCount 1 -Encoding UTF8).Trim()
if ($firstLine -ne $VersionTag) {
    Fail "release_notes.txt 第一行 '$firstLine' 与版本号 '$VersionTag' 不一致，请重新运行 build.ps1"
}
Ok "release_notes.txt 存在且版本号匹配"

# 1.3 zip 包
$zipPath = Join-Path $ProjectRoot $ZipName
if (-not (Test-Path $zipPath)) { Fail "找不到发布包: $ZipName，请先运行 build.ps1" }
$zipSize = (Get-Item $zipPath).Length
$zipSizeMB = [math]::Round($zipSize / 1MB, 2)
if ($zipSize -lt 100MB) { Fail "$ZipName 体积异常偏小（$zipSizeMB MB），可能不完整" }
Ok "$ZipName 存在（$zipSizeMB MB）"

# 1.4 zip 内容再次抽查（双保险）
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $hasApp = $false
    $hasInternal = $false
    foreach ($e in $zip.Entries) {
        $n = $e.FullName -replace '\\', '/'
        if ($n -match '^PPTextureEditor/app/')       { $hasApp = $true }
        if ($n -match '^PPTextureEditor/_internal/') { $hasInternal = $true }
    }
    if (-not $hasApp)      { Fail "zip 内未包含 app/，请重新运行 build.ps1" }
    if (-not $hasInternal) { Fail "zip 内未包含 _internal/" }
    Ok "zip 内容抽查通过（含 app/ 与 _internal/）"
} finally {
    $zip.Dispose()
}

# ────────────────────────────────────────────────────────────────
# 第二步：检查工具与远程
# ────────────────────────────────────────────────────────────────
Write-Step "[2/6] 检查 git 和 gh CLI"

$gitOk = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
if (-not $gitOk) { Fail "git 不在 PATH 中" }
Ok "git 已就绪"

$ghOk = $null -ne (Get-Command gh -ErrorAction SilentlyContinue)
if (-not $ghOk) { Fail "GitHub CLI (gh) 不在 PATH 中" }
Ok "gh 已就绪"

# 检查 gh 登录状态
gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "gh 未登录，请先执行 gh auth login" }
Ok "gh 已登录"

# 检查 tag 是否已存在（远程）
$existingTag = git ls-remote --tags origin "refs/tags/$VersionTag" 2>$null
if ($existingTag) {
    Warn "远程已存在 tag $VersionTag —— 后续将覆盖同名 release"
} else {
    Ok "远程不存在 tag $VersionTag（全新发布）"
}

# ────────────────────────────────────────────────────────────────
# 第三步：检查 git 工作区状态
# ────────────────────────────────────────────────────────────────
Write-Step "[3/6] 检查 git 工作区"

$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "  待提交的改动：" -ForegroundColor Yellow
    $gitStatus -split "`n" | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
} else {
    Info "工作区干净，无待提交改动"
}

# 当前分支
$branch = git rev-parse --abbrev-ref HEAD
Ok "当前分支: $branch"

# ────────────────────────────────────────────────────────────────
# 第四步：列出待执行操作（dry-run 时停在这里）
# ────────────────────────────────────────────────────────────────
Write-Step "[4/6] 待执行的操作"

Write-Host "  1. git add app/version.py CHANGELOG" -ForegroundColor White
Write-Host "  2. git commit -m `"release: $VersionTag`"" -ForegroundColor White
Write-Host "  3. git push origin $branch" -ForegroundColor White
Write-Host "  4. gh release create $VersionTag --title $VersionTag --notes-file release_notes.txt $ZipName" -ForegroundColor White
Write-Host "     （若同名 tag 已存在，则改为 gh release edit + gh release upload --clobber）" -ForegroundColor DarkGray
Write-Host "  5. gh release view $VersionTag 回查 zip 上传状态" -ForegroundColor White

if (-not $Confirm) {
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Yellow
    Write-Host "  [DRY-RUN] 已显示将执行的操作，未实际推送。" -ForegroundColor Yellow
    Write-Host "  确认无误后，请运行：" -ForegroundColor Yellow
    Write-Host "    .\release.ps1 -Confirm" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Yellow
    exit 0
}

# ────────────────────────────────────────────────────────────────
# 第五步：执行 git commit / push
# ────────────────────────────────────────────────────────────────
Write-Step "[5/6] 提交并推送"

if ($gitStatus) {
    git add -A
    if ($LASTEXITCODE -ne 0) { Fail "git add 失败" }

    git commit -m "release: $VersionTag"
    if ($LASTEXITCODE -ne 0) { Fail "git commit 失败" }
    Ok "已 commit"
} else {
    Info "无改动可提交，跳过 commit"
}

git push origin $branch
if ($LASTEXITCODE -ne 0) { Fail "git push 失败" }
Ok "已 push 到 origin/$branch"

# ────────────────────────────────────────────────────────────────
# 第六步：创建/更新 GitHub Release
# ────────────────────────────────────────────────────────────────
Write-Step "[6/6] 发布 GitHub Release"

# 判断 release 是否已存在
gh release view $VersionTag 2>&1 | Out-Null
$releaseExists = ($LASTEXITCODE -eq 0)

if ($releaseExists) {
    Warn "GitHub Release $VersionTag 已存在，将更新 notes 与 asset"

    # 更新 notes（用 --notes-file 避免中文乱码）
    gh release edit $VersionTag --notes-file $notesPath
    if ($LASTEXITCODE -ne 0) { Fail "gh release edit 失败" }
    Ok "已更新 release notes"

    # 覆盖上传 zip
    gh release upload $VersionTag $zipPath --clobber
    if ($LASTEXITCODE -ne 0) { Fail "gh release upload 失败" }
    Ok "已覆盖上传 $ZipName"
} else {
    gh release create $VersionTag $zipPath `
        --title $VersionTag `
        --notes-file $notesPath
    if ($LASTEXITCODE -ne 0) { Fail "gh release create 失败" }
    Ok "已创建 GitHub Release $VersionTag"
}

# 上传后回查
Info "回查 release 资产..."
$assetsJson = gh release view $VersionTag --json assets 2>&1
if ($LASTEXITCODE -ne 0) { Fail "gh release view 失败：$assetsJson" }

try {
    $assets = ($assetsJson | ConvertFrom-Json).assets
} catch {
    Fail "无法解析 gh release view 的输出"
}

$matched = $assets | Where-Object { $_.name -eq $ZipName }
if (-not $matched) {
    Fail "release 中未找到 $ZipName，上传可能失败！"
}

$remoteSizeMB = [math]::Round($matched.size / 1MB, 2)
if ($matched.size -lt 100MB) {
    Fail "release 上的 $ZipName 体积异常（$remoteSizeMB MB），上传不完整！"
}
Ok "release 上 $ZipName 已确认存在（$remoteSizeMB MB）"

# ────────────────────────────────────────────────────────────────
# 完成
# ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "  🎉 发布完成！" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  版本号    : $VersionTag" -ForegroundColor White
Write-Host "  分支      : $branch（已 push）" -ForegroundColor White
Write-Host "  Release   : https://github.com/evanlumier/PiPiTextureEditor/releases/tag/$VersionTag" -ForegroundColor White
Write-Host "  上传产物  : $ZipName ($remoteSizeMB MB)" -ForegroundColor White
Write-Host ""
