# 皮皮贴图修改器 —— 手工发版流程（唯一权威）

> **约定**：本项目**不再使用任何 PowerShell 一键脚本**。每次发版都严格按本 md 逐条执行，AI 也只按此 md 帮你逐步执行、绝不擅自跳步或合并步骤。

- GitHub 仓库：https://github.com/evanlumier/PiPiTextureEditor
- 打包入口：`launcher.py`
- Spec 文件：`皮皮贴图修改器.spec`
- Python 3.13 真实路径：`C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\python3.13.exe`
  （**不能用** `C:\Users\eyvanlu\AppData\Local\Microsoft\WindowsApps\python.exe`，那是 stub，会"拒绝访问"）

---

## 🚨 红线清单（每次发版前先扫一眼）

1. **exe 文件名必须是**：`皮皮贴图修改器.exe` —— 绝不允许改名。
2. **打包结构必须是拆包**：`PPTextureEditor/` 下同时含 `皮皮贴图修改器.exe`、`_internal/`、`app/` 三样，缺一不可。
3. **hiddenimports 必须包含 `cv2`**（视频导入功能依赖 opencv-python）。
4. **zip 体积必须 ≥ 100MB**。低于此值一律视为依赖漏打，**必须重打**。
5. **`app/` 里的 .py 文件数** = 源 `app/` 里的 .py 文件数（`__pycache__` 除外）。数量对不上 = 漏文件。
6. **release_notes.txt 必须 UTF-8 无 BOM**，否则 GitHub Release 页面中文会乱码。
7. **release_notes.txt 首行必须是版本号 tag**（如 `v0.8.9`），末尾必须有联系尾（`有任何问题，请联系eyvanlu`）。
8. **版本号只做小版本递增**（0.8.8 → 0.8.9），不跳大版本。
9. **只上传 zip，不上传 source**（GitHub 会自动生成 source，release 页面别再手动挂 source zip）。
10. **绝不修改打包后的 exe 名/进程名**（iOA 白名单是按名+哈希放行的）。

---

## 📋 完整发版流程（v0.X.Y → v0.X.(Y+1)）

假设当前版本是 v0.8.8，要发 v0.8.9。

### 阶段 1：准备阶段（改代码 + 定版本号）

1. **确认所有代码改动都已 commit**（发版前工作区尽量干净，或至少你清楚哪些是本版本的改动）。
2. **修改版本号**：编辑 [app/version.py](./app/version.py)，把 `__version__ = "0.8.8"` 改成 `"0.8.9"`。
   - 这个文件必须 UTF-8 保存。
3. **写 release notes 草稿**（建议先在别处写好，格式如下）：
   ```
   v0.8.9

   【新增】
   - xxx

   【优化】
   - xxx

   【修复】
   - xxx

   有任何问题，请联系eyvanlu
   ```
   - **首行必须是** `v0.8.9`（带 v 前缀）
   - **末行必须是** `有任何问题，请联系eyvanlu`
   - 中间空行用来分段，格式随意

### 🚨 阶段 1.5：架构变更红线检查表（每次发版必查）

> **本节新增于 v0.8.10 加固**。目的：把"本次能不能走增量更新"这个判断，从**依赖记忆**变成**依赖清单**。
>
> **反面案例**：v0.8.9 事故的根因就是没有这个表——本次拆出了 `theme.py` 新模块，但发版时没意识到"新增 py 文件"必须走完整包，结果老用户走增量更新时 theme.py 没被下来，`ImportError` → 用户装完打不开。

**在写 release_notes.txt 前，逐条对照下列 6 个问题。任何一项答"是"，本次必须标记为完整包发版**（详见 [RELIABILITY_PLAN.md](./RELIABILITY_PLAN.md) B 章节 UPDATE_MODE 协议）：

- [ ] 本次是否**新增了 `app/*.py` 文件**？
      （例：v0.8.9 新增了 `app/theme.py`）
- [ ] 本次是否**删除了 `app/*.py` 文件**？
- [ ] 本次是否**修改了 import 语句**引入了此前未使用过的第三方库？
      （例：新增 `import cv2`、`import moviepy` 等）
- [ ] 本次是否**修改了 `_internal/` 里的依赖版本**？
      （例：升级了 opencv-python、PySide6 等）
- [ ] 本次是否**修改了 `launcher.py` 的 `EXPECTED_MODULES` 清单**？
      （v0.8.10 起 launcher.py 里有此常量）
- [ ] 本次是否**修改了 `皮皮贴图修改器.spec` 的 hiddenimports**？

**执行规则**：

| 检查结果 | UPDATE_MODE 值 | 说明 |
|---------|---------------|------|
| 全部为「否」 | `incremental` | 老用户可走增量更新（自动） |
| 任一为「是」 | `full` | 老用户被强制引导下完整包（弹窗） |

**填写位置**：`release_notes.txt` 第二行（版本号之后）。具体协议格式见 [RELIABILITY_PLAN.md](./RELIABILITY_PLAN.md) B 章节，v0.8.10 起阶段 2 会同步更新 release_notes 模板。

**在此之前（v0.8.10 加固完成前）**：本节先做人工判断记录，如任一为"是"，请在**用户群通知**中显式提醒"请下完整包"。

---

### 阶段 2：写入 release_notes.txt（UTF-8 无 BOM）

**⚠️ 关键：不要用记事本/VSCode 直接保存**（默认可能带 BOM 或用 GBK）。用下面这段 PowerShell 一次性写入：

```powershell
$notes = @"
v0.8.9

【新增】
- xxx

【优化】
- xxx

有任何问题，请联系eyvanlu
"@
[System.IO.File]::WriteAllText(
    (Join-Path $PWD 'release_notes.txt'),
    $notes,
    (New-Object System.Text.UTF8Encoding $false)
)
```

写完后**必须验证一次**：
```powershell
# 首行必须是 v0.8.9（不是空行也不是 BOM）
Get-Content .\release_notes.txt -TotalCount 1 -Encoding UTF8
# 大小必须 > 10 字节
(Get-Item .\release_notes.txt).Length
```

同步把这次的 notes **前置追加**到 `CHANGELOG`（最新版本在最上面）：

```powershell
$new = Get-Content .\release_notes.txt -Raw -Encoding UTF8
$old = if (Test-Path .\CHANGELOG) { Get-Content .\CHANGELOG -Raw -Encoding UTF8 } else { "" }
[System.IO.File]::WriteAllText(
    (Join-Path $PWD 'CHANGELOG'),
    ($new + "`r`n`r`n" + $old),
    (New-Object System.Text.UTF8Encoding $false)
)
```

### 阶段 3：清理上次残留

```powershell
if (Test-Path .\dist\PPTextureEditor) { Remove-Item .\dist\PPTextureEditor -Recurse -Force }
Get-ChildItem .\PPTextureEditor_v*.zip -ErrorAction SilentlyContinue | Remove-Item -Force
```

`build/` 目录（PyInstaller 中间产物）可保留可删，不影响功能。要删就 `Remove-Item .\build -Recurse -Force`。

### 阶段 4：PyInstaller 打包

**必须用真实 Python 路径**，不能靠 PATH：

```powershell
$py = 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\python3.13.exe'
& $py -m PyInstaller "皮皮贴图修改器.spec" --noconfirm
```

如果这个路径失效了，用注册表重新查一次：
```powershell
(Get-ItemProperty 'HKLM:\Software\Python\PythonCore\3.13\InstallPath').ExecutablePath
```

打完后 **必须** 检查退出码：LASTEXITCODE 必须是 0。

### 阶段 5：把 app/ 复制进 dist（拆包架构必需）

PyInstaller 只打了 exe + _internal，业务代码 app/ 需要手工放进去：

```powershell
$distApp = '.\dist\PPTextureEditor'
if (Test-Path "$distApp\app") { Remove-Item "$distApp\app" -Recurse -Force }
Copy-Item .\app "$distApp\app" -Recurse

# 清 __pycache__
Get-ChildItem "$distApp\app" -Recurse -Directory -Filter "__pycache__" | 
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force }

# 清可能残留的脏文件
Get-ChildItem $distApp -Filter "error_log.txt" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $distApp -Filter "*.log" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $distApp -Filter "_update_in_progress.lock" -ErrorAction SilentlyContinue | Remove-Item -Force
```

### 阶段 6：产物完整性硬校验（红线关卡）

**任何一项不通过 = 立即回到阶段 4 重打**。

```powershell
$distApp = '.\dist\PPTextureEditor'

# 1. exe 存在且名字正确
$exe = Get-Item "$distApp\皮皮贴图修改器.exe"
if (-not $exe) { throw "exe 缺失" }

# 2. exe 名字是唯一的、正确的
$exeFiles = Get-ChildItem $distApp -Filter "*.exe"
if ($exeFiles.Count -ne 1 -or $exeFiles[0].Name -ne '皮皮贴图修改器.exe') {
    throw "exe 名字异常: $($exeFiles.Name -join ',')"
}

# 3. _internal 存在且非空
$internalCount = (Get-ChildItem "$distApp\_internal" -Recurse -File).Count
if ($internalCount -lt 100) { throw "_internal 文件数异常: $internalCount" }

# 4. app/ .py 文件数与源一致
$srcPy  = (Get-ChildItem .\app -Recurse -File -Filter *.py | Where-Object FullName -notmatch '__pycache__').Count
$distPy = (Get-ChildItem "$distApp\app" -Recurse -File -Filter *.py | Where-Object FullName -notmatch '__pycache__').Count
if ($srcPy -ne $distPy) { throw "app/ py 数量不一致 src=$srcPy dist=$distPy" }

Write-Host "✅ 产物校验通过：exe存在，_internal $internalCount 文件，app/ $distPy 个 py"
```

### 阶段 7：打 zip

```powershell
$version = (Select-String -Path .\app\version.py -Pattern '__version__\s*=\s*"([^"]+)"').Matches.Groups[1].Value
$zipName = "PPTextureEditor_v$version.zip"
$zipPath = Join-Path $PWD $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    (Resolve-Path .\dist\PPTextureEditor).Path,
    $zipPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $true    # 包含顶层 PPTextureEditor/ 目录
)

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "生成 $zipName: $sizeMB MB"
```

### 阶段 8：zip 内容硬校验（红线关卡）

**⚠️ 必须等阶段 7 那条命令**完全**返回后再查大小**。压缩过程中查到的大小是瞬时值，不是最终值。

```powershell
$zipPath = ".\PPTextureEditor_v$version.zip"

# 体积必须 >= 100MB
$size = (Get-Item $zipPath).Length
$sizeMB = [math]::Round($size / 1MB, 2)
if ($size -lt 100MB) { throw "zip 体积异常: $sizeMB MB (必须 >= 100MB)" }

# 内容必须完整
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $zipPath).Path)
try {
    $hasExe=$false;$hasApp=$false;$hasInternal=$false;$hasVer=$false
    foreach ($e in $z.Entries) {
        $n = $e.FullName -replace '\\','/'
        if ($n -eq 'PPTextureEditor/皮皮贴图修改器.exe') {$hasExe=$true}
        if ($n -like 'PPTextureEditor/app/*') {$hasApp=$true}
        if ($n -like 'PPTextureEditor/_internal/*') {$hasInternal=$true}
        if ($n -eq 'PPTextureEditor/app/version.py') {$hasVer=$true}
    }
    if (-not $hasExe)      { throw "zip 内缺 exe" }
    if (-not $hasInternal) { throw "zip 内缺 _internal/" }
    if (-not $hasApp)      { throw "zip 内缺 app/  ★ 曾经踩过的坑 ★" }
    if (-not $hasVer)      { throw "zip 内缺 app/version.py" }
    Write-Host "✅ zip 校验通过: $sizeMB MB, entries=$($z.Entries.Count)"
} finally { $z.Dispose() }
```

### 阶段 9：人工冒烟测试（不可省）

```powershell
# 直接双击 dist 里的 exe，跑 30 秒确认能启动、主界面出得来
Start-Process ".\dist\PPTextureEditor\皮皮贴图修改器.exe"
```

**必须确认**：
- exe 能启动，主窗口出现
- 各标签页能切换（贴图/Flowmap/等）
- 能加载一张图片，视频导入按钮可用（验证 cv2 打进去了）

冒烟不通过 = 回到问题原因排查，不要发。

### 阶段 10：git commit + push

```powershell
git status
git add app/version.py CHANGELOG
# 如果本次还有其他代码改动，一并 add 上
git commit -m "release: v$version"
git push origin (git rev-parse --abbrev-ref HEAD)
```

### 📢 阶段 10.5：Release 描述固定模板（每次发版必用）

> **本节新增于 v0.8.10 加固**。目的：让主动来 GitHub 看新版本的用户，第一时间被拦住，避免踩"跳版升级"的坑。

**每次执行阶段 11 的 `gh release create` 前，必须先准备好包含以下模板的 notes 文件**（可以直接用 release_notes.txt，也可以专门准备一个 `release_description.md` 供 GitHub Release 页面使用）：

```markdown
## ⚠ 从 v0.8.9 及以下版本升级的用户请注意

如果你当前使用的是 v0.8.9 或更早版本，**请务必先手动下载并安装 v0.8.10 完整包**（这是可靠性加固版），之后的所有更新才能自动完成：

👉 [下载 v0.8.10 完整包](https://github.com/evanlumier/PiPiTextureEditor/releases/tag/v0.8.10)

如果你已经在 v0.8.10 及以上，可以直接使用软件内的自动更新，或下载本页面的 zip 手动更新。

---

## 📝 本版本更新内容

（此处填写本版本 changelog，即 release_notes.txt 的内容）
```

**强制步骤**：

1. **v0.8.10 本身发版时**：这段模板顶部就是引导目标（"请下 v0.8.10"），一定要保留。
2. **v0.8.11 及以后每次发版**：模板顶部这段引导语**保留不删**，作为长期兜底。
3. **准备方式**：可以把上面的模板 + `release_notes.txt` 内容合并到一个 `release_description.md`，然后阶段 11 里用 `--notes-file .\release_description.md` 替代 `--notes-file .\release_notes.txt`。

**⚠ 注意**：`release_notes.txt` 是**软件内更新弹窗**读取的（app/updater.py），走 UTF-8 无 BOM 严格协议；而 `release_description.md` 是**GitHub Release 网页**上展示的，两者不冲突，可以内容不同。

---

### 阶段 11：创建 GitHub Release

**前置检查**：
```powershell
gh --version         # 必须已安装
gh auth status       # 必须已登录
```

**首次发这个版本**：
```powershell
gh release create "v$version" ".\PPTextureEditor_v$version.zip" `
    --title "v$version" `
    --notes-file .\release_notes.txt
```

**如果 tag 已存在**（要重发/覆盖）：
```powershell
# 更新 notes
gh release edit "v$version" --notes-file .\release_notes.txt
# 覆盖上传 zip
gh release upload "v$version" ".\PPTextureEditor_v$version.zip" --clobber
```

### 阶段 12：发布后回查（必须做）

```powershell
# 拉 release 详情，确认 zip 已上传且大小正常
gh release view "v$version" --json assets | ConvertFrom-Json | 
    Select-Object -ExpandProperty assets | 
    ForEach-Object { "{0} - {1} MB" -f $_.name, [math]::Round($_.size/1MB,2) }
```

**必须看到**：`PPTextureEditor_vX.Y.Z.zip - 12X MB`（≥100MB）。

浏览器打开 Release 页面确认：
- 版本号正确
- notes 中文没乱码
- zip 附件在
- source（GitHub 自动生成的）**不管它**，用户下的都是我们上传的 zip

---

## 🔥 常见坑速查

| 症状 | 原因 | 解决 |
|---|---|---|
| zip 体积只有几十 MB | _internal 没打全 / app/ 忘复制 | 阶段 6 校验没跑通就打了 zip，回去重跑 |
| GitHub 页面中文乱码 | notes 带 BOM 或非 UTF-8 | 用阶段 2 的 PowerShell `WriteAllText + UTF8Encoding $false` 重写 |
| pyinstaller 报"拒绝访问" | 用了 WindowsApps stub 的 python.exe | 改用真实路径 `C:\Program Files\WindowsApps\...\python3.13.exe` |
| 用户反馈"点视频导入没反应" | hiddenimports 漏了 cv2 | 检查 `皮皮贴图修改器.spec`，`hiddenimports=[..., 'cv2', ...]` |
| Release 页面 tag 已存在但上传失败 | 用了 create 而不是 upload --clobber | 见阶段 11 "如果 tag 已存在" 分支 |
| 打包后 exe 名字不对 | spec 里 `name=` 被改过 | 立刻改回 `name='皮皮贴图修改器'` —— iOA 白名单会失效 |
| 中间查 zip 体积很小 | 压缩还没完成，看到的是瞬时值 | 等阶段 7 命令返回后再看，别抢跑 |

---

## 📎 附录 A：build.ps1 原文（存档，仅供参考）

<details>
<summary>展开查看历史脚本</summary>

```powershell
# build.ps1 - 皮皮贴图修改器 一键打包脚本（强化版）
# 用法：
#   .\build.ps1                              # 默认自动小版本+1（patch+1），交互式输入 notes
#   .\build.ps1 -BumpPatch:$false            # 不递增版本号，沿用 version.py 里的当前值
#   .\build.ps1 -Version "v0.9.0"            # 手动指定版本号
#   .\build.ps1 -NotesFile release_notes.txt # 从文件读取 release notes（跳过交互，用于自动化）
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

# 【完整逻辑已迁移到本 md 的阶段 1-8】
# 【关键实现细节：从注册表 HKLM:\Software\Python\...\InstallPath 读取真实 python 路径】
# 【关键守门：zip 体积 < 100MB 立即失败；zip 内 app/ 缺失立即失败】
```

（完整原始脚本已在 v0.8.9 之前的 git 历史里，如需回溯：`git log --all --diff-filter=D -- build.ps1`）

</details>

## 📎 附录 B：release.ps1 原文（存档，仅供参考）

<details>
<summary>展开查看历史脚本</summary>

```powershell
# release.ps1 - 皮皮贴图修改器 一键发布脚本
# 用法：
#   .\release.ps1                    # Dry-run：只打印将要执行的操作，不实际执行
#   .\release.ps1 -Confirm           # 实际执行：commit + push + 创建 GitHub Release
#
# ★★★ 设计原则 ★★★
# 1. 必须人工核验 build.ps1 产物之后，再运行此脚本
# 2. 必须显式传 -Confirm 才会真正动手，否则只 dry-run
# 3. 任何关键步骤失败 → 立即 exit 1
# 4. release notes 用 --notes-file 传递，避免命令行中文乱码

# 【完整逻辑已迁移到本 md 的阶段 10-12】
# 【关键实现：dry-run 模式 / gh release create vs edit --clobber 分支 / 上传后 --json assets 回查】
```

（完整原始脚本已在 v0.8.9 之前的 git 历史里）

</details>

---

_最后更新：2026-07-03，从 build.ps1 + release.ps1 迁移而来。以后每次改流程都直接改本 md。_
