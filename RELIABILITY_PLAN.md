# 皮皮贴图修改器 · 可靠性加固实施计划

> **文档目的**：把 v0.8.9 手动包发布事件复盘出的经验，落地成**可执行、可追溯、可维护**的加固措施，避免同类事故重演。
>
> **执行原则**：手工推进，逐条对照，绝不跳步。所有代码改动必须小步、可回滚、有验证。
>
> **当前状态**：v0.8.9 手动包已发布，本文档指导 **v0.8.10（协议升级过渡版）** 的加固工作。
>
> **⚠ 重要认知**：v0.8.10 是**协议过渡版**，从 v0.8.9 升级到 v0.8.10 这一步，**部分老用户仍需手动下完整包**（且这是行业通病，任何自动更新协议升级都躲不开）。**从 v0.8.11 起**，兜底体系才真正闭环，实现"在线自动更新失败率 < 5%、失败时用户能自助恢复"。

---

## 📍 事故复盘（为什么要做这份计划）

### v0.8.9 事故还原

- **诱因**：本次版本把 `theme.py` 等风格文件从主脚本中拆出为独立模块。
- **问题链**：
  1. 老 updater 走**增量更新**（只拉取本次改动的 `.py` 文件）
  2. 增量更新把新版 `Texture_tool_GUI_with_tabs.py`（已改为 `from theme import ...`）覆盖到用户机器
  3. 但 `theme.py` 是新拆出来的文件，在**用户旧的 app/ 目录里根本不存在**
  4. 用户重启程序 → `ImportError: No module named 'theme'` → 黑窗一闪就崩
  5. 用户看不懂错误 → 只能反馈"更新完打不开"

### 根本原因

**"增量更新"这个机制无法感知"架构变更"**。只要新版本的模块拓扑变了（拆分/合并/新增），增量就必然残缺。

### 最终决策

- v0.8.9 → 走**手动完整安装包**收尾（已完成）
- **v0.8.10（本次）** → 落地本文档所列**兜底体系**，作为协议升级过渡版
- v0.8.11 及以后 → 兜底体系闭环生效，回归全自动更新

---

## 🚧 关于协议升级的过渡代价（必须先说清楚）

**在开工之前，必须先对齐这件事**：v0.8.10 无法做到"老 v0.8.9 用户 100% 无感自动升级"。

### 为什么？

老 v0.8.9 用户手上的 `updater.py` 代码**已经烧死在他们的机器上**，我们无法远程升级它 —— 除非它自己"换脑子"（也就是升级到 v0.8.10）。

而老 updater 读不懂本次新增的 `[UPDATE_MODE] full` 协议，还是会走增量路径。这条路径的结果依赖 launcher 自检兜底（C 措施），**大概率**能被友好弹窗拦住，但不是 100%。

### 用户会经历什么？

```
老 v0.8.9 用户
   │
   ├─ 场景 A（约 50-70%）：老 updater 增量下 v0.8.10
   │      ↓
   │   新 launcher.py 已被下来 → 启动时自检
   │      ↓
   │   若某个新加的模块（如 self_check.py）没被增量下来
   │      ↓
   │   🟡 弹友好窗口："请下完整包"（用户点按钮跳 GitHub）
   │
   └─ 场景 B（约 30-50%）：老 updater 增量完整
          ↓
       正常启动，用户无感升级 ✅
```

### 和这次 v0.8.9 事件的本质区别

| 对比项 | v0.8.9 事件（这次） | v0.8.10 过渡（未来） |
|-------|-------------------|---------------------|
| 用户体感 | 双击 exe → 黑窗一闪就没了 | 弹友好窗口，明确告知点这里下载 |
| 用户懂不懂发生了啥 | 完全不懂，只能求助 | 一目了然 |
| 你的排查成本 | 用户描述"打不开"，抓瞎 | 用户截图弹窗，一眼看到缺哪个模块 |
| 用户手动装的门槛 | 高（要联系客服要包） | 低（弹窗按钮直接跳 GitHub） |

### 时间线预期

```
v0.8.9  (已发) 手动装
   │
   ▼
v0.8.10 (本次)  ← 协议过渡版
   │  ← 老用户从 v0.8.9 升到这里，
   │    30-50% 概率仍需手动一次，
   │    但会看到清晰弹窗指引
   ▼
v0.8.11 及以后  ← 兜底体系闭环
   │  ← 从这里起，在线自动更新失败率 <5%
   │    失败时用户能自助恢复
   ▼
稳态运行
```

**⚠ 这不是我们做得不够好，是任何自动更新系统协议升级的必经之路（Chrome、微信、Steam 都经历过）。**

---

## 🛡️ 三层兜底体系总览

针对"老用户跳过 v0.8.10 直接升到更新版本"这个最痛的盲区，我们建立三层兜底：

```
┌─────────────────────────────────────────────────────────┐
│  🥇 兜底 1：GitHub Release 描述固定引导语（运营侧）        │
│                                                          │
│  每个新版本 Release 描述顶部固定一段：                      │
│  "从 v0.8.9 及以下升级的用户，请先下载 v0.8.10 完整包"     │
│                                                          │
│  ✅ 主动看 GitHub 的用户被拦住                            │
│  ❌ 没主动看的用户漏网                                    │
│  兜底率：~40%                                            │
└─────────────────────────────────────────────────────────┘
                          ↓ 未拦住
┌─────────────────────────────────────────────────────────┐
│  🥈 兜底 2：updater_generation.txt 跳版警告（代码侧）      │
│                                                          │
│  v0.8.10+ 的 updater 每次启动写标记文件：                  │
│    %APPDATA%\PPTextureEditor\updater_generation.txt      │
│                                                          │
│  launcher 启动检测：                                      │
│  · 文件存在且值 ≥ 0.8.10 → 一切正常                       │
│  · 文件不存在 或 值 < 0.8.10 → 弹跳版警告                 │
│                                                          │
│  ✅ 拦住"跳版升级"场景                                    │
│  兜底率：~70%                                            │
└─────────────────────────────────────────────────────────┘
                          ↓ 未拦住
┌─────────────────────────────────────────────────────────┐
│  🥉 兜底 3：launcher 依赖自检（保底，一定生效）             │
│                                                          │
│  启动时探测 EXPECTED_MODULES 清单，任何一个 import 失败：   │
│    → 弹友好窗口，列出缺失模块，引导下完整包                 │
│                                                          │
│  ✅ 只要 launcher.py 本身正确被下来（几乎必然），就一定生效  │
│  兜底率：~95%                                            │
└─────────────────────────────────────────────────────────┘
```

### 综合覆盖度评估

| 用户类型 | 兜底 1 | 兜底 2 | 兜底 3 | 最终结果 |
|---------|:-----:|:-----:|:-----:|---------|
| 主动看 GitHub 的用户 | ✅ | - | - | 🟢 完美体验 |
| 直接升级、跳过 v0.8.10 | ❌ | ✅ | - | 🟡 弹跳版警告，明确指引 |
| 兜底 2 失效（罕见） | ❌ | ❌ | ✅ | 🟡 弹自检失败弹窗，虽不精准但不崩 |
| 三层全失效（极罕见） | ❌ | ❌ | ❌ | 🔴 双击打不开，联系客服 |

**综合覆盖度：~92-95%**（诚实数字，不做营销包装）。

---

## 🎯 加固目标

**不追求 99% 的营销数字，追求"故障可控"：**

> **任何故障发生时，用户不会陷入死循环、不会数据丢失，且我们能快速定位和修复。**

达成这个目标后：
- ✅ 不会再有"用户装完打不开还不知道为啥" — 弹窗告知重下完整包
- ✅ 不会再有"忘了这次是架构变更导致老用户增量残缺" — UPDATE_MODE 强制标记
- ✅ 不会再有"跳版升级导致用户悄无声息损坏" — updater_generation.txt 主动提醒
- ✅ 不会再有"打包漏依赖" — 冒烟测试拦发版

---

## 📦 计划分波次

### 🚀 第一波（v0.8.10 核心内容，本次做完）

| # | 措施 | 类型 | 预估工时 |
|---|------|------|---------|
| A | RELEASE.md 增加"架构变更红线检查表" | 纯文档 | 15 分钟 |
| B | UPDATE_MODE 元数据机制（release_notes.txt 头部标记 + updater 读取） | 代码 | 2 小时 |
| C | launcher.py 依赖自检 + 友好错误弹窗 + 跳版警告 | 代码 | 2 小时 |
| **R** | **GitHub Release 描述固定模板（兜底 1）** | **纯运营** | **10 分钟** |

**第一波完成后覆盖度**：**~92-95% 场景**，剩余 5-8% 也能被 launcher 弹窗兜住不崩。

### 🎯 第二波（v0.8.11 或之后，从容做）

| # | 措施 | 类型 | 预估工时 |
|---|------|------|---------|
| D | updater 更新前备份 + 失败自动回滚 | 代码 | 3 小时 |
| E | 打包冒烟测试（--self-check 参数） | 代码 + 脚本 | 2 小时 |
| F | updater 失败次数熔断 | 代码 | 1 小时 |
| G | Release notes 模板 + MIN_COMPAT_VERSION | 代码 + 文档 | 1.5 小时 |

**第二波完成后覆盖度**：**~97% 场景**（含备份/回滚后损坏率降到极低）。

### 🌟 第三波（可选，看用户量增长决定）

- H：`manifest.json` 完整性校验
- I：崩溃自动上报（Sentry / 自建接口）
- J：双通道更新源

---

# 🚀 第一波详细实施

## 📝 A. RELEASE.md 增加"架构变更红线检查表"

### 目标

把"本次能不能走增量"这个判断，从**依赖记忆**变成**依赖清单**。

### 改动内容

在现有 [RELEASE.md](./RELEASE.md) 的**「阶段 2：写 release_notes.txt」之前**新增一节：

```markdown
## 🚨 阶段 1.5：架构变更红线检查（每次发版必查）

在写 release_notes.txt 前，逐条对照下列问题，任何一项答"是"，
本次必须走 **UPDATE_MODE=full**（详见 UPDATE_MODE 机制章节）：

- [ ] 本次是否**新增了 `app/*.py` 文件**？
      （例：v0.8.9 新增了 `app/theme.py`）
- [ ] 本次是否**删除了 `app/*.py` 文件**？
- [ ] 本次是否**修改了 import 语句**引入了此前未使用过的第三方库？
      （例：新增 `import cv2`、`import moviepy` 等）
- [ ] 本次是否**修改了 `_internal/` 里的依赖版本**？
      （例：升级了 opencv-python、PySide6 等）
- [ ] 本次是否**修改了 launcher.py 的 EXPECTED_MODULES 清单**？
- [ ] 本次是否**修改了 `皮皮贴图修改器.spec` 的 hiddenimports**？

**执行规则**：
- 全部为「否」→ UPDATE_MODE=incremental（老用户走增量更新）
- 任一为「是」→ UPDATE_MODE=full（老用户被强制走完整安装包）

**填写位置**：release_notes.txt 第二行（版本号之后）
```

### 验收标准

- [ ] RELEASE.md 中新增此小节，位于原「阶段 2」之前
- [ ] 已经把 v0.8.9 事故作为反面案例嵌在说明里
- [ ] 与后续 B 措施的 UPDATE_MODE 标记规范一致

---

## 🔄 B. UPDATE_MODE 元数据机制

### 目标

让 updater 能**从 release_notes.txt 读到本次更新是"增量"还是"完整"**，从而自动决定行为。

### 涉及文件

1. **RELEASE.md**：规范 release_notes.txt 头部格式
2. **app/updater.py**：读取头部元数据，分支处理

### B-1. release_notes.txt 新格式

**旧格式**（当前）：
```
v0.8.9
- 修复 XXX
- 新增 XXX
- 优化 XXX

有任何问题，请联系eyvanlu
```

**新格式**（v0.8.10 起）：
```
v0.8.10
[UPDATE_MODE] full
[MIN_COMPAT_VERSION] 0.8.10
- 修复 XXX
- 新增 XXX
- 优化 XXX

有任何问题，请联系eyvanlu
```

**字段说明**：

| 字段 | 值 | 含义 |
|------|-----|------|
| `[UPDATE_MODE]` | `full` | 老用户不能走增量，必须下完整包 |
| `[UPDATE_MODE]` | `incremental` | 老用户可以走增量更新 |
| `[MIN_COMPAT_VERSION]` | `0.8.10` | 低于此版本的老用户即使 mode=incremental 也强制走 full |

**兼容性**：老 updater（v0.8.9 及以前）读不懂这两行，会当作普通 changelog 显示——**不会崩，只是它自己该增量还是增量**（这是不可避免的历史包袱，v0.8.10 起的 updater 才认这个协议，也正是 launcher 自检 + 跳版警告要兜的场景）。

### B-2. updater.py 改动逻辑

**改动点**（伪代码，实际实施时再定精确位置）：

```python
def parse_release_notes(notes_text: str) -> dict:
    """
    解析 release_notes.txt 的头部元数据。
    返回 {'update_mode': 'full'|'incremental', 'min_compat': '0.8.10' or None}
    """
    meta = {'update_mode': 'incremental', 'min_compat': None}  # 默认值
    for line in notes_text.splitlines()[:5]:  # 只扫头 5 行
        if line.startswith('[UPDATE_MODE]'):
            val = line.split(']', 1)[1].strip().lower()
            if val in ('full', 'incremental'):
                meta['update_mode'] = val
        elif line.startswith('[MIN_COMPAT_VERSION]'):
            meta['min_compat'] = line.split(']', 1)[1].strip()
    return meta

def decide_update_strategy(current_ver, remote_ver, meta) -> str:
    """
    返回 'incremental' | 'full_required' | 'skip'
    """
    if meta['update_mode'] == 'full':
        return 'full_required'
    if meta['min_compat'] and version_less_than(current_ver, meta['min_compat']):
        return 'full_required'
    return 'incremental'
```

**当返回 `full_required` 时**：
- 弹窗（**不是崩溃、不是自动下载**）：
  > "本次更新（v0.8.X）包含架构调整，无法通过增量更新完成。
  > 
  > 请点击下方按钮前往 GitHub 下载完整安装包。
  > 
  > [ 前往 GitHub ]  [ 稍后提醒 ]"
- 点击"前往 GitHub" → 打开 https://github.com/evanlumier/PiPiTextureEditor/releases/latest
- 点击"稍后提醒" → 关闭对话框，本次不再检查（下次启动再问）

### B-3. updater_generation.txt 标记（配合 C 措施跳版警告）

**每次 v0.8.10+ 的 updater 启动时**（在做任何更新检查之前）：

```python
def write_updater_generation():
    """
    在 %APPDATA%\PPTextureEditor\ 下写标记文件，
    记录当前 updater 的"代际版本"（不是 app 版本，是 updater 协议版本）。
    """
    import os
    from app.version import UPDATER_GENERATION  # 新增常量，见下
    appdata = os.path.join(os.environ['APPDATA'], 'PPTextureEditor')
    os.makedirs(appdata, exist_ok=True)
    with open(os.path.join(appdata, 'updater_generation.txt'), 'w', encoding='utf-8') as f:
        f.write(UPDATER_GENERATION)
```

**同时在 `app/version.py` 新增**：

```python
# app/version.py
VERSION = '0.8.10'
UPDATER_GENERATION = '0.8.10'   # ← 新增：updater 协议代际
```

**重要**：`UPDATER_GENERATION` **不等于** `VERSION`。它只在 updater 协议本身升级时才递增（大部分小版本发布时保持不变）。这样 launcher 只用检查这个字段，不用管 app 版本号。

### 验收标准

- [ ] `parse_release_notes` 单元自测：能正确读出 mode 和 min_compat
- [ ] 老格式的 release_notes.txt（无这两行）→ 默认 incremental，不影响
- [ ] `full_required` 时弹窗正常，按钮功能正常
- [ ] 弹窗关闭后**不能崩溃、不能自动关软件**（用户还在正常使用旧版）
- [ ] `updater_generation.txt` 每次 updater 启动都被正确写入 %APPDATA%

---

## 🛡️ C. launcher.py 依赖自检 + 友好错误弹窗 + 跳版警告

### 目标

启动时先探测所有关键模块 + 检查 updater 代际，任何异常 → **弹友好窗口而不是崩溃黑窗**。

### 涉及文件

1. **launcher.py**：新增自检逻辑 + 跳版检测
2. （可选）新增 `app/self_check.py` 独立模块，避免 launcher 过于臃肿
3. **app/version.py**：新增 `UPDATER_GENERATION` 常量

### C-1. 关键模块清单（`EXPECTED_MODULES`）

写死在 launcher.py 顶部（每次架构变更时同步——已进 A 的检查表）：

```python
EXPECTED_MODULES = [
    # ── 第三方依赖（安装环境）──
    'cv2',
    'numpy',
    'PIL',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    # ── 项目自有模块（app/ 目录，通过 sys.path.insert 后按平级名 import）──
    # ⚠ 注意：app/ 不是 Python 包（无 __init__.py），launcher 是把 app_dir
    # 加进 sys.path 后直接 import 平级名，所以这里必须写 'theme' 而非 'app.theme'
    'theme',
    'widgets',
    'export_dir_mixin',
    'updater',
    'utils',
    'version',
    'tab_transfer',
    'growth_algorithms',
    'ue4_sync',
    'dialogs',
    'flowmap_tab',
    'growth_gray_tab',
    'image_viewer_tab',
    'sprite_sheet_tab',
    'Texture_tool_GUI_with_tabs',
]
```

**⚠ 前置条件**：`run_self_check()` 必须在 `sys.path.insert(0, app_dir)` **之后**调用，否则 app/ 里的模块全找不到。见下方 C-6 启动流程。

### C-2. 自检流程

```python
def run_self_check() -> list[str]:
    """
    尝试 import 所有关键模块，返回失败的模块列表。
    """
    import importlib
    failed = []
    for mod in EXPECTED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}  →  {type(e).__name__}: {e}")
    return failed
```

### C-3. 跳版警告检测（新增）

```python
def check_updater_generation() -> str | None:
    """
    检查 %APPDATA%\PPTextureEditor\updater_generation.txt。
    返回：
      · None                → 一切正常，不弹警告
      · 'missing'           → 文件不存在（用户是从 v0.8.9 及以下跳版升上来的）
      · 'outdated:X.Y.Z'    → 文件存在但版本过老
    """
    import os
    from app.version import UPDATER_GENERATION_MIN  # 常量：当前 launcher 要求的最低 updater 代际
    path = os.path.join(os.environ['APPDATA'], 'PPTextureEditor', 'updater_generation.txt')
    if not os.path.exists(path):
        return 'missing'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            gen = f.read().strip()
        if version_less_than(gen, UPDATER_GENERATION_MIN):
            return f'outdated:{gen}'
    except Exception:
        return 'missing'
    return None
```

**在 `app/version.py` 中同步新增**：
```python
UPDATER_GENERATION_MIN = '0.8.10'   # launcher 要求 updater 至少是这个代际
```

**跳版警告弹窗**（区别于自检失败弹窗）：

```
┌─────────────────────────────────────────────────┐
│  ⚠  检测到您可能跳过了关键的可靠性加固版本         │
├─────────────────────────────────────────────────┤
│                                                  │
│  您当前的更新组件版本可能不完整。                   │
│  为了确保后续更新的稳定性，                        │
│  建议立即下载最新完整安装包。                       │
│                                                  │
│      [ 前往 GitHub ]   [ 我知道了 ]              │
│                                                  │
└─────────────────────────────────────────────────┘
```

**注意**：跳版警告**不阻断启动**（用户点"我知道了"能继续用），只是主动提醒。而自检失败弹窗（C-4）是**必须阻断启动**的严重问题。

### C-4. 依赖失败时的友好弹窗

**弹窗方案决策**：使用 **PySide6 QMessageBox**（非自定义 QDialog）。

理由：
- **可靠性优先**：QMessageBox 是 Qt 内置最稳定的组件，几乎不会因为主题/样式相关的 import 失败而"弹窗自己也弹不出来"。而 QDialog 需要引入 [app/theme.py](./app/theme.py) 或类似样式模块——如果 theme.py 恰好就是那个坏了的文件，弹窗就绝对失败（这是本次事故的根因场景之一）。
- **展现质感差异不明显**：这是错误弹窗，用户此刻已经很崩溃了，颜值不是关键，"能弹出来 + 信息清晰 + 按钮好按"才是关键。QMessageBox 已经能满足。
- **实施成本低**：QMessageBox 30 分钟能搞定，QDialog 至少 1.5 小时。省下的时间可以留给测试。
- **未来若需升级**：用户量大了、有明确美化需求，再重构成 QDialog 也不晚——本次先追求"能救命"。

**注意**：QMessageBox **必须在 QApplication 已经存在的前提下弹**——但 launcher 依赖自检发生在 QApplication 创建之前，所以自检失败时需要**先创建一个最小化的 QApplication 只用于弹窗**，弹完立即 `sys.exit(1)`。

**弹窗视觉效果**：

```
┌─────────────────────────────────────────────────┐
│  ⚠  皮皮贴图修改器 · 启动失败                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  检测到程序文件不完整，无法正常启动。              │
│                                                  │
│  以下模块缺失或损坏：                              │
│    · theme  →  ModuleNotFoundError: ...          │
│    · cv2    →  ImportError: ...                  │
│                                                  │
│  【推荐处理方式】                                  │
│  请前往 GitHub 下载最新的完整安装包，              │
│  解压覆盖到当前目录后重启程序。                    │
│                                                  │
│      [ 打开 GitHub Release ]   [ 关闭 ]           │
│                                                  │
│  如仍无法解决，请联系 eyvanlu                      │
└─────────────────────────────────────────────────┘
```

### C-5. `--self-check` 命令行参数（为 E 措施铺路）

launcher 支持：
```
皮皮贴图修改器.exe --self-check
```

- 只跑 `run_self_check()`，通过则打印 `SELF_CHECK_OK` 并 `exit(0)`
- 失败则打印失败列表并 `exit(1)`
- **不启动任何 GUI**

这一步为**第二波 E 措施（打包冒烟测试）**提供接口。

### 启动流程（整合三个检查点）

```
launcher.py 启动
    │
    ├─ 1. 解析命令行（--self-check 特殊分支）
    │
    ├─ 2. run_self_check()  ← C-2
    │      └─ 失败 → 弹依赖失败弹窗（阻断） → exit
    │
    ├─ 3. 创建 QApplication（正常启动流程）
    │
    ├─ 4. check_updater_generation()  ← C-3
    │      └─ 返回非 None → 弹跳版警告（不阻断，用户可跳过）
    │
    └─ 5. 进入主界面
```

### 验收标准

- [ ] 所有 EXPECTED_MODULES 都能被正确探测
- [ ] 手工删除 `app/theme.py` 后启动 → 弹依赖失败弹窗（阻断）
- [ ] 手工删除 `%APPDATA%\PPTextureEditor\updater_generation.txt` → 弹跳版警告（不阻断）
- [ ] "打开 GitHub Release" 按钮能正确打开浏览器
- [ ] `--self-check` 参数在完整环境下 exit 0，缺文件时 exit 1
- [ ] 正常启动时**自检对启动速度的影响 < 500ms**（探测清单不宜过大）
- [ ] 跳版警告不影响正常功能（用户点"我知道了"能继续用）

---

## 📢 R. GitHub Release 描述固定模板（兜底 1）

### 目标

让主动来 GitHub 看的用户在**发现新版本的第一秒**就被拦住，避免踩"跳版升级"的坑。

### 改动内容

从 **v0.8.10 起**，每次发 Release 时，描述顶部固定粘贴以下模板（写进 [RELEASE.md](./RELEASE.md) 阶段 11 "gh release create" 之前作为强制步骤）：

```markdown
## ⚠ 从 v0.8.9 及以下版本升级的用户请注意

如果你当前使用的是 v0.8.9 或更早版本，**请务必先手动下载并安装 v0.8.10 完整包**（这是可靠性加固版），之后的所有更新才能自动完成：

👉 [下载 v0.8.10 完整包](https://github.com/evanlumier/PiPiTextureEditor/releases/tag/v0.8.10)

如果你已经在 v0.8.10 及以上，可以直接使用软件内的自动更新，或下载本页面的 zip 手动更新。

---

## 📝 本版本更新内容

（此处填写本版本 changelog）
```

### 验收标准

- [ ] 该模板已加入 [RELEASE.md](./RELEASE.md) 的 "阶段 11：gh release create" 之前
- [ ] v0.8.10 发布时，Release 描述里包含此模板（v0.8.10 本身就是那个"引导目标"）
- [ ] 后续 v0.8.11+ 每次发版都保留此模板不删

---

# 🔄 第一波实施顺序（严格按此推进）

**⚠ 合并原则**：小型改动能合就合，减少上下文切换。全流程压缩到 **8 大步 + 1 次集成测试 + 1 次发版**。每完成一步做一次 git commit，方便中途 review 或回滚。

```
Step 1【纯文档，10 分钟】
├─ A. RELEASE.md 新增"阶段 1.5：架构变更红线检查"
└─ R. RELEASE.md 阶段 11 之前新增"Release 描述模板"章节
   └─ 一次 commit：docs(release): 新增架构变更检查表 + Release 描述模板

Step 2【代码，30 分钟】
├─ 修改 app/version.py，新增：
│   · UPDATER_GENERATION = '0.8.10'
│   · UPDATER_GENERATION_MIN = '0.8.10'
└─ 修改 app/version.py，__version__ 保持 '0.8.9'（改成 0.8.10 放到 Step 8 发版前）
   └─ 一次 commit：feat(version): 新增 updater 协议代际常量

Step 3【代码，1.5 小时】launcher.py 自检骨架（不含跳版警告）
├─ 定义 EXPECTED_MODULES 常量
├─ 实现 run_self_check() 函数
├─ 实现依赖失败弹窗（QMessageBox 方案）
├─ 加入 --self-check 命令行参数分支
└─ 修改 main()：sys.path.insert 之后调 run_self_check → 失败弹窗+exit
   └─ 一次 commit：feat(launcher): 加入依赖自检 + 友好错误弹窗

Step 4【手工验证 Step 3】测试脚本自动化，见下方"🧪 自动化测试清单"

Step 5【代码，2 小时】app/updater.py UPDATE_MODE 元数据机制
├─ 实现 parse_release_notes() 函数
├─ 实现 decide_update_strategy() 函数
├─ 实现 write_updater_generation() 并接入 updater 模块启动流程
├─ 实现 full_required 弹窗（QMessageBox 方案）
└─ 集成到主更新流程（在增量决策点插入分支）
   └─ 一次 commit：feat(updater): 支持 UPDATE_MODE 协议 + 代际标记

Step 6【手工验证 Step 5】测试脚本自动化，见下方"🧪 自动化测试清单"

Step 7【代码，30 分钟】launcher.py 补齐跳版警告（C-3）
├─ 实现 check_updater_generation() 函数
├─ 实现跳版警告弹窗（QMessageBox 方案）
└─ 在 main() 里 QApplication 之后调用（不阻断启动）
   └─ 一次 commit：feat(launcher): 加入跳版警告检测

Step 8【手工验证 Step 7 + 全链路集成测试】
├─ 手工验证跳版警告（自动化脚本兜底）
├─ 修改 app/version.py：__version__ = '0.8.10'
├─ 严格按 RELEASE.md 12 阶段（含新增的 1.5）打 v0.8.10-beta 完整包
├─ 用 v0.8.9 手动包在另一目录模拟老用户
├─ 验证"老 updater → 新 launcher 兜底"链路
└─ 集成通过 → 正式发布 v0.8.10（阶段 11 使用 R 模板）
```

### 步骤依赖关系图

```
Step 1 (文档)          ← 独立，可与任何步骤并行
   ↓
Step 2 (version 常量)  ← Step 3 和 Step 5 都依赖
   ↓
Step 3 (launcher 自检骨架)     Step 5 (updater 全套)
   ↓                              ↓
Step 4 (测试)                  Step 6 (测试)
                                  ↓
                               Step 7 (跳版警告)  ← 依赖 Step 5 的 write_updater_generation
                                  ↓
                               Step 8 (集成 + 发版)
```

**关键路径**：Step 2 → Step 5 → Step 7 → Step 8。其余步骤可穿插做。

### 单次 commit 尺寸控制

- Step 1：**~50 行 md 变更**
- Step 2：**~5 行代码**
- Step 3：**~120 行代码**（launcher.py 增量）
- Step 5：**~150 行代码**（updater.py 增量 + 少量 version.py）
- Step 7：**~50 行代码**

**总计代码变更 ~325 行**，不算大改。每一步都能独立 review。

---

# 🧪 自动化测试清单（由你执行的测试脚本或命令）

**说明**：以下测试你可以自己做，我不代跑。每一步测试都设计成"改动 → 验证 → 恢复"三阶段，确保不留副作用。

## 🔬 Step 4 测试：验证 launcher 自检（Step 3 后跑）

### 测试 4.1：正常启动
```powershell
# 期望：无弹窗，正常进入主界面
python launcher.py
```
✅ 通过标准：无异常，主界面出现

### 测试 4.2：依赖模块缺失（改名法）
```powershell
# 制造故障
Rename-Item app\theme.py app\theme.py.bak

# 触发测试
python launcher.py

# 期望：弹窗"启动失败，缺少 theme 模块"，主界面不出现

# 恢复
Rename-Item app\theme.py.bak app\theme.py
```
✅ 通过标准：弹窗正常出现且列出 theme 缺失

### 测试 4.3：--self-check 参数（正常环境）
```powershell
python launcher.py --self-check
echo "EXIT_CODE=$LASTEXITCODE"
```
✅ 通过标准：输出 `SELF_CHECK_OK`，`EXIT_CODE=0`

### 测试 4.4：--self-check 参数（故障环境）
```powershell
Rename-Item app\theme.py app\theme.py.bak
python launcher.py --self-check
echo "EXIT_CODE=$LASTEXITCODE"
Rename-Item app\theme.py.bak app\theme.py
```
✅ 通过标准：输出失败模块列表，`EXIT_CODE=1`

### 测试 4.5：启动性能
```powershell
Measure-Command { python launcher.py --self-check }
```
✅ 通过标准：TotalMilliseconds < 500

---

## 🔬 Step 6 测试：验证 updater 元数据机制（Step 5 后跑）

**前置**：需要打包一个 exe（因为 updater 只在 `sys.frozen` 环境下工作）。或者临时修改 updater.py 的 frozen 检测让开发环境也能跑。

### 测试 6.1：造假 release_notes（触发 full_required）
```powershell
# 造假一个 release_notes 文件
$fake = @"
v0.8.99
[UPDATE_MODE] full
[MIN_COMPAT_VERSION] 0.8.10
- 测试用假版本

有任何问题，请联系eyvanlu
"@
[System.IO.File]::WriteAllText(
    "$env:TEMP\fake_notes.txt",
    $fake,
    (New-Object System.Text.UTF8Encoding $false)
)

# 手工调用 parse_release_notes 解析（在 python 里跑一段）
python -c @"
import sys; sys.path.insert(0, 'app')
from updater import parse_release_notes, decide_update_strategy
notes = open(r'$env:TEMP\fake_notes.txt', encoding='utf-8').read()
meta = parse_release_notes(notes)
print('META:', meta)
strategy = decide_update_strategy('0.8.9', '0.8.99', meta)
print('STRATEGY:', strategy)
assert meta['update_mode'] == 'full'
assert strategy == 'full_required'
print('OK')
"@
```
✅ 通过标准：输出 `STRATEGY: full_required`，最后 `OK`

### 测试 6.2：老格式 release_notes（走增量）
```powershell
$old = @"
v0.8.99
- 老格式 changelog

有任何问题，请联系eyvanlu
"@
[System.IO.File]::WriteAllText(
    "$env:TEMP\old_notes.txt",
    $old,
    (New-Object System.Text.UTF8Encoding $false)
)

python -c @"
import sys; sys.path.insert(0, 'app')
from updater import parse_release_notes, decide_update_strategy
notes = open(r'$env:TEMP\old_notes.txt', encoding='utf-8').read()
meta = parse_release_notes(notes)
strategy = decide_update_strategy('0.8.9', '0.8.99', meta)
assert meta['update_mode'] == 'incremental'
assert strategy == 'incremental'
print('OK')
"@
```
✅ 通过标准：输出 `OK`（不报错）

### 测试 6.3：MIN_COMPAT_VERSION 强制升级（跨版本）
```powershell
$strict = @"
v0.8.99
[UPDATE_MODE] incremental
[MIN_COMPAT_VERSION] 0.8.15
- 严格协议

有任何问题，请联系eyvanlu
"@
[System.IO.File]::WriteAllText(
    "$env:TEMP\strict_notes.txt",
    $strict,
    (New-Object System.Text.UTF8Encoding $false)
)

python -c @"
import sys; sys.path.insert(0, 'app')
from updater import parse_release_notes, decide_update_strategy
notes = open(r'$env:TEMP\strict_notes.txt', encoding='utf-8').read()
meta = parse_release_notes(notes)
strategy = decide_update_strategy('0.8.10', '0.8.99', meta)
assert strategy == 'full_required'   # 0.8.10 < 0.8.15，被强制 full
print('OK')
"@
```
✅ 通过标准：输出 `OK`（低于 min_compat 触发 full_required）

### 测试 6.4：updater_generation.txt 写入
```powershell
$gen = "$env:APPDATA\PPTextureEditor\updater_generation.txt"
Remove-Item $gen -ErrorAction SilentlyContinue

python -c @"
import sys; sys.path.insert(0, 'app')
from updater import write_updater_generation
write_updater_generation()
"@

Get-Content $gen
```
✅ 通过标准：文件存在且内容为 `0.8.10`

### 测试 6.5：清理
```powershell
Remove-Item "$env:TEMP\fake_notes.txt","$env:TEMP\old_notes.txt","$env:TEMP\strict_notes.txt" -EA 0
```

---

## 🔬 Step 8 测试：验证跳版警告 + 全链路集成

### 测试 8.1：跳版警告（模拟老用户从 v0.8.9 升到 v0.8.10）
```powershell
# 制造"老用户"环境：删除代际标记文件
Remove-Item "$env:APPDATA\PPTextureEditor\updater_generation.txt" -EA 0

# 启动 launcher
python launcher.py

# 期望：先弹跳版警告（不阻断），点"我知道了"后正常进入主界面
```
✅ 通过标准：跳版警告弹窗出现且不阻断启动

### 测试 8.2：正常启动（有代际标记）
```powershell
# 保证代际标记存在
"0.8.10" | Out-File -Encoding utf8 -NoNewline "$env:APPDATA\PPTextureEditor\updater_generation.txt"

# 启动
python launcher.py
```
✅ 通过标准：无跳版警告，直接进主界面

### 测试 8.3：全链路（真实打包场景）
- 严格按 [RELEASE.md](./RELEASE.md) 12 阶段打包 v0.8.10-beta 完整包
- 在 `C:\test_v089\` 解压 v0.8.9 手动包（模拟老用户）
- 修改 `C:\test_v089\PPTextureEditor\app\version.py` 的 `__version__ = "0.8.9"`（如已经是就不改）
- 启动 v0.8.9 手动包里的 exe，让老 updater 检测到 v0.8.10 更新
- 观察行为链：
  - 老 updater 增量下载 → 新 launcher.py 被下来
  - 新 launcher 启动 → 若模块完整则跳版警告，若模块缺失则依赖失败弹窗
- ✅ 通过标准：**弹窗友好、不崩溃、按钮能跳 GitHub**

---

# 📖 章节索引

> 已合并到前面的 **8 步分步表** 和 **🧪 自动化测试清单**。此处保留章节索引方便快速跳转。

| 需要看 | 跳转章节 |
|--------|---------|
| 每一步该做什么 | [🔄 第一波实施顺序](#-第一波实施顺序严格按此推进)（8 步分步表）|
| 步骤间的依赖关系 | 步骤依赖关系图（8 步分步表内）|
| 手工测试用哪些脚本 | [🧪 自动化测试清单](#-自动化测试清单由你执行的测试脚本或命令) |
| A 措施改什么 | [📝 A. RELEASE.md 增加"架构变更红线检查表"](#-a-releasemd-增加架构变更红线检查表) |
| B 措施改什么 | [🔄 B. UPDATE_MODE 元数据机制](#-b-update_mode-元数据机制) |
| C 措施改什么 | [🛡️ C. launcher.py 依赖自检 + 友好错误弹窗 + 跳版警告](#-c-launcherpy-依赖自检--友好错误弹窗--跳版警告) |
| R 措施改什么 | [📢 R. GitHub Release 描述固定模板（兜底 1）](#-r-github-release-描述固定模板兜底-1) |
| 三层兜底怎么串起来 | [🛡️ 三层兜底体系总览](#️-三层兜底体系总览) |
| 已知风险和预案 | 下方 [⚠️ 风险与预案](#️-风险与预案) |

---

# ⚠️ 风险与预案

## 风险 1：C 措施的 EXPECTED_MODULES 清单遗漏

**场景**：某天新增了 `app/new_feature.py`，忘了加进清单。
**后果**：用户装完运行正常，但如果 updater 增量更新丢了这个新文件，自检不会报错，还是会崩。
**预案**：
- A 措施的检查表里已经有一条"是否修改了 launcher.py 的 EXPECTED_MODULES 清单"
- 未来第二波 E 措施（冒烟测试）会在打包环节强制跑一次自检，能进一步兜住

## 风险 2：B 措施的 updater 改动引入回归

**场景**：改 updater.py 时手抖，把原本正常的增量更新逻辑改坏了。
**后果**：v0.8.10 之后的所有增量更新都失败。
**预案**：
- **改动前先备份 updater.py**（`git status` 确认干净后再改）
- 改动**只加新函数、只在决策点插入分支判断**，不动原有下载/写文件逻辑
- 集成测试时**必须**验证"正常增量流程"和"full_required 流程"两条路径

## 风险 3：C 措施的自检拖慢启动

**场景**：EXPECTED_MODULES 太长，每次启动多花 2-3 秒。
**后果**：用户体感变差。
**预案**：
- 清单控制在 20 项以内
- 使用 `importlib.util.find_spec()` 而非真 import（更快，只查存在性）——**如果启动实测超过 500ms 再切换到这个方案**

## 风险 4：v0.8.10 发出去还是有用户装完打不开

**场景**：兜底做完了，但某个我们没想到的场景还是出问题。
**后果**：又是一次事故。
**预案**：
- v0.8.10 的 release_notes 明确说"本版本为可靠性加固版本，请务必下载完整包"
- launcher 弹窗里写了联系方式（eyvanlu），用户能找到我们
- 我们能通过弹窗信息（缺哪个模块）快速定位问题，不再是黑窗一闪

## 风险 5：用户跳过 v0.8.10 直接升更高版本（**最痛盲区**）

**场景**：v0.8.9 用户没升 v0.8.10，等到 v0.8.15 才更新。老 updater 依然读不懂协议，走增量下 v0.8.15，可能残缺。
**后果**：用户装完打不开。
**预案**（三层兜底）：
- 兜底 1（GitHub Release 模板）：主动看 GitHub 的用户被拦住 —— 覆盖 ~40%
- 兜底 2（跳版警告）：`updater_generation.txt` 缺失时弹警告 —— 累计覆盖 ~70%
- 兜底 3（launcher 自检）：即使前两层失效，只要 launcher.py 被增量下来了，自检必然生效 —— 累计覆盖 ~92-95%
- **无法 100% 消灭**（这是行业通病），但故障时用户能自助恢复，不再"黑窗一闪求助"

## 风险 6：`updater_generation.txt` 被杀软/清理软件误删

**场景**：360 或类似软件把 `%APPDATA%\PPTextureEditor\updater_generation.txt` 当垃圾清了。
**后果**：正常用户被误判为"跳版用户"，被弹跳版警告。
**预案**：
- 跳版警告**不阻断启动**（用户点"我知道了"能继续用）
- updater 每次运行都会重写这个文件（自愈）
- 弹窗文案措辞留有余地："**可能**跳过了关键版本"，避免绝对化

---

# 📚 参考文档

- [RELEASE.md](./RELEASE.md) - 12 阶段发布流程
- [CHANGELOG](./CHANGELOG) - 版本变更记录
- [launcher.py](./launcher.py) - 启动器（C 措施主战场）
- [app/updater.py](./app/updater.py) - 自动更新器（B 措施主战场）
- [app/version.py](./app/version.py) - 版本号定义（新增 `UPDATER_GENERATION` / `UPDATER_GENERATION_MIN`）

---

# 📌 附录：第二波 & 第三波简述（仅记录，暂不实施）

### 第二波 D. updater 备份 + 回滚
更新前 `app/` → `app.backup/`，更新失败自动还原。

### 第二波 E. 打包冒烟测试
`build.py`（未来的打包脚本或 RELEASE.md 手工步骤）在生成 zip 前，先跑 `皮皮贴图修改器.exe --self-check`，exit 非 0 则拒绝出包。

### 第二波 F. updater 失败熔断
本地记录连续失败次数，达 2 次自动停用增量，永远走 full_required。

### 第二波 G. Release notes 完整模板
锁死头部三行格式，配合 MIN_COMPAT_VERSION 做严格版本管理。

### 第三波 H/I/J
manifest.json、崩溃上报、双通道更新源。视用户量增长再评估。

---

**本计划最后更新**：v0.8.10 加固启动前
**执行方式**：手工，逐条对照，绝不批处理跳步
**联系人**：eyvanlu
