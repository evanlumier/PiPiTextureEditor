"""
launcher.py —— 轻量级启动器
作为打包入口，在导入主模块之前提供最外层异常兜底。
当主模块 import 失败或初始化阶段崩溃时，用标准库弹窗提示用户并写入日志，
而不是直接无响应退出。

★ 架构说明：
  打包后目录结构为 exe + _internal/ + app/，
  业务代码全部在 app/ 子目录下，launcher 通过 importlib 动态加载。
  这样每次发版只需更新 app/ 目录，exe 和 _internal/ 保持不变，
  避免 iOA 安全白名单因文件哈希变化而失效。
"""
import sys
import os
import traceback
import subprocess
import importlib
from datetime import datetime

# ── exe 文件名白名单配置 ──
_EXPECTED_EXE_NAME = "皮皮贴图修改器.exe"

# ── app/ 目录完整性自检清单（v0.8.10 加固版起启用） ──
# 每次新增或删除 app/ 下的 .py 业务文件，都必须同步维护本清单。
# 启动时任一文件缺失，视为安装损坏，弹窗提示并阻止启动，
# 避免半损坏状态下让用户看到 Python traceback。
_EXPECTED_APP_MODULES = (
    "Texture_tool_GUI_with_tabs.py",  # 主入口
    "dialogs.py",
    "export_dir_mixin.py",
    "flowmap_tab.py",
    "growth_algorithms.py",
    "growth_gray_tab.py",
    "image_viewer_tab.py",
    "sprite_sheet_tab.py",
    "tab_transfer.py",
    "theme.py",
    "ue4_sync.py",
    "updater.py",
    "utils.py",
    "version.py",
    "widgets.py",
)


def _get_base_dir() -> str:
    """获取 exe/脚本所在目录（项目根目录）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _get_app_dir() -> str:
    """
    智能定位 app/ 业务代码目录。
    优先查找 <base_dir>/app/，如果不存在则回退到 base_dir 本身（向后兼容）。
    """
    base = _get_base_dir()
    app_dir = os.path.join(base, "app")
    if os.path.isdir(app_dir):
        return app_dir
    # 向后兼容：如果 app/ 目录不存在（未拆包），回退到根目录
    return base


# ── 在最早的时机设置好日志和全局异常钩子 ──
_BASE_DIR = _get_base_dir()
_LOG_PATH = os.path.join(_BASE_DIR, "error_log.txt")


def _write_log(exc_text: str) -> str:
    """将异常信息追加写入 error_log.txt，返回日志文件路径"""
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"平台: {sys.platform}\n")
            f.write(f"可执行文件: {sys.executable}\n")
            f.write(f"工作目录: {os.getcwd()}\n")
            f.write(exc_text)
            f.write("\n")
    except Exception:
        pass
    return _LOG_PATH


def _show_fatal_dialog(exc_text: str, log_path: str):
    """
    弹出错误对话框，三级兜底：
    1. 优先用 Windows 原生 MessageBoxW（ctypes，零依赖，最可靠）
    2. 备选 tkinter
    3. 都不行时，至少尝试 os.startfile 打开日志
    """
    short_msg = exc_text
    if len(short_msg) > 800:
        short_msg = short_msg[:800] + "\n... (详见日志文件)"

    full_msg = (
        f"程序启动失败，请将错误日志发给开发者：\n\n"
        f"{short_msg}\n"
        f"{'─' * 50}\n"
        f"错误日志已保存至：\n{log_path}"
    )

    # 第一级：Windows 原生 MessageBoxW（ctypes 是内置模块，不需要额外依赖，最可靠）
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            full_msg,
            "皮皮贴图修改器 - 启动失败",
            0x10  # MB_ICONERROR
        )
        return
    except Exception:
        pass

    # 第二级：tkinter
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("皮皮贴图修改器 - 启动失败", full_msg)
        root.destroy()
        return
    except Exception:
        pass

    # 第三级：尝试用系统默认程序打开日志文件
    try:
        os.startfile(log_path)
    except Exception:
        pass


def _global_excepthook(exc_type, exc_value, exc_tb):
    """全局未捕获异常钩子 —— 确保任何层面的崩溃都能被记录"""
    if exc_type is SystemExit:
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    exc_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    # 控制台输出（如果有控制台的话）
    print(f"\n[致命错误]\n{exc_text}", file=sys.stderr, flush=True)
    log_path = _write_log(exc_text)
    _show_fatal_dialog(exc_text, log_path)


# 在 Python 解释器层面注册全局异常钩子（比 try/except 更早生效）
sys.excepthook = _global_excepthook


def _check_exe_name():
    """
    检查 exe 文件名是否被修改，如果被修改则弹窗提示并自动重命名后重启。
    仅在打包环境（frozen）下生效，开发环境跳过。
    """
    if not getattr(sys, 'frozen', False):
        return  # 开发环境，不检查

    exe_path = sys.executable
    exe_name = os.path.basename(exe_path)

    if exe_name == _EXPECTED_EXE_NAME:
        return  # 文件名正确，放行

    # ── 文件名被修改，弹窗提示 ──
    msg = (
        "由于iOA会对插件在线更新版本报风险项，"
        "因此已联系8000加白名单为「皮皮贴图修改器」，"
        "修改名字会造成插件无法使用，接下来将自动重命名。"
    )

    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            msg,
            "皮皮贴图修改器 - 文件名异常",
            0x30  # MB_ICONWARNING
        )
    except Exception:
        pass

    # ── 生成 bat 脚本：等待当前进程退出 → 重命名 → 重启 ──
    exe_dir = os.path.dirname(exe_path)
    new_path = os.path.join(exe_dir, _EXPECTED_EXE_NAME)

    # 如果目标文件名已存在（比如旧版本残留），先删掉
    # 注意：bat 文件用 GBK 编码写入，cmd.exe 默认代码页就是 GBK(936)，
    # 不要使用 chcp 65001 切换到 UTF-8，否则中文路径/文件名会乱码导致操作失败！
    bat_content = f'''@echo off
echo 正在等待程序退出...
:wait_loop
tasklist /FI "PID eq {os.getpid()}" 2>nul | find /I "{os.getpid()}" >nul
if not errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)
echo 程序已退出，正在重命名...
if exist "{new_path}" del /f "{new_path}"
rename "{exe_path}" "{_EXPECTED_EXE_NAME}"
if errorlevel 1 (
    echo 重命名失败！
    pause
    exit /b 1
)
echo 重命名成功，正在重新启动...
start "" "{new_path}"
del "%~f0"
'''

    # 将 bat 写到 exe 同目录下（确保有权限）
    bat_path = os.path.join(exe_dir, "_rename_fix.bat")
    try:
        with open(bat_path, "w", encoding="gbk", errors="replace") as f:
            f.write(bat_content)

        # 启动 bat 脚本（隐藏窗口）
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        subprocess.Popen(
            ["cmd.exe", "/c", bat_path],
            startupinfo=startupinfo,
            cwd=exe_dir,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        # 如果 bat 方案失败，至少告诉用户手动改名
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"自动重命名失败：{e}\n\n"
                f"请手动将文件重命名为：{_EXPECTED_EXE_NAME}",
                "皮皮贴图修改器 - 重命名失败",
                0x10  # MB_ICONERROR
            )
        except Exception:
            pass

    # 退出当前进程，让 bat 脚本完成重命名
    sys.exit(0)


# ── 跳版警告最低阈值（v0.8.10 加固版起启用） ──
# updater_generation.txt 中记录的版本 < 该值 → 判定为跳版用户，弹提醒。
# 修改准则：只有当再次发生破坏性架构变更、需要提醒老用户重新完整安装时，
# 才把这个阈值向上拉；平时保持不变。
_MIN_UPDATER_GENERATION = "0.8.10"


def _parse_version_tuple(v: str) -> tuple:
    """
    简易版本号解析，用于跳版警告的比较。
    "0.8.10" → (0, 8, 10)；解析失败返回 (0, 0, 0)。
    仅在 launcher 内使用，避免引入 app/updater 依赖。
    """
    v = (v or "").strip().lstrip("vV")
    parts = []
    for seg in v.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _check_updater_generation():
    """
    读取 %APPDATA%\\PPTextureEditor\\updater_generation.txt，判断当前用户是否
    可能"跳版"到本版本（漏装了 v0.8.10 加固版）。

    判定规则（宽松版，考虑 360/杀软误删的风险）：
      · 文件不存在                → 弹跳版警告（不阻断）
      · 内容为空 / 无法解析        → 弹跳版警告（不阻断）
      · 版本值 < 0.8.10           → 弹跳版警告（不阻断）
      · 版本值 >= 0.8.10          → 静默通过

    仅在打包环境（frozen）下生效；开发环境跳过（避免开发时反复弹窗）。
    任何异常都吞掉，绝不阻断启动。
    """
    if not getattr(sys, 'frozen', False):
        return

    try:
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return  # 环境异常，不弹（避免误报）
        gen_path = os.path.join(appdata, "PPTextureEditor", "updater_generation.txt")

        recorded = None
        if os.path.isfile(gen_path):
            try:
                with open(gen_path, "r", encoding="utf-8") as f:
                    recorded = f.read().strip()
            except Exception:
                recorded = None

        min_tuple = _parse_version_tuple(_MIN_UPDATER_GENERATION)
        need_warn = False
        detail = ""

        if not recorded:
            need_warn = True
            detail = "（未检测到升级历史记录）"
        else:
            recorded_tuple = _parse_version_tuple(recorded)
            if recorded_tuple < min_tuple:
                need_warn = True
                detail = f"（记录版本 {recorded}，低于加固版基线 {_MIN_UPDATER_GENERATION}）"

        if not need_warn:
            return

        msg = (
            "检测到你可能跳过了 v0.8.10 加固版直接升级到当前版本。\n\n"
            f"{detail}\n\n"
            "如果程序启动或使用过程中出现异常，请前往 GitHub Release 页面\n"
            "下载完整安装包 PPTextureEditor_vX.Y.Z.zip 并覆盖当前安装目录。\n\n"
            "（点击\"我知道了\"可继续使用，本提示不影响正常功能）"
        )
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                msg,
                "皮皮贴图修改器 - 升级提醒",
                0x40  # MB_ICONINFORMATION（提醒性质，不用 ICONERROR）
            )
        except Exception:
            pass
    except Exception:
        # 跳版警告本身出错，绝不阻断启动
        pass


def _check_app_integrity(app_dir: str):
    """
    启动自检：验证 app/ 目录完整性 + 更新协议兼容性。

    检查项：
      1) app/ 目录本身存在（否则视为拆包结构被破坏）
      2) _EXPECTED_APP_MODULES 中列出的每个 .py 文件都真实存在
      3) version.py 可以正常导入，具备 __version__ 属性

    仅在打包环境（frozen）下强校验；开发环境下不做拦截，避免影响本地调试。
    任何一项失败：写日志 → 弹窗提示"安装损坏，请重新完整安装" → 退出。
    这样避免半损坏状态下让用户看到 Python traceback，也避免自动更新失败后
    程序进入不可用死循环。
    """
    # 开发环境不做强校验，只在打包后才拦截
    if not getattr(sys, 'frozen', False):
        return

    problems = []

    # ── 检查 1：app/ 目录本身 ──
    if not os.path.isdir(app_dir) or os.path.basename(app_dir) != "app":
        problems.append(f"缺少业务代码目录 app/（当前定位到：{app_dir}）")
    else:
        # ── 检查 2：清单里的每个 .py 都要在 ──
        missing = [
            name for name in _EXPECTED_APP_MODULES
            if not os.path.isfile(os.path.join(app_dir, name))
        ]
        if missing:
            problems.append("以下核心模块缺失：\n  - " + "\n  - ".join(missing))

    # ── 检查 3：version.py 可以导入且能拿到 __version__ ──
    if not problems:
        try:
            version_mod = importlib.import_module("version")
            _ = getattr(version_mod, "__version__")
        except Exception as e:
            problems.append(f"version.py 无法导入：{type(e).__name__}: {e}")

    if not problems:
        return  # 全部通过

    # ── 检查失败：写日志 + 弹窗 + 退出 ──
    detail = (
        "[启动自检失败]\n"
        f"app 目录: {app_dir}\n"
        f"exe 路径: {sys.executable}\n"
        + "\n".join(problems)
    )
    log_path = _write_log(detail)

    user_msg = (
        "程序检测到安装文件不完整，无法启动。\n\n"
        "问题详情：\n"
        f"{chr(10).join(problems)}\n\n"
        "解决方法：\n"
        "  1. 请前往 GitHub Release 页面下载最新的完整安装包；\n"
        "  2. 完整覆盖当前目录后重启程序。\n\n"
        f"错误日志已保存至：\n{log_path}"
    )
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            user_msg,
            "皮皮贴图修改器 - 安装损坏",
            0x10  # MB_ICONERROR
        )
    except Exception:
        pass

    sys.exit(2)


def main():
    try:
        # ── 检查 exe 文件名是否被篡改 ──
        _check_exe_name()

        # ── 定位 app/ 业务代码目录并加入 sys.path ──
        app_dir = _get_app_dir()
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)

        # ── 启动自检：核对 app/ 完整性（缺文件直接拦截，避免半损坏启动） ──
        _check_app_integrity(app_dir)

        # ── 检测上次更新是否中途中断，如果是则尝试恢复 ──
        try:
            updater_mod = importlib.import_module("updater")
            updater_mod.recover_interrupted_update()
        except Exception:
            pass  # updater 模块不存在或恢复失败，不影响正常启动

        # ── 清理上次更新遗留的旧版本文件夹 ──
        try:
            updater_mod = importlib.import_module("updater")
            updater_mod.cleanup_old_version()
        except Exception:
            pass  # updater 模块不存在或清理失败，不影响正常启动

        # ── 写入代际标记（跳版警告机制的"写入端"） ──
        # 只要新版 updater 启动过一次，就会在 %APPDATA% 下留一份版本记录，
        # 供未来跨版本升级时的 launcher 判断是否是"跳版用户"。
        try:
            updater_mod = importlib.import_module("updater")
            updater_mod.write_updater_generation()
        except Exception:
            pass  # 写入失败不影响正常启动

        # ── 跳版警告检测（不阻断启动） ──
        # 若判定为跳版用户，弹一条提示后照常继续，不影响主流程。
        # 放在 write_updater_generation 之后是为了：本次启动一次后，
        # 下次启动就不会再弹（因为文件已被写入当前版本号）。
        _check_updater_generation()

        # Windows 任务栏图标：设置 AppUserModelID，让系统把 exe 图标正确关联到任务栏
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "PiPi.TextureEditor.皮皮贴图修改器"
                )
            except Exception:
                pass

        # 切换工作目录到 app/ 目录，确保业务代码中的相对路径（如 bug.svg）正确
        os.chdir(app_dir)

        # 环境诊断信息（写入日志方便排查）
        print(f"[启动] 基础目录: {_BASE_DIR}", flush=True)
        print(f"[启动] 业务代码目录: {app_dir}", flush=True)
        print(f"[启动] Python: {sys.version}", flush=True)
        print(f"[启动] 可执行文件: {sys.executable}", flush=True)

        # 分步导入，精确定位哪个环节出问题
        print("[启动] 正在导入 PySide6...", flush=True)
        import PySide6
        print(f"[启动] PySide6 版本: {PySide6.__version__}", flush=True)

        print("[启动] 正在导入主模块...", flush=True)
        main_mod = importlib.import_module("Texture_tool_GUI_with_tabs")
        app_main = getattr(main_mod, "main")

        print("[启动] 导入完成，正在启动 GUI...", flush=True)
        app_main()

    except SystemExit:
        # sys.exit() 是正常退出，不当作错误处理
        raise

    except Exception:
        exc_text = traceback.format_exc()
        # 控制台输出
        print(f"\n[启动失败]\n{exc_text}", file=sys.stderr, flush=True)
        log_path = _write_log(exc_text)
        _show_fatal_dialog(exc_text, log_path)
        # 如果有控制台窗口，暂停一下让用户能看到
        try:
            input("\n按回车键退出...")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
