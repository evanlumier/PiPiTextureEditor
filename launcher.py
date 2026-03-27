"""
launcher.py —— 轻量级启动器
作为打包入口，在导入主模块之前提供最外层异常兜底。
当主模块 import 失败或初始化阶段崩溃时，用标准库弹窗提示用户并写入日志，
而不是直接无响应退出。
"""
import sys
import os
import traceback
import subprocess
from datetime import datetime

# ── exe 文件名白名单配置 ──
_EXPECTED_EXE_NAME = "皮皮贴图修改器.exe"


def _get_base_dir() -> str:
    """获取 exe/脚本所在目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


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
    bat_content = f'''@echo off
chcp 65001 >nul
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
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(bat_content)

        # 启动 bat 脚本（隐藏窗口）
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        subprocess.Popen(
            ["cmd.exe", "/c", bat_path],
            startupinfo=startupinfo,
            cwd=exe_dir,
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


def main():
    try:
        # ── 检查 exe 文件名是否被篡改 ──
        _check_exe_name()

        # ── 检测上次更新是否中途中断，如果是则尝试恢复 ──
        try:
            from updater import recover_interrupted_update
            recover_interrupted_update()
        except Exception:
            pass  # updater 模块不存在或恢复失败，不影响正常启动

        # ── 清理上次更新遗留的旧版本文件夹 ──
        try:
            from updater import cleanup_old_version
            cleanup_old_version()
        except Exception:
            pass  # updater 模块不存在或清理失败，不影响正常启动

        # Windows 任务栏图标：设置 AppUserModelID，让系统把 exe 图标正确关联到任务栏
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "PiPi.TextureEditor.皮皮贴图修改器"
                )
            except Exception:
                pass

        # 切换工作目录，确保相对路径正确
        os.chdir(_BASE_DIR)

        # 环境诊断信息（写入日志方便排查）
        print(f"[启动] 基础目录: {_BASE_DIR}", flush=True)
        print(f"[启动] Python: {sys.version}", flush=True)
        print(f"[启动] 可执行文件: {sys.executable}", flush=True)

        # 分步导入，精确定位哪个环节出问题
        print("[启动] 正在导入 PySide6...", flush=True)
        import PySide6
        print(f"[启动] PySide6 版本: {PySide6.__version__}", flush=True)

        print("[启动] 正在导入主模块...", flush=True)
        from Texture_tool_GUI_with_tabs import main as app_main

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
