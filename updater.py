"""
updater.py —— GitHub Release 在线更新模块
负责：检查新版本、下载 zip、解压替换、重启应用。

适用于 PyInstaller 文件夹模式打包（--onedir），
Release Asset 为 zip 压缩包。
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
import tempfile
from urllib.request import urlopen, Request
from urllib.error import URLError

from version import __version__

# ====================================================================
# ★ 配置区 —— 发布前请修改为你的真实 GitHub 用户名和仓库名 ★
# ====================================================================
GITHUB_OWNER = "evanlumier"
GITHUB_REPO = "PiPiTextureEditor"
ASSET_SUFFIX = ".zip"                   # Release Asset 必须是 zip
API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# 旧版本备份文件夹名（更新时把旧文件移到这里，下次启动时清理）
OLD_BACKUP_DIR_NAME = "_old_version_backup"


# ── 版本比较工具 ──────────────────────────────────────────────────
def _parse_version(v: str) -> tuple:
    """
    将版本字符串解析为可比较的元组。
    支持 '0.7.0', 'v0.7.0', '1.0' 等格式。
    """
    v = v.strip().lstrip("vV")
    parts = []
    for seg in v.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    # 补齐到至少 3 段
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# ── 公共工具 ──────────────────────────────────────────────────────
def get_app_dir() -> str:
    """获取当前应用（exe / 脚本）所在目录"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ── 检查更新 ─────────────────────────────────────────────────────
def check_for_update() -> dict | None:
    """
    检查 GitHub 上是否有新版本。
    
    返回值:
        有新版本时返回 dict:
            {
                "version": "0.8.0",
                "download_url": "https://github.com/.../xxx.zip",
                "changelog": "更新说明...",
                "asset_name": "PPEditor_v0.8.0.zip"
            }
        无更新或检查失败时返回 None（静默失败，不影响正常使用）。
    """
    try:
        req = Request(API_URL, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PPEditor-Updater",
        })
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        latest_tag = data.get("tag_name", "")
        if not latest_tag:
            return None

        latest_ver = _parse_version(latest_tag)
        current_ver = _parse_version(__version__)

        if latest_ver <= current_ver:
            return None  # 已是最新版本

        # 在 Release Assets 中寻找 zip 文件
        download_url = None
        asset_name = None
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if name.lower().endswith(ASSET_SUFFIX):
                download_url = asset["browser_download_url"]
                asset_name = name
                break

        if not download_url:
            return None  # Release 中没有找到 zip 文件

        return {
            "version": latest_tag.lstrip("vV"),
            "download_url": download_url,
            "changelog": data.get("body", "") or "暂无更新说明",
            "asset_name": asset_name,
        }

    except Exception:
        # 网络超时、DNS 失败、API 限流等各种情况，全部静默跳过
        return None


# ── 下载并应用更新 ────────────────────────────────────────────────
def download_update(download_url: str, progress_callback=None) -> str:
    """
    下载 zip 文件到临时目录。
    
    参数:
        download_url: zip 的下载链接
        progress_callback: 可选，回调函数 callback(percent: int)，percent 范围 0-100
    
    返回值:
        下载好的 zip 文件完整路径
    
    异常:
        下载失败时抛出异常，由调用方处理。
    """
    tmp_dir = tempfile.mkdtemp(prefix="ppeditor_update_")
    zip_path = os.path.join(tmp_dir, "update.zip")

    req = Request(download_url, headers={"User-Agent": "PPEditor-Updater"})
    with urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(int(downloaded * 100 / total))

    return zip_path


def apply_update(zip_path: str) -> bool:
    """
    解压 zip 并替换当前程序目录，然后重启。
    
    更新策略（文件夹模式）：
    1. 将当前目录中的文件移到 _old_version_backup/ 备份
    2. 将 zip 中的文件解压到当前目录
    3. 启动新版本的 exe
    4. 退出当前进程
    5. 下次启动时 launcher.py 负责清理 _old_version_backup/
    
    返回值:
        成功启动新版本后不会返回（进程已退出）。
        失败时返回 False 并尝试回滚。
    """
    if not getattr(sys, "frozen", False):
        # 开发环境下不执行自更新
        return False

    app_dir = get_app_dir()
    backup_dir = os.path.join(app_dir, OLD_BACKUP_DIR_NAME)
    extract_dir = tempfile.mkdtemp(prefix="ppeditor_extract_")

    try:
        # ① 解压 zip 到临时目录
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # zip 内可能有一层根目录，也可能直接是文件
        # 检测：如果解压后只有一个子目录，就进入那个子目录
        entries = os.listdir(extract_dir)
        if len(entries) == 1:
            single = os.path.join(extract_dir, entries[0])
            if os.path.isdir(single):
                extract_dir = single

        # ② 备份当前目录中的文件（排除备份目录本身和临时文件）
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
        os.makedirs(backup_dir, exist_ok=True)

        # 需要保留的文件/目录（不参与更新替换）
        _skip = {
            OLD_BACKUP_DIR_NAME,
            "error_log.txt",
            ".git",
            "__pycache__",
        }

        for item in os.listdir(app_dir):
            if item in _skip:
                continue
            src = os.path.join(app_dir, item)
            dst = os.path.join(backup_dir, item)
            try:
                if os.path.isdir(src):
                    shutil.move(src, dst)
                else:
                    # exe 自身可能被锁定，跳过移动失败的情况
                    shutil.move(src, dst)
            except PermissionError:
                # 当前正在运行的 exe 可能无法移动，跳过
                pass

        # ③ 将新版本文件复制到应用目录
        for item in os.listdir(extract_dir):
            src = os.path.join(extract_dir, item)
            dst = os.path.join(app_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # ④ 启动新版本
        new_exe = os.path.join(app_dir, os.path.basename(sys.executable))
        if os.path.exists(new_exe):
            subprocess.Popen([new_exe], cwd=app_dir)
            sys.exit(0)
        else:
            # 如果 exe 名变了，尝试找任意 .exe
            for f in os.listdir(app_dir):
                if f.lower().endswith(".exe") and not f.startswith("_"):
                    subprocess.Popen(
                        [os.path.join(app_dir, f)], cwd=app_dir
                    )
                    sys.exit(0)

        return False  # 没找到 exe，不重启

    except Exception:
        # ⑤ 回滚：尝试把备份的文件还原
        try:
            if os.path.exists(backup_dir):
                for item in os.listdir(backup_dir):
                    src = os.path.join(backup_dir, item)
                    dst = os.path.join(app_dir, item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
        except Exception:
            pass
        return False

    finally:
        # 清理下载的 zip 临时目录
        try:
            zip_parent = os.path.dirname(zip_path)
            if zip_parent.startswith(tempfile.gettempdir()):
                shutil.rmtree(zip_parent, ignore_errors=True)
        except Exception:
            pass


# ── 清理旧版本备份（由 launcher.py 在启动时调用）─────────────────
def cleanup_old_version():
    """
    清理上次更新遗留的旧版本备份文件夹。
    应在每次启动时尽早调用。
    """
    app_dir = get_app_dir()
    backup_dir = os.path.join(app_dir, OLD_BACKUP_DIR_NAME)
    if os.path.exists(backup_dir):
        try:
            shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:
            pass
