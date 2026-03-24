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
import time
import ssl
import logging
from urllib.request import urlopen, Request
from urllib.error import URLError

from version import __version__

# 更新模块日志（仅在开发调试时有用）
_log = logging.getLogger("updater")

# ====================================================================
# ★ 配置区 —— 发布前请修改为你的真实 GitHub 用户名和仓库名 ★
# ====================================================================
GITHUB_OWNER = "evanlumier"
GITHUB_REPO = "PiPiTextureEditor"
ASSET_SUFFIX = ".zip"                   # Release Asset 必须是 zip

# ── API 地址列表（按优先级排列，依次尝试直到成功）──
# 官方 GitHub API + 多个国内可用的加速代理站点
_API_URLS = [
    # ① GitHub 官方
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest",
    # ② gh-proxy.org 代理（国内优化线路）
    f"https://gh-proxy.org/https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest",
    # ③ ghfast 代理
    f"https://ghfast.top/https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest",
    # ④ gh-proxy.com 代理
    f"https://gh-proxy.com/https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest",
    # ⑤ llkk.cc 代理
    f"https://gh.llkk.cc/https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest",
]

# ── 下载加速镜像前缀（用于替换 github.com 下载链接）──
_DOWNLOAD_MIRRORS = [
    "",                                     # 空 = 使用原始链接
    "https://gh-proxy.org/",                # gh-proxy.org 加速
    "https://ghfast.top/",                  # ghfast 加速
    "https://gh.llkk.cc/",                   # llkk 加速
    "https://gh-proxy.com/",                # gh-proxy 加速
]

# 重试配置
_MAX_RETRIES = 2          # 整个镜像列表最多循环几轮
_RETRY_DELAY = 2          # 每轮之间等待秒数
_REQUEST_TIMEOUT = 15     # 单次请求超时秒数（公司 SSL 代理 renegotiation 较慢）

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


# ── 内部：查找系统 curl.exe ──────────────────────────────────────────
def _find_curl() -> str | None:
    """
    查找系统自带的 curl.exe 路径。
    Windows 10+ 自带 curl.exe（位于 System32），使用 schannel SSL 引擎，
    能兼容公司 SSL 代理的 renegotiation，比 Python 的 OpenSSL 更稳定。
    """
    # Windows System32 下的 curl.exe
    sys32_curl = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                              "System32", "curl.exe")
    if os.path.isfile(sys32_curl):
        return sys32_curl
    # 尝试 PATH 中查找
    try:
        result = subprocess.run(
            ["where", "curl.exe"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            path = result.stdout.strip().splitlines()[0]
            if os.path.isfile(path):
                return path
    except Exception:
        pass
    return None

# 启动时缓存 curl 路径
_CURL_PATH = _find_curl()


# ── 内部：通过 curl.exe 请求（优先方案）────────────────────────────
def _fetch_json_via_curl(url: str, timeout: int = _REQUEST_TIMEOUT) -> dict | None:
    """
    通过系统 curl.exe 请求 JSON 数据。
    curl.exe 使用 Windows schannel SSL 引擎，能兼容公司 SSL 代理环境
    （OpenSSL 3.0 对 SSL renegotiation 不兼容会卡死，而 schannel 支持）。
    """
    if not _CURL_PATH:
        return None
    try:
        # -w 在输出末尾追加 HTTP 状态码，用于判断是否成功
        result = subprocess.run(
            [_CURL_PATH, "-s", "--max-time", str(timeout),
             "-H", "User-Agent: PPEditor-Updater",
             "-H", "Accept: application/vnd.github.v3+json",
             "-w", "\n__HTTP_CODE__%{http_code}",
             url],
            capture_output=True, timeout=timeout + 10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            return None
        
        raw = result.stdout.decode("utf-8", errors="replace")
        
        # 从输出中分离 HTTP 状态码
        http_code = 0
        marker = "\n__HTTP_CODE__"
        if marker in raw:
            parts = raw.rsplit(marker, 1)
            raw = parts[0]
            try:
                http_code = int(parts[1].strip())
            except ValueError:
                pass
        
        if http_code != 200:
            _log.debug("curl %s 返回 HTTP %d", url, http_code)
            return None
        
        if raw.strip():
            return json.loads(raw)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        _log.debug("curl 请求 %s 失败: %s", url, e)
    return None


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的大小字符串"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def _format_time(seconds: float) -> str:
    """将秒数格式化为人类可读的时间字符串"""
    if seconds < 0 or seconds > 36000:  # 超过10小时视为无效
        return "计算中..."
    if seconds < 60:
        return f"{int(seconds)} 秒"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes} 分 {secs} 秒"


def _download_via_curl(url: str, save_path: str, timeout: int = 600,
                       progress_callback=None, stop_event=None) -> bool:
    """
    通过系统 curl.exe 下载文件。
    支持进度回调和取消。通过监控文件大小提供实时下载进度信息。
    
    progress_callback 接收 dict 参数：
        {
            "downloaded": int,        # 已下载字节数
            "speed": float,           # 当前下载速度 (字节/秒)
            "elapsed": float,         # 已用时间 (秒)
            "downloaded_str": str,    # 已下载大小（人类可读）
            "speed_str": str,         # 速度（人类可读）
            "elapsed_str": str,       # 已用时间（人类可读）
        }
    """
    if not _CURL_PATH:
        return False
    try:
        # curl -L 跟随重定向，-o 输出文件
        # 不使用 stdout=PIPE，避免潜在的缓冲区死锁风险
        proc = subprocess.Popen(
            [_CURL_PATH, "-L", "-s", "--max-time", str(timeout),
             "-H", "User-Agent: PPEditor-Updater",
             "-o", save_path,
             url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        # 通过监控文件大小来提供实时进度信息
        start_time = time.time()
        last_size = 0
        last_speed_time = start_time
        last_speed_size = 0
        speed = 0.0
        
        while proc.poll() is None:
            if stop_event and stop_event.is_set():
                proc.kill()
                return False
            
            time.sleep(0.5)
            
            # 获取当前已下载大小
            current_size = 0
            if os.path.isfile(save_path):
                try:
                    current_size = os.path.getsize(save_path)
                except OSError:
                    pass
            
            # 计算下载速度（每2秒更新一次，避免抖动）
            now = time.time()
            elapsed = now - start_time
            speed_interval = now - last_speed_time
            if speed_interval >= 2.0:
                speed = (current_size - last_speed_size) / speed_interval
                last_speed_time = now
                last_speed_size = current_size
            elif elapsed > 0 and speed == 0:
                # 初始阶段用总平均速度
                speed = current_size / elapsed
            
            # 回调进度信息
            if progress_callback and current_size > last_size:
                last_size = current_size
                progress_callback({
                    "downloaded": current_size,
                    "speed": speed,
                    "elapsed": elapsed,
                    "downloaded_str": _format_size(current_size),
                    "speed_str": _format_size(int(speed)) + "/s" if speed > 0 else "计算中...",
                    "elapsed_str": _format_time(elapsed),
                })
        
        if proc.returncode == 0 and os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            # 下载完成，发送最终进度
            final_size = os.path.getsize(save_path)
            total_time = time.time() - start_time
            avg_speed = final_size / total_time if total_time > 0 else 0
            if progress_callback:
                progress_callback({
                    "downloaded": final_size,
                    "speed": avg_speed,
                    "elapsed": total_time,
                    "downloaded_str": _format_size(final_size),
                    "speed_str": _format_size(int(avg_speed)) + "/s" if avg_speed > 0 else "",
                    "elapsed_str": _format_time(total_time),
                    "done": True,
                })
            return True
    except Exception as e:
        _log.debug("curl 下载 %s 失败: %s", url, e)
    return False


# ── 内部：通过 Python urllib 请求（fallback 方案）──────────────────
def _build_insecure_ssl_context() -> ssl.SSLContext:
    """构建跳过 SSL 验证的上下文（降级兜底方案）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch_json_via_urllib(url: str, timeout: int = _REQUEST_TIMEOUT) -> dict | None:
    """
    通过 Python urllib 请求 JSON 数据（fallback 方案）。
    注意：在某些公司网络下 OpenSSL 3.0 可能因 SSL renegotiation 卡死。
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PPEditor-Updater",
    }
    # 尝试两种 SSL 策略：默认 → 跳过验证
    for ssl_ctx in (None, _build_insecure_ssl_context()):
        try:
            req = Request(url, headers=headers)
            kwargs = {"timeout": timeout}
            if ssl_ctx:
                kwargs["context"] = ssl_ctx
            with urlopen(req, **kwargs) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except ssl.SSLCertVerificationError:
            continue
        except ssl.SSLError:
            continue
        except Exception as e:
            _log.debug("urllib 请求 %s 失败: %s", url, e)
            return None
    return None


# ── 内部：统一的 API 请求（curl 优先 → urllib fallback）─────────────
def _fetch_release_json(api_url: str, timeout: int = _REQUEST_TIMEOUT) -> dict | None:
    """
    向单个 API 地址请求最新 Release 信息。
    优先使用 curl.exe（schannel SSL，兼容性最好），失败则 fallback 到 urllib。
    """
    # 方案1：curl.exe（Windows schannel SSL）
    data = _fetch_json_via_curl(api_url, timeout)
    if data is not None:
        return data
    
    # 方案2：Python urllib（OpenSSL）
    return _fetch_json_via_urllib(api_url, timeout)


# ── 检查更新（带镜像 + 重试）────────────────────────────────────────
def check_for_update() -> dict | None:
    """
    检查 GitHub 上是否有新版本。
    
    策略：依次尝试多个 API 镜像地址，全部失败后等待几秒重试，
    最多循环 _MAX_RETRIES 轮，尽最大努力获取版本信息。
    
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
    data = None

    # 多轮重试，每轮遍历所有镜像
    for attempt in range(_MAX_RETRIES):
        for api_url in _API_URLS:
            data = _fetch_release_json(api_url)
            if data is not None:
                _log.debug("第 %d 轮，通过 %s 获取成功", attempt + 1, api_url)
                break
        if data is not None:
            break
        # 本轮所有镜像都失败了，等待后重试
        if attempt < _MAX_RETRIES - 1:
            _log.debug("第 %d 轮全部失败，%d 秒后重试...", attempt + 1, _RETRY_DELAY)
            time.sleep(_RETRY_DELAY)

    if data is None:
        return None  # 所有镜像 × 所有轮次均失败

    # ── 解析版本信息 ──
    try:
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
                download_url = asset.get("browser_download_url", "")
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
        return None


# ── 下载并应用更新 ────────────────────────────────────────────────
class UpdateCancelledError(Exception):
    """用户取消更新时抛出的异常"""
    pass


def _build_mirror_urls(original_url: str) -> list[str]:
    """
    根据原始 GitHub 下载链接，生成包含多个镜像的下载地址列表。
    """
    urls = []
    for prefix in _DOWNLOAD_MIRRORS:
        if prefix:
            urls.append(prefix + original_url)
        else:
            urls.append(original_url)
    return urls


def download_update(download_url: str, progress_callback=None, stop_event=None) -> str:
    """
    下载 zip 文件到临时目录。
    自动尝试多个镜像下载地址，任一成功即返回。
    优先使用 curl.exe 下载（兼容公司 SSL 代理环境）。
    
    参数:
        download_url: zip 的下载链接（原始 GitHub 链接）
        progress_callback: 可选，回调函数。
            - curl 下载时：callback(dict) 传递详细进度信息
              dict 包含 downloaded, speed, elapsed 等字段
            - urllib 下载时：callback(dict) 同样传递详细进度信息
        stop_event: 可选，threading.Event 对象，set() 后中断下载
    
    返回值:
        下载好的 zip 文件完整路径
    
    异常:
        所有镜像均下载失败时抛出异常，由调用方处理。
        用户取消时抛出 UpdateCancelledError。
    """
    tmp_dir = tempfile.mkdtemp(prefix="ppeditor_update_")
    zip_path = os.path.join(tmp_dir, "update.zip")

    mirror_urls = _build_mirror_urls(download_url)
    last_error = None

    for url in mirror_urls:
        # 检查是否被取消
        if stop_event and stop_event.is_set():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise UpdateCancelledError("用户取消了下载")

        # 方案1：优先用 curl.exe 下载
        if _download_via_curl(url, zip_path, timeout=600,
                              progress_callback=progress_callback,
                              stop_event=stop_event):
            _log.debug("curl 下载成功: %s", url)
            return zip_path

        # 检查是否被取消（curl 下载失败可能是因为取消）
        if stop_event and stop_event.is_set():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise UpdateCancelledError("用户取消了下载")

        # 方案2：fallback 到 urllib
        try:
            _log.debug("curl 下载失败，尝试 urllib: %s", url)
            req = Request(url, headers={"User-Agent": "PPEditor-Updater"})
            ssl_ctx = _build_insecure_ssl_context()
            with urlopen(req, timeout=300, context=ssl_ctx) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                start_time = time.time()
                last_speed_time = start_time
                last_speed_size = 0
                speed = 0.0
                with open(zip_path, "wb") as f:
                    while True:
                        if stop_event and stop_event.is_set():
                            raise UpdateCancelledError("用户取消了下载")
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        now = time.time()
                        elapsed = now - start_time
                        speed_interval = now - last_speed_time
                        if speed_interval >= 2.0:
                            speed = (downloaded - last_speed_size) / speed_interval
                            last_speed_time = now
                            last_speed_size = downloaded
                        elif elapsed > 0 and speed == 0:
                            speed = downloaded / elapsed
                        
                        # 计算预计剩余时间
                        eta_str = "计算中..."
                        if speed > 0 and total > 0:
                            remaining = (total - downloaded) / speed
                            eta_str = _format_time(remaining)
                        
                        if progress_callback:
                            info = {
                                "downloaded": downloaded,
                                "total": total,
                                "speed": speed,
                                "elapsed": elapsed,
                                "downloaded_str": _format_size(downloaded),
                                "total_str": _format_size(total) if total > 0 else "未知",
                                "speed_str": _format_size(int(speed)) + "/s" if speed > 0 else "计算中...",
                                "elapsed_str": _format_time(elapsed),
                                "eta_str": eta_str,
                            }
                            if total > 0:
                                info["percent"] = int(downloaded * 100 / total)
                            progress_callback(info)

            # 下载成功，验证文件大小合理
            if os.path.getsize(zip_path) > 0:
                _log.debug("urllib 下载成功: %s (%d bytes)", url, os.path.getsize(zip_path))
                return zip_path

        except UpdateCancelledError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception as e:
            _log.debug("urllib 下载失败 %s: %s", url, e)
            last_error = e
            # 清理不完整的文件，继续尝试下一个镜像
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    pass
            continue

    # 所有镜像都失败了
    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise last_error or RuntimeError("所有下载镜像均失败")


def apply_update(zip_path: str, progress_callback=None) -> bool:
    """
    解压 zip 并替换当前程序目录，然后重启。
    
    更新策略（文件夹模式）：
    1. 将当前目录中的文件移到 _old_version_backup/ 备份
    2. 将 zip 中的文件解压到当前目录
    3. 启动新版本的 exe
    4. 退出当前进程
    5. 下次启动时 launcher.py 负责清理 _old_version_backup/
    
    参数:
        zip_path: 下载好的 zip 文件路径
        progress_callback: 可选，回调函数 callback(percent: int, stage: str)
                          percent 范围 0-100，stage 为当前阶段描述
    
    返回值:
        成功启动新版本后不会返回（进程已退出）。
        失败时返回 False 并尝试回滚。
    """
    def _progress(percent, stage):
        if progress_callback:
            progress_callback(percent, stage)

    if not getattr(sys, "frozen", False):
        # 开发环境下不执行自更新
        return False

    app_dir = get_app_dir()
    backup_dir = os.path.join(app_dir, OLD_BACKUP_DIR_NAME)
    extract_dir = tempfile.mkdtemp(prefix="ppeditor_extract_")

    try:
        # ① 解压 zip 到临时目录
        _progress(5, "正在解压更新包...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            total_members = len(members)
            for i, member in enumerate(members):
                zf.extract(member, extract_dir)
                # 解压阶段占 5% ~ 35%
                pct = 5 + int((i + 1) / total_members * 30)
                _progress(pct, f"正在解压更新包... ({i+1}/{total_members})")

        # zip 内可能有一层根目录，也可能直接是文件
        # 检测：如果解压后只有一个子目录，就进入那个子目录
        entries = os.listdir(extract_dir)
        if len(entries) == 1:
            single = os.path.join(extract_dir, entries[0])
            if os.path.isdir(single):
                extract_dir = single

        # ② 备份当前目录中的文件（排除备份目录本身和临时文件）
        _progress(36, "正在备份旧版本...")
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

        backup_items = [item for item in os.listdir(app_dir) if item not in _skip]
        total_backup = len(backup_items) if backup_items else 1
        for i, item in enumerate(backup_items):
            src = os.path.join(app_dir, item)
            dst = os.path.join(backup_dir, item)
            try:
                if os.path.isdir(src):
                    shutil.move(src, dst)
                else:
                    shutil.move(src, dst)
            except PermissionError:
                # 当前正在运行的 exe 可能无法移动，跳过
                pass
            # 备份阶段占 36% ~ 60%
            pct = 36 + int((i + 1) / total_backup * 24)
            _progress(pct, f"正在备份旧版本... ({i+1}/{total_backup})")

        # ③ 将新版本文件复制到应用目录
        _progress(61, "正在安装新版本...")
        new_items = os.listdir(extract_dir)
        total_new = len(new_items) if new_items else 1
        for i, item in enumerate(new_items):
            src = os.path.join(extract_dir, item)
            dst = os.path.join(app_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            # 安装阶段占 61% ~ 90%
            pct = 61 + int((i + 1) / total_new * 29)
            _progress(pct, f"正在安装新版本... ({i+1}/{total_new})")

        # ④ 查找新版本 exe 路径（不再在子线程中启动/退出，交给主线程）
        _progress(95, "正在准备重启...")
        new_exe = os.path.join(app_dir, os.path.basename(sys.executable))
        if os.path.exists(new_exe):
            _progress(100, "更新完成，即将重启...")
            return new_exe  # 返回 exe 路径，由主线程启动并退出
        else:
            # 如果 exe 名变了，尝试找任意 .exe
            for f in os.listdir(app_dir):
                if f.lower().endswith(".exe") and not f.startswith("_"):
                    _progress(100, "更新完成，即将重启...")
                    return os.path.join(app_dir, f)

        return False  # 没找到 exe，不重启

    except Exception:
        # ⑤ 回滚：尝试把备份的文件还原
        _progress(0, "更新失败，正在回滚...")
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
