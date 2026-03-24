"""
updater.py —— GitHub Release 在线更新模块
负责：检查新版本、下载 zip、解压替换、重启应用。

适用于 PyInstaller 文件夹模式打包（--onedir），
Release Asset 为 zip 压缩包。

★ 发布提醒：上传 Release Asset 时请使用英文文件名（如 PPTextureEditor_vX.X.X.zip），
  避免 PowerShell 下 gh CLI 中文文件名被截断的问题。
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
import hashlib
import logging
from urllib.request import urlopen, Request
from urllib.error import URLError

from version import __version__

# 更新模块日志（仅在开发调试时有用）
_log = logging.getLogger("updater")

# ====================================================================
# ★ 配置区 —— 发布前请修改为你的真实 GitHub 用户名和仓库名 ★
# ★ 上传 Release Asset 时请统一使用英文文件名，例如：
# ★   PPTextureEditor_v0.8.0.zip
# ★ 不要使用中文文件名，否则 gh CLI 在 PowerShell 中会截断中文部分。
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

# 更新锁文件（用于检测更新中途中断）
_UPDATE_LOCK_FILE = "_update_in_progress.lock"

# API 请求缓存（避免同一 IP 下多次请求触发 GitHub 限流）
_update_check_cache = {
    "result": None,
    "timestamp": 0,
    "ttl": 600,  # 缓存有效期 10 分钟
}

# 镜像健康度追踪（记录每个镜像最近的成功/失败情况，用于动态排序）
_mirror_health = {}  # key: 镜像前缀, value: {"success": int, "fail": int, "last_ok": float}


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


# ── 内部：镜像健康度管理 ─────────────────────────────────────────
def _record_mirror_result(prefix: str, success: bool):
    """记录镜像请求结果，用于后续动态排序"""
    if prefix not in _mirror_health:
        _mirror_health[prefix] = {"success": 0, "fail": 0, "last_ok": 0}
    if success:
        _mirror_health[prefix]["success"] += 1
        _mirror_health[prefix]["last_ok"] = time.time()
    else:
        _mirror_health[prefix]["fail"] += 1


def _sort_mirrors_by_health(mirrors: list[str]) -> list[str]:
    """根据镜像健康度动态排序，优先使用成功率高的镜像"""
    def score(prefix):
        h = _mirror_health.get(prefix)
        if h is None:
            return 0  # 未使用过的排中间
        total = h["success"] + h["fail"]
        if total == 0:
            return 0
        # 成功率 × 100 + 最近成功时间的新鲜度
        rate = h["success"] / total * 100
        recency = min(h["last_ok"] / 1e9, 1)  # 归一化
        return rate + recency
    return sorted(mirrors, key=score, reverse=True)


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


# ── 内部：通过 curl HEAD 请求获取文件大小 ─────────────────────────
def _get_content_length_via_curl(url: str, timeout: int = 15) -> int:
    """
    通过 curl.exe 发送 HEAD 请求获取文件大小（Content-Length）。
    用于在 curl 下载前获知文件总大小，以便显示百分比进度。
    返回 0 表示无法获取。
    """
    if not _CURL_PATH:
        return 0
    try:
        result = subprocess.run(
            [_CURL_PATH, "-sI", "-L", "--max-time", str(timeout),
             "-H", "User-Agent: PPEditor-Updater",
             url],
            capture_output=True, timeout=timeout + 10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            return 0
        headers = result.stdout.decode("utf-8", errors="replace")
        for line in headers.splitlines():
            if line.lower().startswith("content-length:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    except Exception as e:
        _log.debug("curl HEAD 请求 %s 失败: %s", url, e)
    return 0


def _download_via_curl(url: str, save_path: str, timeout: int = 600,
                       progress_callback=None, stop_event=None,
                       total_size: int = 0) -> bool:
    """
    通过系统 curl.exe 下载文件。
    支持进度回调和取消。通过监控文件大小提供实时下载进度信息。
    
    参数:
        total_size: 文件总大小（字节），如果提供则可计算百分比和剩余时间
    
    progress_callback 接收 dict 参数：
        {
            "downloaded": int,        # 已下载字节数
            "total": int,             # 文件总大小（0 表示未知）
            "speed": float,           # 当前下载速度 (字节/秒)
            "elapsed": float,         # 已用时间 (秒)
            "downloaded_str": str,    # 已下载大小（人类可读）
            "total_str": str,         # 总大小（人类可读）
            "speed_str": str,         # 速度（人类可读）
            "elapsed_str": str,       # 已用时间（人类可读）
            "eta_str": str,           # 预计剩余时间（人类可读）
            "percent": int,           # 百分比（仅当 total_size > 0 时）
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
            
            # 计算预计剩余时间
            eta_str = "计算中..."
            if speed > 0 and total_size > 0:
                remaining = (total_size - current_size) / speed
                eta_str = _format_time(remaining)
            
            # 回调进度信息
            if progress_callback and current_size > last_size:
                last_size = current_size
                info = {
                    "downloaded": current_size,
                    "total": total_size,
                    "speed": speed,
                    "elapsed": elapsed,
                    "downloaded_str": _format_size(current_size),
                    "total_str": _format_size(total_size) if total_size > 0 else "未知",
                    "speed_str": _format_size(int(speed)) + "/s" if speed > 0 else "计算中...",
                    "elapsed_str": _format_time(elapsed),
                    "eta_str": eta_str,
                }
                if total_size > 0:
                    info["percent"] = int(current_size * 100 / total_size)
                progress_callback(info)
        
        if proc.returncode == 0 and os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            # 下载完成，发送最终进度
            final_size = os.path.getsize(save_path)
            total_time = time.time() - start_time
            avg_speed = final_size / total_time if total_time > 0 else 0
            if progress_callback:
                info = {
                    "downloaded": final_size,
                    "total": total_size if total_size > 0 else final_size,
                    "speed": avg_speed,
                    "elapsed": total_time,
                    "downloaded_str": _format_size(final_size),
                    "total_str": _format_size(total_size) if total_size > 0 else _format_size(final_size),
                    "speed_str": _format_size(int(avg_speed)) + "/s" if avg_speed > 0 else "",
                    "elapsed_str": _format_time(total_time),
                    "eta_str": "0 秒",
                    "done": True,
                }
                if total_size > 0:
                    info["percent"] = 100
                progress_callback(info)
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


# ── 下载文件完整性校验 ────────────────────────────────────────────
def _verify_zip_integrity(zip_path: str) -> bool:
    """
    校验下载的 zip 文件完整性：
    1. 验证是有效的 zip 文件格式
    2. 尝试读取 zip 中的文件列表（检测损坏）
    
    返回 True 表示文件完好，False 表示损坏。
    """
    try:
        if not os.path.isfile(zip_path):
            return False
        file_size = os.path.getsize(zip_path)
        if file_size < 1024:  # zip 文件不应该小于 1KB
            _log.debug("zip 文件过小 (%d bytes)，可能损坏", file_size)
            return False
        if not zipfile.is_zipfile(zip_path):
            _log.debug("文件不是有效的 zip 格式")
            return False
        # 尝试读取文件列表，检测内部结构完整性
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                _log.debug("zip 文件中存在损坏的成员: %s", bad)
                return False
            # 确保包含至少一个文件
            if len(zf.namelist()) == 0:
                _log.debug("zip 文件为空")
                return False
        return True
    except (zipfile.BadZipFile, Exception) as e:
        _log.debug("zip 完整性校验失败: %s", e)
        return False


def _compute_sha256(file_path: str) -> str:
    """计算文件的 SHA256 哈希值"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ── 检查更新（带镜像 + 重试 + 缓存）─────────────────────────────
def check_for_update(force: bool = False) -> dict | None:
    """
    检查 GitHub 上是否有新版本。
    
    策略：依次尝试多个 API 镜像地址，全部失败后等待几秒重试，
    最多循环 _MAX_RETRIES 轮，尽最大努力获取版本信息。
    
    参数:
        force: 是否强制刷新（忽略缓存），用于手动检查更新
    
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
    # ── 缓存检查（防止短时间内重复请求 GitHub API 触发限流）──
    now = time.time()
    if not force and _update_check_cache["result"] is not None:
        if now - _update_check_cache["timestamp"] < _update_check_cache["ttl"]:
            _log.debug("使用缓存的更新检查结果（%d 秒前）",
                       int(now - _update_check_cache["timestamp"]))
            return _update_check_cache["result"]

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
        # 缓存"无结果"状态，避免频繁重试
        _update_check_cache["result"] = None
        _update_check_cache["timestamp"] = now
        return None  # 所有镜像 × 所有轮次均失败

    # ── 解析版本信息 ──
    try:
        latest_tag = data.get("tag_name", "")
        if not latest_tag:
            return None

        latest_ver = _parse_version(latest_tag)
        current_ver = _parse_version(__version__)

        if latest_ver <= current_ver:
            # 缓存"已是最新"结果
            _update_check_cache["result"] = None
            _update_check_cache["timestamp"] = now
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

        result = {
            "version": latest_tag.lstrip("vV"),
            "download_url": download_url,
            "changelog": data.get("body", "") or "暂无更新说明",
            "asset_name": asset_name,
        }
        # 缓存有新版本的结果
        _update_check_cache["result"] = result
        _update_check_cache["timestamp"] = now
        return result

    except Exception:
        return None


# ── 下载并应用更新 ────────────────────────────────────────────────
class UpdateCancelledError(Exception):
    """用户取消更新时抛出的异常"""
    pass


def _build_mirror_urls(original_url: str) -> list[str]:
    """
    根据原始 GitHub 下载链接，生成包含多个镜像的下载地址列表。
    会根据镜像健康度动态排序，优先使用成功率高的镜像。
    """
    urls = []
    sorted_mirrors = _sort_mirrors_by_health(_DOWNLOAD_MIRRORS)
    for prefix in sorted_mirrors:
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
    下载完成后会验证 zip 文件完整性。
    
    参数:
        download_url: zip 的下载链接（原始 GitHub 链接）
        progress_callback: 可选，回调函数。
            - curl 下载时：callback(dict) 传递详细进度信息
              dict 包含 downloaded, speed, elapsed, percent, eta_str 等字段
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
        # 确定当前使用的镜像前缀
        current_prefix = ""
        for prefix in _DOWNLOAD_MIRRORS:
            if prefix and url.startswith(prefix):
                current_prefix = prefix
                break

        # 检查是否被取消
        if stop_event and stop_event.is_set():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise UpdateCancelledError("用户取消了下载")

        # 方案1：优先用 curl.exe 下载
        # 先通过 HEAD 请求获取文件总大小，用于显示百分比进度
        total_size = _get_content_length_via_curl(url, timeout=10)
        _log.debug("文件总大小: %d bytes (%s)", total_size, _format_size(total_size))

        if _download_via_curl(url, zip_path, timeout=600,
                              progress_callback=progress_callback,
                              stop_event=stop_event,
                              total_size=total_size):
            # 验证下载文件完整性
            if _verify_zip_integrity(zip_path):
                _log.debug("curl 下载成功且 zip 完整: %s", url)
                _record_mirror_result(current_prefix, True)
                return zip_path
            else:
                _log.debug("curl 下载的 zip 文件损坏，尝试下一个镜像: %s", url)
                _record_mirror_result(current_prefix, False)
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except Exception:
                        pass
                continue

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

            # 下载成功，验证文件完整性
            if _verify_zip_integrity(zip_path):
                _log.debug("urllib 下载成功且 zip 完整: %s (%d bytes)", url, os.path.getsize(zip_path))
                _record_mirror_result(current_prefix, True)
                return zip_path
            else:
                _log.debug("urllib 下载的 zip 文件损坏: %s", url)
                _record_mirror_result(current_prefix, False)
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except Exception:
                        pass
                continue

        except UpdateCancelledError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception as e:
            _log.debug("urllib 下载失败 %s: %s", url, e)
            last_error = e
            _record_mirror_result(current_prefix, False)
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
    解压 zip 并通过 bat 脚本延迟替换当前程序目录，然后重启。
    
    更新策略（文件夹模式 - bat 脚本延迟替换）：
    1. 解压 zip 到临时目录
    2. 验证解压内容包含有效的 exe 文件
    3. 生成 bat 脚本，该脚本会：
       a. 等待当前进程退出
       b. 备份旧文件到 _old_version_backup/
       c. 复制新文件到应用目录
       d. 启动新版本 exe
       e. 清理临时文件
    4. 启动 bat 脚本并退出当前进程
    
    这种方案解决了「exe 占用无法移动/覆盖」的问题，
    因为 bat 脚本会等待当前进程完全退出后再操作文件。
    
    参数:
        zip_path: 下载好的 zip 文件路径
        progress_callback: 可选，回调函数 callback(percent: int, stage: str)
                          percent 范围 0-100，stage 为当前阶段描述
    
    返回值:
        成功时返回 bat 脚本路径（由主线程启动）。
        失败时返回 False。
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
    lock_file = os.path.join(app_dir, _UPDATE_LOCK_FILE)

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

        # ── 智能检测 zip 内的根目录层级 ──
        # 策略：从解压目录向下查找，直到找到包含 .exe 文件的层级
        _progress(36, "正在验证更新包...")
        source_dir = extract_dir
        entries = os.listdir(extract_dir)

        # 如果解压后只有一个子目录，检查里面是否有 exe
        if len(entries) == 1:
            single = os.path.join(extract_dir, entries[0])
            if os.path.isdir(single):
                # 检查这个子目录里是否有 exe 文件
                has_exe = any(f.lower().endswith(".exe") for f in os.listdir(single)
                             if os.path.isfile(os.path.join(single, f)))
                if has_exe:
                    source_dir = single
                else:
                    # 再往下一层检查（最多两层）
                    sub_entries = os.listdir(single)
                    if len(sub_entries) == 1:
                        sub_single = os.path.join(single, sub_entries[0])
                        if os.path.isdir(sub_single):
                            has_exe_inner = any(
                                f.lower().endswith(".exe")
                                for f in os.listdir(sub_single)
                                if os.path.isfile(os.path.join(sub_single, f))
                            )
                            if has_exe_inner:
                                source_dir = sub_single

        # 最终验证：确认 source_dir 中确实有 exe 文件
        exe_files = [f for f in os.listdir(source_dir)
                     if f.lower().endswith(".exe") and not f.startswith("_")
                     and os.path.isfile(os.path.join(source_dir, f))]
        if not exe_files:
            _log.error("更新包中未找到有效的 exe 文件")
            _progress(0, "更新包验证失败：未找到可执行文件")
            return False

        # ② 确定新版本的 exe 文件名
        current_exe_name = os.path.basename(sys.executable)
        if current_exe_name in exe_files:
            new_exe_name = current_exe_name
        else:
            new_exe_name = exe_files[0]

        _progress(40, "正在准备更新脚本...")

        # ③ 写入更新锁文件（用于崩溃恢复检测）
        try:
            with open(lock_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": time.time(),
                    "source_dir": source_dir,
                    "backup_dir": backup_dir,
                    "new_exe_name": new_exe_name,
                    "version": __version__,
                }, f)
        except Exception as e:
            _log.debug("写入锁文件失败: %s", e)

        # ④ 生成 bat 更新脚本
        # bat 脚本会在当前进程退出后执行文件替换
        _progress(50, "正在生成更新脚本...")

        bat_path = os.path.join(tempfile.gettempdir(), "ppeditor_update.bat")
        current_pid = os.getpid()

        # 需要跳过的目录/文件
        skip_items = [
            OLD_BACKUP_DIR_NAME,
            _UPDATE_LOCK_FILE,
            "error_log.txt",
            ".git",
            "__pycache__",
        ]

        # 生成跳过条件的 bat 语法
        skip_conditions = ""
        for item in skip_items:
            skip_conditions += f'    if "%%f"=="{item}" goto :skip_backup\n'

        # 生成 bat 脚本内容
        bat_content = f'''@echo off
chcp 65001 >nul 2>&1
title 皮皮贴图修改器 - 正在更新...
echo ========================================
echo   皮皮贴图修改器 - 自动更新
echo ========================================
echo.

:: 等待旧进程退出（最多等 30 秒）
echo [1/5] 等待旧版本退出...
set /a count=0
:wait_loop
tasklist /FI "PID eq {current_pid}" 2>nul | find "{current_pid}" >nul 2>&1
if errorlevel 1 goto :process_exited
set /a count+=1
if %count% geq 60 (
    echo 等待超时，强制继续...
    goto :process_exited
)
timeout /t 1 /nobreak >nul
goto :wait_loop

:process_exited
echo 旧版本已退出。
echo.

:: 备份旧文件
echo [2/5] 备份旧版本文件...
if exist "{backup_dir}" rmdir /s /q "{backup_dir}" >nul 2>&1
mkdir "{backup_dir}" >nul 2>&1

for %%f in ("{app_dir}\\*") do (
{skip_conditions}    move /y "%%f" "{backup_dir}\\" >nul 2>&1
    :skip_backup
)
for /d %%f in ("{app_dir}\\*") do (
    if "%%~nxf"=="{OLD_BACKUP_DIR_NAME}" goto :skip_dir
    if "%%~nxf"==".git" goto :skip_dir
    if "%%~nxf"=="__pycache__" goto :skip_dir
    move /y "%%f" "{backup_dir}\\" >nul 2>&1
    :skip_dir
)
echo 备份完成。
echo.

:: 复制新文件
echo [3/5] 安装新版本...
xcopy /s /e /y /q "{source_dir}\\*" "{app_dir}\\" >nul 2>&1
if errorlevel 1 (
    echo 复制新文件失败！正在回滚...
    goto :rollback
)
echo 安装完成。
echo.

:: 删除锁文件
echo [4/5] 清理临时文件...
del /f /q "{lock_file}" >nul 2>&1
echo 清理完成。
echo.

:: 启动新版本
echo [5/5] 启动新版本...
start "" "{os.path.join(app_dir, new_exe_name)}"
echo.
echo ========================================
echo   更新完成！新版本已启动。
echo   此窗口将在 3 秒后自动关闭。
echo ========================================
timeout /t 3 /nobreak >nul

:: 清理临时解压目录
rmdir /s /q "{extract_dir}" >nul 2>&1
:: 删除自身
del /f /q "%~f0" >nul 2>&1
exit /b 0

:rollback
echo 正在回滚到旧版本...
for %%f in ("{backup_dir}\\*") do (
    move /y "%%f" "{app_dir}\\" >nul 2>&1
)
for /d %%f in ("{backup_dir}\\*") do (
    move /y "%%f" "{app_dir}\\" >nul 2>&1
)
del /f /q "{lock_file}" >nul 2>&1
echo 回滚完成。正在重新启动旧版本...
start "" "{os.path.join(app_dir, current_exe_name)}"
timeout /t 3 /nobreak >nul
del /f /q "%~f0" >nul 2>&1
exit /b 1
'''

        with open(bat_path, "w", encoding="gbk", errors="replace") as f:
            f.write(bat_content)

        _progress(90, "更新脚本已就绪...")
        _progress(100, "更新完成，即将重启...")

        # 返回 bat 脚本路径，由主线程启动后退出
        return bat_path

    except Exception as e:
        _log.error("apply_update 失败: %s", e)
        _progress(0, "更新失败...")
        # 清理锁文件
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
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


# ── 崩溃恢复检测（启动时调用）────────────────────────────────────
def recover_interrupted_update() -> bool:
    """
    检测上次更新是否中途中断（通过锁文件判断）。
    如果检测到中断，尝试从备份中恢复旧版本。
    
    应在每次启动时尽早调用（在 cleanup_old_version 之前）。
    返回 True 表示检测到中断并已尝试恢复。
    """
    app_dir = get_app_dir()
    lock_file = os.path.join(app_dir, _UPDATE_LOCK_FILE)
    backup_dir = os.path.join(app_dir, OLD_BACKUP_DIR_NAME)

    if not os.path.exists(lock_file):
        return False

    _log.warning("检测到更新锁文件，上次更新可能中途中断")

    try:
        # 读取锁文件信息
        with open(lock_file, "r", encoding="utf-8") as f:
            lock_info = json.load(f)
        _log.warning("中断的更新信息: %s", lock_info)
    except Exception:
        lock_info = {}

    # 尝试从备份中恢复
    if os.path.exists(backup_dir) and os.listdir(backup_dir):
        _log.warning("发现备份目录，尝试恢复旧版本...")
        try:
            for item in os.listdir(backup_dir):
                src = os.path.join(backup_dir, item)
                dst = os.path.join(app_dir, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            _log.warning("旧版本恢复完成")
        except Exception as e:
            _log.error("恢复旧版本失败: %s", e)

    # 清理锁文件
    try:
        os.remove(lock_file)
    except Exception:
        pass

    return True


# ── 清理旧版本备份（由 launcher.py 在启动时调用）─────────────────
def cleanup_old_version():
    """
    清理上次更新遗留的旧版本备份文件夹。
    应在每次启动时尽早调用（在 recover_interrupted_update 之后）。
    """
    app_dir = get_app_dir()
    backup_dir = os.path.join(app_dir, OLD_BACKUP_DIR_NAME)
    if os.path.exists(backup_dir):
        try:
            shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:
            pass
