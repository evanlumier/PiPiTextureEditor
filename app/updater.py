"""
updater.py —— GitHub Release 在线更新模块
负责：检查新版本、下载 zip、解压替换、重启应用。

适用于 PyInstaller 文件夹模式打包（--onedir），
Release Asset 为 zip 压缩包。

★ 更新源：GitHub 公开仓库（无需任何认证）
★ 版本检查：通过 raw.githubusercontent.com 读取远程 version.py（秒回，无 rate limit）
★ 文件下载：通过 GitHub Release 直连下载（CDN 分发，稳定快速）
★ 发布提醒：上传 Release Asset 时请使用英文文件名（如 PPTextureEditor_vX.X.X.zip）
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
import tempfile
import time
import re
import ssl
import hashlib
import logging
from urllib.request import urlopen, Request
from urllib.error import URLError

from version import __version__

# —— 更新协议元数据（v0.8.10 加固版起启用） ——
# 说明：老版本的 version.py 可能没有这两个常量，用 try 兜底避免更新逻辑
# 反而先崩掉。缺失时按最保守值处理：本地协议 = 1，兼容底 = 当前版本。
try:
    from version import __update_protocol__ as _LOCAL_UPDATE_PROTOCOL  # type: ignore
except ImportError:
    _LOCAL_UPDATE_PROTOCOL = 1

try:
    from version import __min_compatible_version__ as _LOCAL_MIN_COMPATIBLE  # type: ignore
except ImportError:
    _LOCAL_MIN_COMPATIBLE = __version__

# 更新模块日志（仅在开发调试时有用）
_log = logging.getLogger("updater")

# ====================================================================
# ★ 配置区 —— GitHub Release 更新源配置 ★
# ====================================================================
GITHUB_REPO = "evanlumier/PiPiTextureEditor"  # GitHub 仓库
ASSET_SUFFIX = ".zip"                          # Release Asset 必须是 zip

# ── 版本检查地址（通过 raw.githubusercontent.com 读取 version.py）──
# 该域名是 GitHub 官方 CDN，无 rate limit，国内网络通常可直连
_VERSION_CHECK_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/app/version.py"

# ── Release 下载地址模板 ──
# GitHub Release 下载链接会 302 重定向到 release-assets.githubusercontent.com CDN
_RELEASE_DOWNLOAD_URL_TEMPLATE = f"https://github.com/{GITHUB_REPO}/releases/download/v{{version}}/PPTextureEditor_v{{version}}.zip"

# ── Release 页面 changelog 地址（用于获取更新说明）──
_RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# ── Changelog 文件地址（通过 raw CDN 获取，无 rate limit）──
_CHANGELOG_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/CHANGELOG"

# 重试配置
_MAX_RETRIES = 2          # 最多重试几次
_RETRY_DELAY = 2          # 重试之间等待秒数
_REQUEST_TIMEOUT = 15     # 单次请求超时秒数

# 旧版本备份文件夹名（更新时把旧文件移到这里，下次启动时清理）
OLD_BACKUP_DIR_NAME = "_old_version_backup"

# 更新锁文件（用于检测更新中途中断）
_UPDATE_LOCK_FILE = "_update_in_progress.lock"

# API 请求缓存（避免短时间内重复请求）
_CACHE_NO_UPDATE = "__no_update__"  # 哨兵值，表示已检查过且无更新
_update_check_cache = {
    "result": None,
    "timestamp": 0,
    "ttl": 600,  # 缓存有效期 10 分钟
}


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
    """获取当前应用（exe / 脚本）所在目录（即 exe 所在的根目录）"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # 开发环境：如果当前文件在 app/ 子目录下，返回上一级
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(this_dir) == "app":
        return os.path.dirname(this_dir)
    return this_dir


def get_code_dir() -> str:
    """获取业务代码目录（app/ 子目录）"""
    base = get_app_dir()
    app_dir = os.path.join(base, "app")
    if os.path.isdir(app_dir):
        return app_dir
    return base


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


# ── 内部：通过 curl.exe 获取文本内容 ─────────────────────────────
def _fetch_text_via_curl(url: str, timeout: int = _REQUEST_TIMEOUT) -> str | None:
    """
    通过系统 curl.exe 请求文本内容。
    curl.exe 使用 Windows schannel SSL 引擎，兼容公司 SSL 代理环境。
    """
    if not _CURL_PATH:
        return None
    try:
        result = subprocess.run(
            [_CURL_PATH, "-s", "--max-time", str(timeout),
             "-H", "User-Agent: PPEditor-Updater",
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

        return raw.strip() if raw.strip() else None
    except (subprocess.TimeoutExpired, Exception) as e:
        _log.debug("curl 请求 %s 失败: %s", url, e)
    return None


def _fetch_json_via_curl(url: str, timeout: int = _REQUEST_TIMEOUT) -> dict | None:
    """
    通过系统 curl.exe 请求 JSON 数据。
    """
    raw = _fetch_text_via_curl(url, timeout)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
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
        cmd = [_CURL_PATH, "-sI", "-L", "--max-time", str(timeout),
             "-H", "User-Agent: PPEditor-Updater"]
        cmd.append(url)
        result = subprocess.run(
            cmd,
            capture_output=True, timeout=timeout + 10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            return 0
        headers = result.stdout.decode("utf-8", errors="replace")
        # 注意：curl -L 跟随重定向时会输出多个响应头块（302 + 200），
        # 需要取最后一个 Content-Length（即最终 200 响应的真实文件大小）
        content_length = 0
        for line in headers.splitlines():
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return content_length
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
        cmd = [_CURL_PATH, "-L", "-s", "--max-time", str(timeout),
             "-H", "User-Agent: PPEditor-Updater",
             "-o", save_path, url]
        proc = subprocess.Popen(
            cmd,
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


def _fetch_text_via_urllib(url: str, timeout: int = _REQUEST_TIMEOUT) -> str | None:
    """
    通过 Python urllib 请求文本内容（fallback 方案）。
    注意：在某些公司网络下 OpenSSL 3.0 可能因 SSL renegotiation 卡死。
    """
    headers = {
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
                return resp.read().decode("utf-8")
        except ssl.SSLCertVerificationError:
            continue
        except ssl.SSLError:
            continue
        except Exception as e:
            _log.debug("urllib 请求 %s 失败: %s", url, e)
            return None
    return None


# ── 内部：从远程 version.py 内容中解析版本号 ─────────────────────
def _parse_remote_version(content: str) -> str | None:
    """
    从远程 version.py 文件内容中提取 __version__ 的值。
    匹配格式：__version__ = "x.y.z"
    """
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    return None


def _parse_remote_version_info(content: str) -> dict | None:
    """
    从远程 version.py 文件内容中一并提取协议元数据。

    返回 dict:
        {
            "version": "0.8.10",              # 必须
            "protocol": 2 | None,              # 可选，__update_protocol__
            "min_compatible": "0.8.10" | None, # 可选，__min_compatible_version__
        }
    如果连版本号都拿不到，返回 None（视为拉取失败）。
    """
    ver = _parse_remote_version(content)
    if not ver:
        return None

    # 协议号：__update_protocol__ = 2 （无引号的整数）
    proto: int | None = None
    m_proto = re.search(r'__update_protocol__\s*=\s*(\d+)', content)
    if m_proto:
        try:
            proto = int(m_proto.group(1))
        except ValueError:
            proto = None

    # 兼容底：__min_compatible_version__ = "0.8.10"
    min_compat: str | None = None
    m_min = re.search(
        r'__min_compatible_version__\s*=\s*["\']([^"\']+)["\']',
        content,
    )
    if m_min:
        min_compat = m_min.group(1)

    return {
        "version": ver,
        "protocol": proto,
        "min_compatible": min_compat,
    }


def _is_incremental_update_safe(
    remote_protocol: int | None,
    remote_min_compatible: str | None,
) -> tuple[bool, str]:
    """
    判断当前本地客户端能否安全走"增量"更新。

    规则（Q2=Y 宽松策略：拿不到远程协议信息 → 视为兼容，交给下载后的 zip 二次校验兜底）：
      1. 远程未提供任何协议信息 → 视为兼容（老版本 release）。
      2. 本地协议号 >= 远程 __min_compatible_version__ 对应的协议要求：
         - 若远程给了 min_compatible 版本号，本地 __version__ 必须 >= 该版本，
           否则说明本地过旧、走增量会破坏架构，判定不安全。
      3. 远程协议 - 本地协议 差距 >= 1 且远程未给出兼容底：视为不安全（保守）。

    返回 (safe: bool, reason: str)。
    """
    # 远程协议信息完全缺失 → 视为兼容（Q2=Y 宽松兜底）
    if remote_protocol is None and remote_min_compatible is None:
        return True, "remote_no_protocol_info"

    # 只要远程明确给出兼容底，就以兼容底为准做严格判断
    if remote_min_compatible:
        try:
            local_v = _parse_version(__version__)
            min_v = _parse_version(remote_min_compatible)
            if local_v < min_v:
                return False, (
                    f"local_below_min_compatible: {__version__} < {remote_min_compatible}"
                )
        except Exception:
            # 解析失败按保守处理
            return False, "parse_min_compatible_failed"

    # 远程只给了协议号：与本地做代际差比较
    if remote_protocol is not None:
        try:
            if int(remote_protocol) > int(_LOCAL_UPDATE_PROTOCOL):
                return False, (
                    f"protocol_gap: local={_LOCAL_UPDATE_PROTOCOL} < remote={remote_protocol}"
                )
        except Exception:
            pass

    return True, "compatible"


# ── 内部：获取远程最新版本号 ─────────────────────────────────────
def _fetch_remote_version() -> str | None:
    """
    从 raw.githubusercontent.com 读取远程 version.py，解析出最新版本号。
    优先使用 curl.exe，失败则 fallback 到 urllib。
    """
    # 方案1：curl.exe（Windows schannel SSL，兼容性最好）
    content = _fetch_text_via_curl(_VERSION_CHECK_URL)
    if content:
        ver = _parse_remote_version(content)
        if ver:
            return ver

    # 方案2：Python urllib（OpenSSL，fallback）
    content = _fetch_text_via_urllib(_VERSION_CHECK_URL)
    if content:
        ver = _parse_remote_version(content)
        if ver:
            return ver

    return None


def _fetch_remote_version_info() -> dict | None:
    """
    从远程 version.py 一并读回版本号 + 协议信息。

    返回 dict：{"version": ..., "protocol": ..., "min_compatible": ...}，
    连版本号都拿不到时返回 None。
    与 _fetch_remote_version() 一样优先 curl 后退 urllib。
    """
    for fetcher in (_fetch_text_via_curl, _fetch_text_via_urllib):
        content = fetcher(_VERSION_CHECK_URL)
        if not content:
            continue
        info = _parse_remote_version_info(content)
        if info and info.get("version"):
            return info
    return None


# ── 内部：获取 Release changelog（可选，失败不影响更新）──────────
def _fetch_changelog(version: str) -> str:
    """
    尝试从仓库的 CHANGELOG 文件获取更新说明。
    优先通过 raw.githubusercontent.com（无 rate limit），
    失败时 fallback 到 GitHub API（有 rate limit）。
    这是可选操作，失败时返回默认文本，不影响更新流程。
    """
    # 方案1：通过 raw CDN 获取 CHANGELOG 文件（无 rate limit，推荐）
    try:
        text = _fetch_text_via_curl(_CHANGELOG_URL, timeout=10)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    # 方案2：fallback 到 GitHub API 获取 Release body（有 rate limit）
    try:
        data = _fetch_json_via_curl(_RELEASE_API_URL, timeout=10)
        if data and isinstance(data, dict):
            body = data.get("body", "")
            if body:
                return body
    except Exception:
        pass
    return "暂无更新说明"


# ── 检查更新 ─────────────────────────────────────────────────────
def check_for_update(force: bool = False) -> dict | None:
    """
    检查 GitHub 上是否有新版本。

    策略：通过 raw.githubusercontent.com 读取远程 version.py 获取最新版本号，
    然后构造 Release 下载链接。简单、快速、无 rate limit。

    参数:
        force: 是否强制刷新（忽略缓存），用于手动检查更新

    返回值:
        有新版本时返回 dict:
            {
                "version": "0.8.6",
                "download_url": "https://github.com/.../PPTextureEditor_v0.8.6.zip",
                "changelog": "更新说明...",
                "asset_name": "PPTextureEditor_v0.8.6.zip"
            }
        无更新或检查失败时返回 None（静默失败，不影响正常使用）。
    """
    # ── 缓存检查（防止短时间内重复请求）──
    now = time.time()
    if not force and _update_check_cache["result"] is not None:
        if now - _update_check_cache["timestamp"] < _update_check_cache["ttl"]:
            cached = _update_check_cache["result"]
            _log.debug("使用缓存的更新检查结果（%d 秒前）",
                       int(now - _update_check_cache["timestamp"]))
            # 哨兵值表示"已检查过且无更新"，返回 None
            return None if cached == _CACHE_NO_UPDATE else cached

    # ── 获取远程最新版本号 + 协议信息 ──
    remote_info: dict | None = None
    last_error = None

    for attempt in range(_MAX_RETRIES):
        try:
            remote_info = _fetch_remote_version_info()
            if remote_info and remote_info.get("version"):
                break
        except Exception as e:
            last_error = e
        # 重试前等待
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_DELAY)

    if not remote_info or not remote_info.get("version"):
        raise ConnectionError("无法连接更新服务器，请检查网络连接")

    remote_version = remote_info["version"]
    remote_protocol = remote_info.get("protocol")
    remote_min_compatible = remote_info.get("min_compatible")

    # ── 比较版本号 ──
    latest_ver = _parse_version(remote_version)
    current_ver = _parse_version(__version__)

    if latest_ver <= current_ver:
        # 缓存"已是最新"结果
        _update_check_cache["result"] = _CACHE_NO_UPDATE
        _update_check_cache["timestamp"] = now
        return None  # 已是最新版本

    # ── 构造下载链接 ──
    version_str = remote_version.lstrip("vV")
    download_url = _RELEASE_DOWNLOAD_URL_TEMPLATE.format(version=version_str)
    asset_name = f"PPTextureEditor_v{version_str}.zip"

    # ── 获取 changelog（可选，失败不影响）──
    changelog = _fetch_changelog(version_str)

    # ── 协议代际兼容性判断（v0.8.10 加固版新增） ──
    # 判定不安全时，UI 层应引导用户去下完整包，而不是自动增量替换。
    incremental_safe, protocol_reason = _is_incremental_update_safe(
        remote_protocol, remote_min_compatible
    )
    if not incremental_safe:
        _log.warning(
            "检测到协议代际不兼容，需要完整安装包（原因：%s）", protocol_reason
        )

    result = {
        "version": version_str,
        "download_url": download_url,
        "changelog": changelog,
        "asset_name": asset_name,
        # 新增字段（旧 UI 可以忽略，新 UI 可以据此弹“手动下完整包”引导弹窗）
        "remote_protocol": remote_protocol,
        "min_compatible_version": remote_min_compatible,
        "require_full_install": not incremental_safe,
        "protocol_reason": protocol_reason,
    }
    # 缓存有新版本的结果
    _update_check_cache["result"] = result
    _update_check_cache["timestamp"] = now
    return result


# ── 下载并应用更新 ────────────────────────────────────────────────
class UpdateCancelledError(Exception):
    """用户取消更新时抛出的异常"""
    pass


# 可信的下载域名白名单（防止下载链接指向恶意文件）
_TRUSTED_DOWNLOAD_DOMAINS = [
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
]

def _is_trusted_url(url: str) -> bool:
    """检查下载链接是否来自可信域名"""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in _TRUSTED_DOWNLOAD_DOMAINS)
    except Exception:
        return False


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


def download_update(download_url: str, progress_callback=None, stop_event=None) -> str:
    """
    下载 zip 文件到临时目录。
    优先使用 curl.exe 下载（兼容公司 SSL 代理环境）。
    下载完成后会验证 zip 文件完整性。

    参数:
        download_url: zip 的下载链接（GitHub Release 链接）
        progress_callback: 可选，回调函数 callback(dict) 传递详细进度信息
        stop_event: 可选，threading.Event 对象，set() 后中断下载

    返回值:
        下载好的 zip 文件完整路径

    异常:
        下载失败时抛出异常，由调用方处理。
        用户取消时抛出 UpdateCancelledError。
    """
    # 安全校验：只信任白名单域名的下载链接
    if not _is_trusted_url(download_url):
        raise RuntimeError(f"不可信的下载链接: {download_url}")

    tmp_dir = tempfile.mkdtemp(prefix="ppeditor_update_")
    zip_path = os.path.join(tmp_dir, "update.zip")

    # 检查是否被取消
    if stop_event and stop_event.is_set():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise UpdateCancelledError("用户取消了下载")

    # 方案1：优先用 curl.exe 下载（跟随 302 重定向到 CDN）
    # 先通过 HEAD 请求获取文件总大小，用于显示百分比进度
    total_size = _get_content_length_via_curl(download_url, timeout=10)
    _log.debug("文件总大小: %d bytes (%s)", total_size, _format_size(total_size))

    if _download_via_curl(download_url, zip_path, timeout=600,
                          progress_callback=progress_callback,
                          stop_event=stop_event,
                          total_size=total_size):
        # 验证下载文件完整性
        if _verify_zip_integrity(zip_path):
            _log.debug("curl 下载成功且 zip 完整")
            return zip_path
        else:
            _log.debug("curl 下载的 zip 文件损坏")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    pass

    # 检查是否被取消
    if stop_event and stop_event.is_set():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise UpdateCancelledError("用户取消了下载")

    # 方案2：fallback 到 urllib
    try:
        _log.debug("curl 下载失败，尝试 urllib: %s", download_url)
        dl_headers = {"User-Agent": "PPEditor-Updater"}
        req = Request(download_url, headers=dl_headers)
        ssl_ctx = _build_insecure_ssl_context()
        with urlopen(req, timeout=600, context=ssl_ctx) as resp:
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
            _log.debug("urllib 下载成功且 zip 完整: %d bytes", os.path.getsize(zip_path))
            return zip_path
        else:
            _log.debug("urllib 下载的 zip 文件损坏")

    except UpdateCancelledError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        _log.debug("urllib 下载失败: %s", e)

    # 全部失败
    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError("下载失败，请检查网络连接后重试")


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

    参数:
        zip_path: 下载好的 zip 文件路径
        progress_callback: 可选，回调函数 callback(percent: int, stage: str)

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

    app_dir = get_app_dir()       # exe 所在根目录
    code_dir = get_code_dir()     # 业务代码目录（app/ 子目录）
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
        _progress(36, "正在验证更新包...")
        source_dir = extract_dir
        entries = os.listdir(extract_dir)

        # 如果解压后只有一个子目录，检查里面是否有 exe
        if len(entries) == 1:
            single = os.path.join(extract_dir, entries[0])
            if os.path.isdir(single):
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
        _progress(50, "正在生成更新脚本...")

        current_pid = os.getpid()
        bat_path = os.path.join(tempfile.gettempdir(), f"ppeditor_update_{current_pid}.bat")

        # ── 【二次校验】读取 zip 内 app/version.py 的协议信息（v0.8.10 加固版起启用） ──
        # 目的：远程 raw CDN 那份 version.py 有可能被缓存/网络抖动，第一次判断可能被绕过。
        # 但下载完成解压后的 version.py 是100%权威的，这里做严格判断，防止不兼容的增量替换真的落地。
        # 一旦这里判定不兼容 → 中止更新、清理锁文件与解压目录、返回 False；由 UI 引导用户手动下完整包。
        try:
            zip_version_py = os.path.join(source_dir, "app", "version.py")
            if os.path.isfile(zip_version_py):
                with open(zip_version_py, "r", encoding="utf-8") as _vf:
                    _v_content = _vf.read()
                _zip_info = _parse_remote_version_info(_v_content)
                if _zip_info:
                    _zip_proto = _zip_info.get("protocol")
                    _zip_min = _zip_info.get("min_compatible")
                    _safe, _reason = _is_incremental_update_safe(_zip_proto, _zip_min)
                    if not _safe:
                        _log.error(
                            "zip 内二次校验：协议不兼容，拒绝增量替换（原因: %s）", _reason
                        )
                        _progress(0, "更新包与当前版本架构不兼容，请前往 GitHub 下载完整安装包")
                        # 清理锁文件（注意：此时锁文件已写，必须清理，否则下次启动会误判为中断更新）
                        try:
                            if os.path.exists(lock_file):
                                os.remove(lock_file)
                        except Exception:
                            pass
                        # 清理解压目录
                        try:
                            shutil.rmtree(extract_dir, ignore_errors=True)
                        except Exception:
                            pass
                        return False
            # zip 里没有 app/version.py（可能是老架构全量包），此处不拦截，交给下方 is_split_arch 分支处理
        except Exception as _e:
            _log.debug("zip 内二次校验读取失败（不拦截，走后续分支）: %s", _e)

        # ── 智能检测新版本包结构 ──
        new_has_app_dir = os.path.isdir(os.path.join(source_dir, "app"))
        is_split_arch = new_has_app_dir or (code_dir != app_dir)

        # 生成 bat 脚本内容（GBK 编码，cmd.exe 默认代码页）
        if is_split_arch and new_has_app_dir:
            # ── 拆包架构：只替换 app/ 目录 ──
            new_app_source = os.path.join(source_dir, "app")
            bat_content = f'''@echo off
title 皮皮贴图修改器 - 正在更新...
echo ========================================
echo   皮皮贴图修改器 - 自动更新（增量模式）
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

:: 备份旧 app/ 目录
echo [2/5] 备份旧版本业务代码...
if exist "{backup_dir}" rmdir /s /q "{backup_dir}" >nul 2>&1
mkdir "{backup_dir}" >nul 2>&1
if exist "{code_dir}" (
    xcopy /s /e /y /q "{code_dir}\\*" "{backup_dir}\\" >nul 2>&1
)
echo 备份完成。
echo.

:: 删除旧 app/ 目录并复制新的
echo [3/5] 安装新版本业务代码...
if exist "{code_dir}" rmdir /s /q "{code_dir}" >nul 2>&1
mkdir "{code_dir}" >nul 2>&1
xcopy /s /e /y /q "{new_app_source}\\*" "{code_dir}\\" >nul 2>&1
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

:: 清理旧版本备份目录
rmdir /s /q "{backup_dir}" >nul 2>&1
:: 清理临时解压目录
rmdir /s /q "{extract_dir}" >nul 2>&1
:: 删除自身
del /f /q "%~f0" >nul 2>&1
exit /b 0

:rollback
echo 正在回滚到旧版本...
if exist "{code_dir}" rmdir /s /q "{code_dir}" >nul 2>&1
mkdir "{code_dir}" >nul 2>&1
xcopy /s /e /y /q "{backup_dir}\\*" "{code_dir}\\" >nul 2>&1
del /f /q "{lock_file}" >nul 2>&1
rmdir /s /q "{backup_dir}" >nul 2>&1
echo 回滚完成。正在重新启动旧版本...
start "" "{os.path.join(app_dir, current_exe_name)}"
timeout /t 3 /nobreak >nul
del /f /q "%~f0" >nul 2>&1
exit /b 1
'''
        else:
            # ── 全量替换模式（向后兼容老版本包结构）──
            bat_content = f'''@echo off
title 皮皮贴图修改器 - 正在更新...
echo ========================================
echo   皮皮贴图修改器 - 自动更新（全量模式）
echo ========================================
echo.

:: 等待旧进程退出（最多等 30 秒）
echo [1/5] 等待旧版本退出...
set /a count=0
:wait_loop2
tasklist /FI "PID eq {current_pid}" 2>nul | find "{current_pid}" >nul 2>&1
if errorlevel 1 goto :process_exited2
set /a count+=1
if %count% geq 60 (
    echo 等待超时，强制继续...
    goto :process_exited2
)
timeout /t 1 /nobreak >nul
goto :wait_loop2

:process_exited2
echo 旧版本已退出。
echo.

:: 备份旧文件
echo [2/5] 备份旧版本文件...
if exist "{backup_dir}" rmdir /s /q "{backup_dir}" >nul 2>&1
mkdir "{backup_dir}" >nul 2>&1

for %%f in ("{app_dir}\\*") do (
    if not "%%~nxf"=="{OLD_BACKUP_DIR_NAME}" if not "%%~nxf"=="{_UPDATE_LOCK_FILE}" if not "%%~nxf"=="error_log.txt" if not "%%~nxf"==".git" if not "%%~nxf"=="__pycache__" move /y "%%f" "{backup_dir}\\" >nul 2>&1
)
for /d %%f in ("{app_dir}\\*") do (
    if not "%%~nxf"=="{OLD_BACKUP_DIR_NAME}" if not "%%~nxf"==".git" if not "%%~nxf"=="__pycache__" move /y "%%f" "{backup_dir}\\" >nul 2>&1
)
echo 备份完成。
echo.

:: 兜底清理：确保旧 _internal 目录被彻底删除
if exist "{app_dir}\\_internal" rmdir /s /q "{app_dir}\\_internal" >nul 2>&1

:: 复制新文件
echo [3/5] 安装新版本...
xcopy /s /e /y /q "{source_dir}\\*" "{app_dir}\\" >nul 2>&1
if errorlevel 1 (
    echo 复制新文件失败！正在回滚...
    goto :rollback2
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

:: 清理旧版本备份目录
rmdir /s /q "{backup_dir}" >nul 2>&1
:: 清理临时解压目录
rmdir /s /q "{extract_dir}" >nul 2>&1
:: 删除自身
del /f /q "%~f0" >nul 2>&1
exit /b 0

:rollback2
echo 正在回滚到旧版本...
for %%f in ("{backup_dir}\\*") do (
    move /y "%%f" "{app_dir}\\" >nul 2>&1
)
for /d %%f in ("{backup_dir}\\*") do (
    move /y "%%f" "{app_dir}\\" >nul 2>&1
)
del /f /q "{lock_file}" >nul 2>&1
rmdir /s /q "{backup_dir}" >nul 2>&1
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
        # 失败时清理解压临时目录
        try:
            if extract_dir and os.path.isdir(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
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
    code_dir = get_code_dir()
    lock_file = os.path.join(app_dir, _UPDATE_LOCK_FILE)
    backup_dir = os.path.join(app_dir, OLD_BACKUP_DIR_NAME)

    if not os.path.exists(lock_file):
        return False

    _log.warning("检测到更新锁文件，上次更新可能中途中断")

    try:
        with open(lock_file, "r", encoding="utf-8") as f:
            lock_info = json.load(f)
        _log.warning("中断的更新信息: %s", lock_info)
    except Exception:
        lock_info = {}

    # 尝试从备份中恢复
    if os.path.exists(backup_dir) and os.listdir(backup_dir):
        _log.warning("发现备份目录，尝试恢复旧版本...")

        # 判断备份类型：增量（app/ 内容）还是全量（整个目录）
        backup_items = os.listdir(backup_dir)
        has_exe = any(f.lower().endswith(".exe") for f in backup_items)
        has_py = any(f.lower().endswith(".py") for f in backup_items)
        restore_target = code_dir if (has_py and not has_exe) else app_dir
        _log.warning("恢复目标目录: %s（增量=%s）", restore_target, has_py and not has_exe)

        try:
            for item in os.listdir(backup_dir):
                src = os.path.join(backup_dir, item)
                dst = os.path.join(restore_target, item)
                try:
                    if os.path.isdir(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    elif os.path.exists(dst):
                        os.remove(dst)
                except Exception:
                    pass
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


# ── 跳版警告代际标记（v0.8.10 加固版起启用） ──
# 文件位置固定为 %APPDATA%\PPTextureEditor\updater_generation.txt
# 只要 v0.8.10 及以后版本的 updater 被启动过一次，就会写入当前版本号。
# 后续 launcher 启动时读取该文件：
#   · 文件不存在 或 值 < 0.8.10 → 判定为"跳版用户"，弹提醒（不阻断启动）
#   · 值 >= 0.8.10             → 静默通过
# 这样可以在用户漏装 v0.8.10 加固版、直接从 v0.8.9 跳到 v0.8.11+ 时主动提醒。
_UPDATER_GENERATION_DIR_NAME = "PPTextureEditor"
_UPDATER_GENERATION_FILE_NAME = "updater_generation.txt"


def _get_updater_generation_path() -> str | None:
    """
    获取 updater_generation.txt 的完整路径。
    非 Windows 或 APPDATA 环境变量缺失时返回 None（视为不写入）。
    """
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return os.path.join(appdata, _UPDATER_GENERATION_DIR_NAME, _UPDATER_GENERATION_FILE_NAME)


def write_updater_generation() -> bool:
    """
    将当前版本号写入 %APPDATA%\\PPTextureEditor\\updater_generation.txt。
    应在每次 updater 模块被导入并进入正常启动流程时调用一次（幂等）。

    这是"跳版警告"机制的写入端，配合 launcher.check_updater_generation() 使用。
    任何异常都吞掉——写不上不能影响用户正常启动。

    返回 True 表示写入成功，False 表示失败或环境不支持。
    """
    path = _get_updater_generation_path()
    if not path:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 只写版本号本身，UTF-8 无 BOM，末尾不加换行，方便 launcher 精确比较
        with open(path, "w", encoding="utf-8") as f:
            f.write(__version__)
        return True
    except Exception as e:
        _log.warning("写入 updater_generation.txt 失败：%s", e)
        return False
