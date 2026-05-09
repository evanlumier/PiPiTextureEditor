"""
ue4_sync.py - 皮皮贴图修改器 <-> UE4 同步模块

职责：
- 创建 Named Mutex（供 UE4 检测皮皮是否在线）
- 读取 sync_path.txt 获取 UE4 Sync 目录路径
- 写入 import_xxx.json 请求文件（原子写入：先 .tmp 再重命名）
- 等待 .result 文件并读取结果（10 秒超时）
- 监听 UE4 导出的 export_xxx.json 文件（接收贴图）
- 提供连接状态检测

协议版本：1
"""

import os
import json
import time
import uuid
import ctypes
import threading
import glob
import logging
from ctypes import wintypes
from typing import Optional, Tuple, Callable, List

_log = logging.getLogger(__name__)

# Windows API 常量
SYNCHRONIZE = 0x00100000
MUTEX_ALL_ACCESS = 0x001F0001
ERROR_ALREADY_EXISTS = 183

# Windows API 函数
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

CreateMutexW = kernel32.CreateMutexW
CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
CreateMutexW.restype = wintypes.HANDLE

OpenMutexW = kernel32.OpenMutexW
OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
OpenMutexW.restype = wintypes.HANDLE

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

ReleaseMutex = kernel32.ReleaseMutex
ReleaseMutex.argtypes = [wintypes.HANDLE]
ReleaseMutex.restype = wintypes.BOOL

# 协议常量
PROTOCOL_VERSION = 1
MUTEX_NAME = "Global\\PiPiTextureEditor_AWT_Sync"
SYNC_PATH_FILE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "AWT", "sync_path.txt")
RESULT_TIMEOUT = 10.0  # 等待 .result 文件的超时时间（秒）
RESULT_POLL_INTERVAL = 0.1  # 轮询间隔（秒）
EXPORT_POLL_INTERVAL = 0.5  # export 监听轮询间隔（秒）


class UE4SyncManager:
    """皮皮侧的 UE4 同步管理器"""

    def __init__(self):
        self._mutex_handle: Optional[int] = None
        self._sync_dir: Optional[str] = None
        self._export_listener_running: bool = False
        self._export_thread: Optional[threading.Thread] = None
        self._export_callback: Optional[Callable[[dict], None]] = None
        self._export_consecutive_errors: int = 0  # 连续失败计数

    # ========== 生命周期 ==========

    def start(self) -> bool:
        """
        启动同步：创建 Named Mutex + 读取 Sync 目录路径。
        返回 True 表示启动成功。
        """
        # 1. 创建 Named Mutex（供 UE4 检测皮皮是否在线）
        if self._mutex_handle is None:
            handle = CreateMutexW(None, False, MUTEX_NAME)
            if handle:
                self._mutex_handle = handle
            else:
                return False

        # 2. 读取 sync_path.txt
        self._sync_dir = self._read_sync_path()

        return True

    def stop(self):
        """停止同步：释放 Named Mutex + 停止 export 监听"""
        self.stop_export_listener()

        if self._mutex_handle is not None:
            CloseHandle(self._mutex_handle)
            self._mutex_handle = None
        self._sync_dir = None

    def is_running(self) -> bool:
        """是否已启动"""
        return self._mutex_handle is not None

    # ========== 连接状态 ==========

    def is_ue4_available(self) -> bool:
        """
        检测 UE4 是否可用（sync_path.txt 存在且 Sync 目录存在）。
        注意：这不检测 UE4 进程是否在运行，只检测 Sync 基础设施是否就绪。
        """
        sync_dir = self._read_sync_path()
        if sync_dir and os.path.isdir(sync_dir):
            self._sync_dir = sync_dir
            return True
        return False

    def get_sync_dir(self) -> Optional[str]:
        """获取当前 Sync 目录路径"""
        if self._sync_dir and os.path.isdir(self._sync_dir):
            return self._sync_dir
        # 尝试重新读取
        self._sync_dir = self._read_sync_path()
        return self._sync_dir

    # ========== 皮皮 → UE4 导入请求 ==========

    def send_import_request(
        self,
        png_path: str,
        target_path: str,
        asset_name: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        发送导入请求到 UE4。

        参数：
            png_path: PNG 文件的绝对路径
            target_path: UE4 资产目标路径（如 /Game/Art/UI/Textures）
            asset_name: 资产名称（可选，默认从 png_path 提取）

        返回：
            (success: bool, message: str)
        """
        # 1. 检查 Sync 目录
        sync_dir = self.get_sync_dir()
        if not sync_dir:
            return False, "无法找到 UE4 Sync 目录，请确认 UE4 编辑器已启动并启用了联动功能"

        # 2. 检查 PNG 文件
        if not os.path.isfile(png_path):
            return False, f"PNG 文件不存在: {png_path}"

        # 3. 构造 JSON 请求
        if asset_name is None:
            asset_name = os.path.splitext(os.path.basename(png_path))[0]

        timestamp = int(time.time())
        request_id = uuid.uuid4().hex[:8]
        filename = f"import_{timestamp}_{request_id}"

        request_data = {
            "version": PROTOCOL_VERSION,
            "action": "import",
            "png_path": png_path.replace("/", "\\"),
            "target_path": target_path,
            "asset_name": asset_name,
            "timestamp": timestamp,
            "request_id": request_id,
        }

        # 4. 原子写入：先写 .tmp，再重命名为 .json
        json_path = os.path.join(sync_dir, f"{filename}.json")
        tmp_path = json_path + ".tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(request_data, f, ensure_ascii=False, indent=2)

            # 原子重命名
            os.rename(tmp_path, json_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False, f"写入请求文件失败: {e}"

        # 5. 等待 .result 文件
        result_path = json_path + ".result"
        start_time = time.time()

        while time.time() - start_time < RESULT_TIMEOUT:
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        result = json.load(f)

                    try:
                        os.remove(result_path)
                    except OSError:
                        pass

                    status = result.get("status", "")
                    message = result.get("message", "")

                    if status == "success":
                        return True, message or "导入成功"
                    else:
                        return False, message or "导入失败"

                except (json.JSONDecodeError, IOError):
                    time.sleep(RESULT_POLL_INTERVAL)
                    continue

            time.sleep(RESULT_POLL_INTERVAL)

        return False, "UE4 未响应（超时 10 秒），请检查 UE4 编辑器是否正在运行"

    # ========== UE4 → 皮皮 导出监听 ==========

    def start_export_listener(self, on_export_received: Callable[[dict], None]):
        """
        启动 export 监听线程。
        当 UE4 发送 export_xxx.json 到 Sync 目录时，回调 on_export_received。

        参数：
            on_export_received: 回调函数，接收 export JSON 的 dict 内容
                dict 包含: tga_path, asset_path, asset_name, timestamp, request_id
        """
        if self._export_listener_running:
            return

        self._export_callback = on_export_received
        self._export_listener_running = True
        self._export_thread = threading.Thread(
            target=self._export_listener_loop,
            daemon=True,
            name="PiPi-ExportListener"
        )
        self._export_thread.start()

    def stop_export_listener(self):
        """停止 export 监听线程"""
        self._export_listener_running = False
        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=2.0)
        self._export_thread = None
        self._export_callback = None

    def _export_listener_loop(self):
        """export 监听线程主循环"""
        _MAX_CONSECUTIVE_ERRORS = 10  # 连续失败超过此数时降低轮询频率
        while self._export_listener_running:
            try:
                sync_dir = self.get_sync_dir()
                if sync_dir and os.path.isdir(sync_dir):
                    pattern = os.path.join(sync_dir, "export_*.json")
                    for json_path in glob.glob(pattern):
                        basename = os.path.basename(json_path)
                        if ".tmp" in basename or ".processing" in basename or ".result" in basename:
                            continue
                        self._process_export_file(json_path)
                # 成功一次就重置计数
                self._export_consecutive_errors = 0
            except Exception as e:
                self._export_consecutive_errors += 1
                # 首次失败和每10次失败记录一次日志，避免日志刷屏
                if self._export_consecutive_errors == 1 or self._export_consecutive_errors % 10 == 0:
                    _log.warning("UE4 export 监听异常（连续第 %d 次）: %s",
                                 self._export_consecutive_errors, e)

            # 连续失败过多时降低轮询频率，减少无效IO
            if self._export_consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                time.sleep(EXPORT_POLL_INTERVAL * 4)
            else:
                time.sleep(EXPORT_POLL_INTERVAL)

    def _process_export_file(self, json_path: str):
        """处理单个 export JSON 文件"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            version = data.get("version", 0)
            if version != PROTOCOL_VERSION:
                return

            action = data.get("action", "")
            if action != "export":
                return

            tga_path = data.get("tga_path", "")
            if not tga_path or not os.path.isfile(tga_path):
                return

            # 删除 JSON 文件（表示已接收）
            try:
                os.remove(json_path)
            except OSError:
                pass

            # 回调通知主界面（主界面负责加载 TGA 后清理文件）
            if self._export_callback:
                self._export_callback(data)

        except (json.JSONDecodeError, IOError, OSError):
            pass  # 文件可能还没写完，下次再试

    # ========== 内部方法 ==========

    @staticmethod
    def _read_sync_path() -> Optional[str]:
        """读取 %LOCALAPPDATA%\\AWT\\sync_path.txt"""
        try:
            if os.path.isfile(SYNC_PATH_FILE):
                with open(SYNC_PATH_FILE, "r", encoding="utf-8") as f:
                    path = f.read().strip()
                if path and os.path.isdir(path):
                    return path
        except Exception:
            pass
        return None


# 全局单例
_sync_manager: Optional[UE4SyncManager] = None


def get_sync_manager() -> UE4SyncManager:
    """获取全局 UE4SyncManager 单例"""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = UE4SyncManager()
    return _sync_manager
