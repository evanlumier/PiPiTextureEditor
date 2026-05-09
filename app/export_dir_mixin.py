"""
export_dir_mixin.py - 导出路径记忆 Mixin

将各 Tab 中重复的导出路径缓存逻辑（_get_export_dir_cache_path /
_load_last_export_dir / _save_last_export_dir）抽取为公共基类。

使用方式：
    class MyTab(QWidget, ExportDirMixin):
        _export_dir_cache_name = "my_tab_last_export_dir.txt"

        def _get_default_export_dir(self) -> str:
            # 可选：自定义默认目录回退逻辑
            return os.path.dirname(self._src_path) if self._src_path else ""
"""

import os
from typing import Optional


class ExportDirMixin:
    """
    导出路径记忆 Mixin。

    子类需定义类属性：
        _export_dir_cache_name: str  — 缓存文件名（如 "flowmap_last_export_dir.txt"）

    子类可选覆盖：
        _get_default_export_dir() -> str  — 缓存未命中时的默认目录
    """

    # 子类必须覆盖
    _export_dir_cache_name: str = "last_export_dir.txt"

    # ── 公共方法 ──────────────────────────────────────────────────────

    def _get_export_dir_cache_path(self) -> str:
        """返回缓存文件的完整路径（%APPDATA%/GUITextureEditor/<name>）。"""
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, self._export_dir_cache_name)

    def _load_last_export_dir(self) -> str:
        """读取上次导出目录，缓存未命中时回退到 _get_default_export_dir()。"""
        try:
            with open(self._get_export_dir_cache_path(), "r", encoding="utf-8") as f:
                d = f.read().strip()
                if d and os.path.isdir(d):
                    return d
        except Exception:
            pass
        return self._get_default_export_dir()

    def _save_last_export_dir(self, path: str):
        """保存导出目录到缓存文件。path 可以是文件路径（自动取 dirname）或目录路径。"""
        try:
            dir_path = path if os.path.isdir(path) else os.path.dirname(path)
            with open(self._get_export_dir_cache_path(), "w", encoding="utf-8") as f:
                f.write(dir_path)
        except Exception:
            pass

    def _get_default_export_dir(self) -> str:
        """缓存未命中时的默认目录（子类可覆盖）。"""
        return ""
