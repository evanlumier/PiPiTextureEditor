# -*- coding: utf-8 -*-
"""
tab_transfer.py - 跨板块通信管理模块

提供各板块之间“发送到...”功能的统一基础设施：
- 板块枚举定义
- 右键菜单创建
- 图片转临时 PNG 的工具函数
- 右键单击/拖拽区分的阈值常量
"""
import os
import tempfile
from typing import Optional, Callable

from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMenu, QWidget


# ── 板块标识 ──────────────────────────────────────────────────────────
# Tab 索引常量（与主窗口 addTab 顺序一致）
TAB_TEXTURE = 0      # 贴图修改
TAB_SPRITE = 1       # 精灵图制作
TAB_FLOWMAP = 2      # 法线绘制
TAB_GRAYGROWTH = 3   # 灰度图生成
TAB_VIEWER = 4       # 全能看图

# 板块显示名称
TAB_NAMES = {
    TAB_TEXTURE: "贴图修改",
    TAB_SPRITE: "精灵图",
    TAB_FLOWMAP: "法线绘制",
    TAB_GRAYGROWTH: "灰度图",
    TAB_VIEWER: "全能看图",
}

# 右键菜单中各板块的提示文字（仅对需要特殊说明的板块）
TAB_HINTS = {
    TAB_FLOWMAP: "发送：参考图",
    TAB_GRAYGROWTH: "发送：素材图",
}

# 5px 阈值，用于区分右键单击和右键拖拽
RIGHT_CLICK_THRESHOLD = 5


# ── 工具函数 ──────────────────────────────────────────────────────────

def pil_to_temp_png(pil_img: Image.Image, prefix: str = "transfer_") -> Optional[str]:
    """
    将 PIL Image 保存为临时 PNG 文件，返回文件路径。
    调用方负责在使用完毕后删除临时文件。
    """
    if pil_img is None:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", prefix=prefix, delete=False
        )
        tmp_path = tmp.name
        tmp.close()
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        pil_img.save(tmp_path, "PNG")
        return tmp_path
    except Exception:
        return None


def qpixmap_to_pil(pm: QPixmap) -> Optional[Image.Image]:
    """将 QPixmap 转为 PIL Image (RGBA)。"""
    if pm is None or pm.isNull():
        return None
    qimg = pm.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    arr = bytes(ptr)
    return Image.frombytes("RGBA", (w, h), arr, "raw", "RGBA")


def build_send_menu(
    parent: QWidget,
    current_tab: int,
    on_send: Callable[[int], None],
    hint: Optional[str] = None,
) -> QMenu:
    """
    构建"发送到..."右键菜单。

    参数:
        parent: 菜单的父控件
        current_tab: 当前板块索引（菜单中不显示自身）
        on_send: 回调函数，参数为目标板块索引
        hint: 可选的提示文字，显示在菜单顶部（如"发送：参考图"）
    """
    menu = QMenu(parent)
    menu.setStyleSheet("""
        QMenu {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 6px;
            padding: 4px 0px;
        }
        QMenu::item {
            padding: 6px 16px;
            color: #cdd6f4;
            font-size: 12px;
        }
        QMenu::item:selected {
            background-color: #313244;
            color: #89b4fa;
        }
        QMenu::separator {
            height: 1px;
            background: #45475a;
            margin: 4px 8px;
        }
    """)

    # 如果有提示文字，添加在顶部（灰色小字）
    if hint:
        hint_action = menu.addAction(hint)
        hint_action.setEnabled(False)
        menu.addSeparator()

    # 添加各板块菜单项（排除自身）
    for tab_idx in [TAB_TEXTURE, TAB_SPRITE, TAB_FLOWMAP, TAB_GRAYGROWTH, TAB_VIEWER]:
        if tab_idx == current_tab:
            continue
        name = TAB_NAMES[tab_idx]
        action = menu.addAction(f"► {name}")
        action.triggered.connect(lambda checked, t=tab_idx: on_send(t))

    return menu