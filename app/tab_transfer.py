# -*- coding: utf-8 -*-
"""
tab_transfer.py - 跨板块通信管理模块

提供各板块之间"发送到..."功能的统一基础设施：
- 板块枚举定义
- 右键菜单创建
- 图片转临时 PNG 的工具函数
- 右键单击/拖拽区分的 Mixin 类
"""

import os
import tempfile
from typing import Optional, Callable

from PIL import Image

from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QAction, QImage, QPixmap, QMouseEvent, QCursor
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


class RightClickSendMixin:
    """
    为画布控件提供"右键单击弹菜单 / 右键拖拽平移"区分能力的 Mixin。

    使用方法：
    1. 在目标控件类中混入此 Mixin
    2. 在 __init__ 中调用 self._init_right_click_send()
    3. 在 mousePressEvent / mouseMoveEvent / mouseReleaseEvent 中
       调用对应的 _rcs_* 方法
    4. 实现 _on_right_click_send(pos) 方法来处理右键单击弹菜单
    """

    def _init_right_click_send(self):
        """初始化右键单击/拖拽区分所需的状态变量。"""
        self._rcs_press_pos: Optional[QPointF] = None
        self._rcs_is_dragging: bool = False

    def _rcs_press(self, event: QMouseEvent) -> bool:
        """
        在 mousePressEvent 中调用。
        返回 True 表示已处理（右键按下），调用方应 return。
        """
        if event.button() == Qt.RightButton:
            self._rcs_press_pos = event.position()
            self._rcs_is_dragging = False
            return True
        return False

    def _rcs_move(self, event: QMouseEvent) -> bool:
        """
        在 mouseMoveEvent 中调用。
        返回 True 表示已超过阈值，进入拖拽模式，调用方应执行平移逻辑。
        返回 False 表示还在阈值内或非右键状态。
        """
        if self._rcs_press_pos is None:
            return False
        if not (event.buttons() & Qt.RightButton):
            return False
        if self._rcs_is_dragging:
            return True  # 已经在拖拽中
        # 检查是否超过阈值
        delta = event.position() - self._rcs_press_pos
        dist = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
        if dist >= RIGHT_CLICK_THRESHOLD:
            self._rcs_is_dragging = True
            return True
        return False

    def _rcs_release(self, event: QMouseEvent) -> bool:
        """
        在 mouseReleaseEvent 中调用。
        返回 True 表示这是一次"右键单击"（未拖拽），调用方应弹出菜单。
        返回 False 表示这是一次拖拽结束或非右键。
        """
        if event.button() != Qt.RightButton:
            return False
        was_dragging = self._rcs_is_dragging
        self._rcs_press_pos = None
        self._rcs_is_dragging = False
        return not was_dragging

    def _on_right_click_send(self, global_pos):
        """
        子类需要实现此方法：在右键单击时弹出发送菜单。
        参数 global_pos 是全局坐标，用于 menu.exec(global_pos)。
        """
        raise NotImplementedError
