# -*- coding: utf-8 -*-
"""
utils.py - 图像/UI 通用工具函数

从 Texture_tool_GUI_with_tabs.py 拆分出的公共工具函数，
供多个模块共享使用。
"""

from PIL import Image, ImageOps
from PySide6.QtGui import QPixmap, QImage


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """将 PIL Image 转换为 Qt QPixmap（统一转为 RGBA 格式）"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


def to_bw_rgba(img_rgba: Image.Image) -> Image.Image:
    """将 RGBA 图像转为黑白（保留 Alpha 通道）"""
    base = img_rgba.convert("RGBA")
    gray = ImageOps.grayscale(base)
    alpha = base.split()[-1]
    return Image.merge("RGBA", (gray, gray, gray, alpha))
