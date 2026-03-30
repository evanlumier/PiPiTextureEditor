# -*- coding: utf-8 -*-
"""
生长灰度图生成器 Tab
布局：
  左列  = 原图预览（上）+ Mask 预览（中）+ 灰度结果预览（下）
  中列  = 主画布 GrowthCanvas（支持缩放 + 鼠标画路径）
  右列  = 导入 / 模式 / Mask / 手绘 / 导出控制区

核心数据：
  source_image   : PIL RGBA          原始图像
  sequence_frames: list[PIL Image]   序列帧列表
  mask_map       : numpy float32 H×W  0~1 遮罩
  gray_map       : numpy float32 H×W  0~1 灰度结果
  seed_map       : numpy float32 H×W  -1 = 未赋值，0~1 = 手绘时间进度

第一步骨架：
  - 导入单图 / 序列帧文件夹
  - 从 alpha / 亮度阈值生成 mask
  - 主画布缩放 + 鼠标画路径
  - 路径按时间进度写入 seed_map
  - 刷新预览
  - 导出 gray_map 为灰度 PNG
"""

import os
import re
from typing import Optional, List

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QPoint, QRect, QRectF, QRegularExpression, QTimer, QThread, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QCursor,
    QRegularExpressionValidator, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QSlider, QComboBox,
    QLineEdit, QGroupBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFileDialog, QMessageBox,
    QDoubleSpinBox, QSpinBox, QSizePolicy, QFrame, QCheckBox,
    QScrollArea, QApplication,
)

from growth_algorithms import (
    natural_sort_key,
    generate_growth_gray_from_sequence,
    cross_frame_auto_detect,
    rasterize_stroke_to_seed,
    propagate_seed_to_gray,
    smooth_gray_map,
)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── 常量 ──────────────────────────────────────────────────────────────
CANVAS_W = 512
CANVAS_H = 512
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp")
SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv")
PREVIEW_LIMIT = 1024  # 画布预览最大边长（超过此尺寸时使用缩放代理图，提升交互流畅度）


# ── 工具函数 ──────────────────────────────────────────────────────────
def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qi = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi)


def np_gray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """将 H×W float32 (0~1) 灰度图转为 QPixmap（RGBA 显示）。"""
    h, w = arr.shape
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    rgba = np.stack([u8, u8, u8, np.full_like(u8, 255)], axis=-1)
    rgba_c = np.ascontiguousarray(rgba)
    qi = QImage(rgba_c.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


def np_mask_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """将 H×W float32 mask (0~1) 转为黑白二值 QPixmap（白=主体，黑=背景）。"""
    h, w = arr.shape
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    rgba = np.stack([u8, u8, u8, np.full_like(u8, 255)], axis=-1)
    rgba_c = np.ascontiguousarray(rgba)
    qi = QImage(rgba_c.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


def np_mask_overlay_qpixmap(arr: np.ndarray) -> QPixmap:
    """将 H×W float32 mask (0~1) 转为半透明青色叠加 QPixmap，用于画布叠加。"""
    h, w = arr.shape
    u8 = (np.clip(arr, 0.0, 1.0) * 200).astype(np.uint8)  # alpha
    r = np.zeros_like(u8)
    g = np.full_like(u8, 220)
    b = np.full_like(u8, 220)
    rgba = np.stack([r, g, b, u8], axis=-1)
    rgba_c = np.ascontiguousarray(rgba)
    qi = QImage(rgba_c.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


def checkerboard_qpixmap(w: int, h: int, cell: int = 16) -> QPixmap:
    """生成棋盘格 QPixmap，用于空预览占位（替代纯黑块）。"""
    img = QImage(w, h, QImage.Format_RGB888)
    c1 = QColor("#2a2a3e")
    c2 = QColor("#1a1a2e")
    p = QPainter(img)
    for row in range(0, h, cell):
        for col in range(0, w, cell):
            color = c1 if ((row // cell + col // cell) % 2 == 0) else c2
            p.fillRect(col, row, cell, cell, color)
    p.end()
    return QPixmap.fromImage(img)


def _preview_scale_factor(w: int, h: int) -> float:
    """计算预览缩放因子，使最长边不超过 PREVIEW_LIMIT。
    返回 1.0 表示无需缩放。"""
    max_side = max(w, h)
    if max_side <= PREVIEW_LIMIT:
        return 1.0
    return PREVIEW_LIMIT / max_side


def _downscale_array(arr: np.ndarray, factor: float) -> np.ndarray:
    """将 H×W 的 float32 数组按比例缩小（最近邻采样，用于预览加速）。"""
    if factor >= 1.0:
        return arr
    h, w = arr.shape
    new_h = max(1, int(h * factor))
    new_w = max(1, int(w * factor))
    # 使用 PIL 进行缩放（比 numpy 循环快得多）
    from PIL import Image as _PILImage
    pil_img = _PILImage.fromarray((np.clip(arr, 0.0, 1.0) * 65535).astype(np.uint16), mode="I;16")
    pil_small = pil_img.resize((new_w, new_h), _PILImage.BILINEAR)
    return np.array(pil_small, dtype=np.float32) / 65535.0


def _downscale_pil(img: Image.Image, factor: float) -> Image.Image:
    """将 PIL Image 按比例缩小（用于预览加速）。"""
    if factor >= 1.0:
        return img
    new_w = max(1, int(img.width * factor))
    new_h = max(1, int(img.height * factor))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _sorted_image_files(folder: str) -> List[str]:
    """返回文件夹内按自然排序的图像文件路径列表。"""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    files.sort(key=natural_sort_key)
    return files


def compute_growth_preview_pixmap(
    source_image: Image.Image,
    gray_map: np.ndarray,
    mask_map: Optional[np.ndarray],
    progress: float,
    invert: bool = False,
) -> QPixmap:
    """
    根据 gray_map 和当前进度生成生长预览图（RGBA QPixmap）。
    严格显隐逻辑，无软过渡：
    - 不反转：gray_map <= progress 的区域显示，其余完全透明
    - 反转：  gray_map >= 1 - progress 的区域显示，其余完全透明
    最终 Alpha = src_alpha * mask_map * reveal（reveal 为严格 0/1）
    RGB 始终来自 source_image 原图，不受灰度图影响。
    """
    src = source_image if source_image.mode == "RGBA" else source_image.convert("RGBA")
    src_arr = np.array(src, dtype=np.float32)  # H×W×4, 0~255

    gray = np.clip(gray_map, 0.0, 1.0)
    p = float(np.clip(progress, 0.0, 1.0))

    # 严格显隐：满足条件为 1，不满足为 0，无任何软过渡
    if not invert:
        reveal = (gray <= p).astype(np.float32)
    else:
        reveal = (gray >= (1.0 - p)).astype(np.float32)

    # mask 限制：只在主体区域内显示
    if mask_map is not None:
        reveal = reveal * (mask_map > 0.5).astype(np.float32)

    # 原图 alpha（0~1）
    src_alpha = src_arr[:, :, 3] / 255.0
    final_alpha = src_alpha * reveal  # 0~1

    h, w = gray.shape
    # 使用 clip 确保 float32→uint8 不溢出
    r8 = np.clip(src_arr[:, :, 0], 0, 255).astype(np.uint8)
    g8 = np.clip(src_arr[:, :, 1], 0, 255).astype(np.uint8)
    b8 = np.clip(src_arr[:, :, 2], 0, 255).astype(np.uint8)
    a8 = np.clip(final_alpha * 255, 0, 255).astype(np.uint8)

    rgba = np.stack([r8, g8, b8, a8], axis=-1)
    rgba_c = np.ascontiguousarray(rgba)
    qi = QImage(rgba_c.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


def _seed_map_to_qpixmap(seed_map: np.ndarray) -> QPixmap:
    """
    将 seed_map（-1=未赋值，0~1=时间值）转为热图 QPixmap 用于画布叠加。
    有效区域：黑(0) → 蓝 → 绿 → 黄 → 红(1)，无效区域透明。
    """
    h, w = seed_map.shape
    valid = seed_map >= 0.0
    t = np.clip(seed_map, 0.0, 1.0)  # (H, W)

    # 热图颜色映射：0→蓝，0.25→青，0.5→绿，0.75→黄，1→红
    # 用分段线性插值
    r_ch = np.zeros((h, w), dtype=np.float32)
    g_ch = np.zeros((h, w), dtype=np.float32)
    b_ch = np.zeros((h, w), dtype=np.float32)

    # 段 0: t in [0, 0.25]  蓝→青  (0,0,1)→(0,1,1)
    m = (t >= 0.0) & (t < 0.25)
    s = t[m] / 0.25
    r_ch[m] = 0.0
    g_ch[m] = s
    b_ch[m] = 1.0

    # 段 1: t in [0.25, 0.5]  青→绿  (0,1,1)→(0,1,0)
    m = (t >= 0.25) & (t < 0.5)
    s = (t[m] - 0.25) / 0.25
    r_ch[m] = 0.0
    g_ch[m] = 1.0
    b_ch[m] = 1.0 - s

    # 段 2: t in [0.5, 0.75]  绿→黄  (0,1,0)→(1,1,0)
    m = (t >= 0.5) & (t < 0.75)
    s = (t[m] - 0.5) / 0.25
    r_ch[m] = s
    g_ch[m] = 1.0
    b_ch[m] = 0.0

    # 段 3: t in [0.75, 1.0]  黄→红  (1,1,0)→(1,0,0)
    m = (t >= 0.75) & (t <= 1.0)
    s = (t[m] - 0.75) / 0.25
    r_ch[m] = 1.0
    g_ch[m] = 1.0 - s
    b_ch[m] = 0.0

    # 转 uint8
    r8 = (r_ch * 255).astype(np.uint8)
    g8 = (g_ch * 255).astype(np.uint8)
    b8 = (b_ch * 255).astype(np.uint8)
    # alpha：有效区域不透明，无效区域透明
    a8 = (valid.astype(np.uint8)) * 220  # 略微半透明，叠加效果更好

    rgba = np.stack([r8, g8, b8, a8], axis=-1)
    rgba_c = np.ascontiguousarray(rgba)
    qi = QImage(rgba_c.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


# ── 可折叠分组框 ──────────────────────────────────────────────────────
class CollapsibleGroupBox(QWidget):
    """
    可折叠的分组框：标题栏带 ▼/▶ 箭头按钮，点击可折叠/展开内部内容。
    用法与 QGroupBox 类似，通过 content_layout 添加子控件。
    """

    def __init__(self, title: str = "", color: str = "#a6adc8",
                 border_color: str = "#585b70", collapsed: bool = False, parent=None):
        super().__init__(parent)
        self._collapsed = collapsed
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # 主布局
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 外框容器（模拟 QGroupBox 样式）
        self._frame = QFrame()
        self._frame.setStyleSheet(
            f"QFrame#collapsible_frame {{ border:1px solid {border_color}; border-radius:8px;"
            f"margin-top:7px; padding:0px; }}"
        )
        self._frame.setObjectName("collapsible_frame")
        frame_lay = QVBoxLayout(self._frame)
        frame_lay.setContentsMargins(6, 4, 6, 6)
        frame_lay.setSpacing(0)

        # 标题按钮
        arrow = "▶" if collapsed else "▼"
        self._toggle_btn = QPushButton(f"{arrow}  {title}")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._toggle_btn.setStyleSheet(
            f"QPushButton {{ color:{color}; font-size:11px; font-weight:700;"
            f"text-align:left; padding:4px 2px; border:none; }}"
            f"QPushButton:hover {{ color:#cdd6f4; }}"
        )
        self._toggle_btn.clicked.connect(self._toggle)
        self._title = title
        self._color = color
        frame_lay.addWidget(self._toggle_btn)

        # 内容区域
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 4, 0, 0)
        self._content_layout.setSpacing(6)
        self._content.setVisible(not collapsed)
        frame_lay.addWidget(self._content)

        outer.addWidget(self._frame)

    @property
    def content_layout(self) -> QVBoxLayout:
        """获取内容区域的 layout，用于添加子控件。"""
        return self._content_layout

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        arrow = "▶" if self._collapsed else "▼"
        self._toggle_btn.setText(f"{arrow}  {self._title}")
        self._refresh_ancestor_layouts()

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        arrow = "▶" if collapsed else "▼"
        self._toggle_btn.setText(f"{arrow}  {self._title}")
        self._refresh_ancestor_layouts()

    def _refresh_ancestor_layouts(self):
        """递归向上刷新所有祖先的布局，确保折叠/展开后尺寸正确收缩。"""
        from PySide6.QtWidgets import QScrollArea
        from PySide6.QtCore import QTimer

        # 第一步：沿着 widget 树向上，invalidate 每一级 layout
        w = self
        while w is not None:
            w.updateGeometry()
            lay = w.layout()
            if lay:
                lay.invalidate()
            if isinstance(w, QScrollArea):
                break
            w = w.parentWidget()

        # 第二步：从 QScrollArea 的 inner widget 开始，重新 activate 布局
        # 这样 Qt 会根据 sizeHint / sizePolicy 重新分配空间
        scroll = w if isinstance(w, QScrollArea) else None
        if scroll:
            inner = scroll.widget()
            if inner and inner.layout():
                inner.layout().activate()
            # 让 QScrollArea 根据新的 sizeHint 重新决定是否需要滚动条
            scroll.updateGeometry()

        # 延迟再刷新一次，应对异步布局更新
        QTimer.singleShot(0, lambda: self._deferred_refresh(scroll))

    @staticmethod
    def _deferred_refresh(scroll):
        """延迟刷新：确保 QScrollArea 正确更新。"""
        if scroll is None:
            return
        inner = scroll.widget()
        if inner and inner.layout():
            inner.layout().invalidate()
            inner.layout().activate()
        scroll.updateGeometry()


# ── 小预览标签 ────────────────────────────────────────────────────────
class PreviewLabel(QWidget):
    """
    固定高度的缩略图预览组件，保持宽高比居中显示。
    - 空状态显示棋盘格 + 提示文字
    - 支持底部说明文字（hint）
    """

    def __init__(self, placeholder: str = "（未导入）", hint: str = "", parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._hint = hint
        self._pixmap: Optional[QPixmap] = None
        self._checker: Optional[QPixmap] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._img_label = QLabel()
        self._img_label.setMinimumHeight(72)
        self._img_label.setMaximumHeight(140)
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet(
            "background:#1e1e2e; border:1px solid #383850; border-radius:6px;"
            "color:#585b70; font-size:11px;"
        )
        self._img_label.setText(placeholder)
        # 宽度方向使用 Ignored，防止 setPixmap 后 sizeHint 跟随图片尺寸撑大父控件
        from PySide6.QtWidgets import QSizePolicy
        self._img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addWidget(self._img_label)

        if hint:
            self._hint_label = QLabel(hint)
            self._hint_label.setStyleSheet("color:#45475a; font-size:9px;")
            self._hint_label.setAlignment(Qt.AlignCenter)
            self._hint_label.setWordWrap(True)
            layout.addWidget(self._hint_label)
        else:
            self._hint_label = None

    def set_pixmap(self, px: Optional[QPixmap]):
        self._pixmap = px
        if px is None:
            # 显示棋盘格 + 提示文字
            self._img_label.setPixmap(QPixmap())
            self._img_label.setText(self._placeholder)
        else:
            self._img_label.setText("")
            self._refresh_scaled()

    def set_hint(self, text: str):
        if self._hint_label:
            self._hint_label.setText(text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pixmap:
            self._refresh_scaled()

    def _refresh_scaled(self):
        if self._pixmap:
            w = self._img_label.width()
            h = self._img_label.height()
            scaled = self._pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            # 在棋盘格背景上绘制图像（支持透明）
            checker = checkerboard_qpixmap(w, h, 8)
            result = QPixmap(w, h)
            result.fill(Qt.transparent)
            p = QPainter(result)
            p.drawPixmap(0, 0, checker)
            # 居中绘制缩放后的图像
            ox = (w - scaled.width()) // 2
            oy = (h - scaled.height()) // 2
            p.drawPixmap(ox, oy, scaled)
            p.end()
            self._img_label.setPixmap(result)


# ── 主画布 ────────────────────────────────────────────────────────────
class GrowthCanvas(QWidget):
    """
    生长灰度图主画布。
    - 显示底图（source_image 或 gray_map 叠加）
    - 支持 Ctrl+滚轮 缩放，中键拖拽平移
    - 鼠标左键画路径，路径点按时间进度写入 seed_map
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # 内部数据（由 GrowthGrayTab 注入）
        self._source_px: Optional[QPixmap] = None   # 底图 QPixmap
        self._overlay_px: Optional[QPixmap] = None  # 叠加层（gray_map / seed_map）
        self._overlay_opaque: bool = False           # True=叠加层完全不透明（生长预览用）

        # 缩放 & 平移
        self._zoom: float = 1.0
        self._offset: QPoint = QPoint(0, 0)
        self._pan_last: Optional[QPoint] = None  # 右键拖动起点

        # 鼠标位置（用于画笔圆圈预览）
        self._cursor_pos: Optional[QPoint] = None
        self._show_cursor: bool = False          # 是否显示画笔圆圈（仅单图模式）

        # 绘制状态
        self._drawing: bool = False
        self._eraser_mode: bool = False          # True = 橡皮擦模式
        self._stroke_points: List[QPoint] = []   # 当前笔触（画布坐标）
        self._brush_radius: int = 12             # 像素半径（画布坐标）
        self._draw_value: float = 0.5            # 写入 seed_map 的值（由外部更新）

        # 回调：笔触结束时通知 Tab
        self.on_stroke_finished = None   # callable(points_canvas: list[tuple[int,int]], value: float)

        self.setStyleSheet("background:#181825;")
        self.setCursor(QCursor(Qt.CrossCursor))

        # 拖拽导入
        self.setAcceptDrops(True)
        self.on_drop_files = None   # callable(paths: list[str])
        self._drag_hover = False    # 拖拽悬停高亮状态

    # ── 属性 ──────────────────────────────────────────────────────────
    @property
    def brush_radius(self) -> int:
        return self._brush_radius

    @brush_radius.setter
    def brush_radius(self, v: int):
        self._brush_radius = max(1, v)

    @property
    def draw_value(self) -> float:
        return self._draw_value

    @draw_value.setter
    def draw_value(self, v: float):
        self._draw_value = float(np.clip(v, 0.0, 1.0))

    # ── 数据注入 ──────────────────────────────────────────────────────
    def set_source(self, px: Optional[QPixmap]):
        self._source_px = px
        self.update()

    def set_overlay(self, px: Optional[QPixmap], opaque: bool = False):
        self._overlay_px = px
        self._overlay_opaque = opaque
        self.update()

    # ── 坐标转换 ──────────────────────────────────────────────────────
    def _widget_to_canvas(self, wp: QPoint) -> QPoint:
        """将 widget 坐标转换为画布（图像）坐标。"""
        cx = (wp.x() - self._offset.x()) / self._zoom
        cy = (wp.y() - self._offset.y()) / self._zoom
        return QPoint(int(cx), int(cy))

    def _canvas_rect(self) -> QRect:
        """当前画布在 widget 中的显示矩形。"""
        if self._source_px:
            w = int(self._source_px.width() * self._zoom)
            h = int(self._source_px.height() * self._zoom)
        else:
            w = int(CANVAS_W * self._zoom)
            h = int(CANVAS_H * self._zoom)
        return QRect(self._offset.x(), self._offset.y(), w, h)

    def _fit_to_view(self):
        """将画布缩放适配到当前 widget 大小。"""
        if self._source_px:
            iw, ih = self._source_px.width(), self._source_px.height()
        else:
            iw, ih = CANVAS_W, CANVAS_H
        ww, wh = self.width(), self.height()
        self._zoom = min(ww / iw, wh / ih, 1.0)
        self._offset = QPoint(
            int((ww - iw * self._zoom) / 2),
            int((wh - ih * self._zoom) / 2),
        )
        self.update()

    # ── 绘制 ──────────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#181825"))

        rect = self._canvas_rect()

        # 底图
        if self._source_px:
            p.drawPixmap(rect, self._source_px)
        else:
            p.setPen(QColor("#383850"))
            p.drawRect(rect)
            p.setPen(QColor("#585b70"))
            p.drawText(rect, Qt.AlignCenter, "导入图像后显示")

        # 叠加层
        if self._overlay_px:
            # 生长预览叠加层需要完全不透明，其他叠加层半透明
            p.setOpacity(1.0 if self._overlay_opaque else 0.6)
            p.drawPixmap(rect, self._overlay_px)
            p.setOpacity(1.0)
        # 画笔圆圈预览（仅单图模式下显示）
        if self._show_cursor and self._cursor_pos is not None:
            r_w = max(2.0, self._brush_radius * self._zoom)
            mx = float(self._cursor_pos.x())
            my = float(self._cursor_pos.y())
            # 橡皮擦模式：橙色圆圈；绘制模式：白色圆圈
            if self._eraser_mode:
                outer_color = QColor(166, 227, 161, 200)  # #a6e3a1 浅绿色
            else:
                outer_color = QColor(255, 255, 255, 180)
            pen_out = QPen(outer_color)
            pen_out.setWidth(1)
            p.setPen(pen_out)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(mx - r_w, my - r_w, r_w * 2, r_w * 2))
            # 内圈：黑色半透明（增强对比度）
            pen_in = QPen(QColor(0, 0, 0, 120))
            pen_in.setWidth(1)
            p.setPen(pen_in)
            p.drawEllipse(QRectF(mx - r_w + 1, my - r_w + 1,
                                 r_w * 2 - 2, r_w * 2 - 2))

        # 拖拽悬停高亮边框
        if self._drag_hover:
            pen_drag = QPen(QColor("#89b4fa"))
            pen_drag.setWidth(3)
            pen_drag.setStyle(Qt.DashLine)
            p.setPen(pen_drag)
            p.setBrush(QColor(137, 180, 250, 30))
            p.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 8, 8)
            # 居中提示文字
            p.setPen(QColor("#89b4fa"))
            font = p.font()
            font.setPointSize(14)
            font.setBold(True)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignCenter, "释放以导入图片")

        p.end()

    # ── 鼠标事件 ──────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            # 右键拖动平移
            self._pan_last = e.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif e.button() == Qt.LeftButton and self._show_cursor:
            # 仅单图模式下允许绘制
            self._drawing = True
            self._stroke_points = [self._widget_to_canvas(e.pos())]
            self.update()

    def mouseMoveEvent(self, e):
        self._cursor_pos = e.pos()
        if self._pan_last is not None and (e.buttons() & Qt.RightButton):
            delta = e.pos() - self._pan_last
            self._offset += delta
            self._pan_last = e.pos()
            self.update()
        elif self._drawing and (e.buttons() & Qt.LeftButton):
            cp = self._widget_to_canvas(e.pos())
            if not self._stroke_points or cp != self._stroke_points[-1]:
                self._stroke_points.append(cp)
            self.update()
        else:
            self.update()  # 刷新画笔圆圈位置

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan_last = None
            self.setCursor(QCursor(Qt.CrossCursor))
        elif e.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            if self._stroke_points and self.on_stroke_finished:
                pts = [(p.x(), p.y()) for p in self._stroke_points]
                # 橡皮擦模式传特殊值 -2.0，由 Tab 层识别并执行擦除
                value = -2.0 if self._eraser_mode else self._draw_value
                self.on_stroke_finished(pts, value)
            self._stroke_points = []
            self.update()

    def leaveEvent(self, e):
        self._cursor_pos = None
        self.update()

    def wheelEvent(self, e):
        # 滚轮直接缩放（不需要 Ctrl）
        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        old_zoom = self._zoom
        self._zoom = max(0.05, min(32.0, self._zoom * factor))
        # 以鼠标位置为缩放中心
        mp = e.position().toPoint()
        self._offset = QPoint(
            int(mp.x() - (mp.x() - self._offset.x()) * self._zoom / old_zoom),
            int(mp.y() - (mp.y() - self._offset.y()) * self._zoom / old_zoom),
        )
        self.update()
        e.accept()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # 首次显示时自动适配
        if self._zoom == 1.0:
            self._fit_to_view()

    # ── 拖拽导入 ──────────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._drag_hover = True
            self.update()

    def dragLeaveEvent(self, e):
        self._drag_hover = False
        self.update()

    def dropEvent(self, e):
        self._drag_hover = False
        self.update()
        urls = e.mimeData().urls()
        if urls and self.on_drop_files:
            paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
            if paths:
                self.on_drop_files(paths)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            self._fit_to_view()
        else:
            super().keyPressEvent(e)


# ── 序列帧生成后台线程 ─────────────────────────────────────────────────
class _SeqGenWorker(QThread):
    """在后台线程中执行 generate_growth_gray_from_sequence，避免阻塞 UI。"""
    progress = Signal(int, int)       # (current_frame, total_frames)
    finished_ok = Signal(dict)        # 生成结果 dict
    finished_err = Signal(str)        # 错误信息
    cancelled = Signal()              # 用户取消

    def __init__(self, frame_iter, frame_count, source_mode, presence_blur,
                 hit_threshold, mask_threshold, invert, force_mode, parent=None):
        super().__init__(parent)
        self._frame_iter = frame_iter
        self._frame_count = frame_count
        self._source_mode = source_mode
        self._presence_blur = presence_blur
        self._hit_threshold = hit_threshold
        self._mask_threshold = mask_threshold
        self._invert = invert
        self._force_mode = force_mode
        self._cancelled = False

    def cancel(self):
        """请求取消生成。"""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            result = generate_growth_gray_from_sequence(
                frame_iter=self._frame_iter,
                frame_count=self._frame_count,
                source_mode=self._source_mode,
                presence_blur=self._presence_blur,
                hit_threshold=self._hit_threshold,
                mask_threshold=self._mask_threshold,
                invert=self._invert,
                force_mode=self._force_mode,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
                cancel_flag=lambda: self._cancelled,
            )
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.finished_ok.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as ex:
            self.finished_err.emit(str(ex))


# ── 主 Tab ────────────────────────────────────────────────────────────
class GrowthGrayTab(QWidget):
    """生长灰度图生成器 Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── 核心数据 ──────────────────────────────────────────────────
        self.source_image: Optional[Image.Image] = None       # PIL RGBA
        self.sequence_frames: List[Image.Image] = []          # 序列帧
        self.mask_map: Optional[np.ndarray] = None            # H×W float32 0~1
        self.gray_map: Optional[np.ndarray] = None            # H×W float32 0~1（兼容旧引用）
        self.base_gray_map: Optional[np.ndarray] = None       # H×W float32 0~1 原始灰度图
        self.noise_map: Optional[np.ndarray] = None           # H×W float32 0~1 噪波图
        self.final_gray_map: Optional[np.ndarray] = None      # H×W float32 0~1 叠加噪波后结果
        self.seed_map: Optional[np.ndarray] = None            # H×W float32 -1 未赋值
        self._noise_image: Optional[Image.Image] = None       # 导入的噪波贴图（PIL）

        self._src_path: Optional[str] = None                  # 导入路径（单图）
        self._output_basename: Optional[str] = None           # 导出基础名
        self._seq_generated: bool = False                     # 是否已生成过序列帧灰度图

        # ── 预览代理缓存（大图加速）──
        self._preview_factor: float = 1.0                     # 预览缩放因子
        self._checker_cache: Optional[QPixmap] = None         # 缓存的棋盘格
        self._checker_cache_size: tuple = (0, 0)              # 缓存尺寸
        self._seq_file_paths: List[str] = []                  # 序列帧文件路径（延迟加载用）
        self._video_path: Optional[str] = None                # 视频文件路径
        self._video_frame_count: int = 0                      # 视频总帧数
        self._video_sample_interval: int = 1                  # 视频帧采样间隔
        self._seq_gen_worker: Optional[_SeqGenWorker] = None  # 序列帧生成后台线程

        # ── 防抖 Timer（参数变化后延迟 600ms 自动重新生成）──────────────
        from PySide6.QtCore import QTimer
        self._auto_regen_timer = QTimer(self)
        self._auto_regen_timer.setSingleShot(True)
        self._auto_regen_timer.setInterval(600)
        self._auto_regen_timer.timeout.connect(self._auto_regen_if_ready)

        # 单图模式：亮度阈值防抖 Timer（300ms，调整后自动刷新 mask）
        self._thresh_timer = QTimer(self)
        self._thresh_timer.setSingleShot(True)
        self._thresh_timer.setInterval(300)
        self._thresh_timer.timeout.connect(self._gen_mask_from_luminance)

        # ── 手绘路径撤销历史 ──────────────────────────────────────────
        self._seed_history: List[np.ndarray] = []             # seed_map 快照栈
        self._MAX_HISTORY = 30                                 # 最多保留 30 步

        self._build_ui()
        self._connect_signals()

    # ── 构建 UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        # ===== 三栏 QSplitter（防止宽图撑开布局遮挡右侧面板）=====
        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            "QSplitter::handle { background:#313244; }"
            "QSplitter::handle:hover { background:#585b70; }"
        )
        root.addWidget(splitter)

        # ===== 左列：预览区 =====
        left_widget = QWidget()
        left_widget.setMaximumWidth(220)
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)

        def _section_title(text, color="#89dceb"):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{color}; font-size:10px; font-weight:600;")
            return lbl

        # ── 序列帧模式预览区 ──
        self._seq_preview_widget = QWidget()
        seq_pv = QVBoxLayout(self._seq_preview_widget)
        seq_pv.setContentsMargins(0, 0, 0, 0)
        seq_pv.setSpacing(4)

        seq_pv.addWidget(_section_title("首帧原图", "#89dceb"))
        self.lbl_src = PreviewLabel("（未导入）")
        seq_pv.addWidget(self.lbl_src)

        seq_pv.addWidget(_section_title("末帧原图", "#89dceb"))
        self.lbl_last_frame = PreviewLabel("（未导入）")
        seq_pv.addWidget(self.lbl_last_frame)

        seq_pv.addWidget(_section_title("主体识别范围", "#a6e3a1"))
        self.lbl_mask = PreviewLabel("（未生成）", hint="白色=主体区域")
        seq_pv.addWidget(self.lbl_mask)

        seq_pv.addWidget(_section_title("生长灰度结果", "#f38ba8"))
        self.lbl_gray = PreviewLabel("（未生成）", hint="黑=先出现  白=后出现")
        seq_pv.addWidget(self.lbl_gray)

        seq_pv.addWidget(_section_title("噪波叠加结果", "#89b4fa"))
        self.lbl_noise_result = PreviewLabel("（未启用）", hint="叠加噪波后的最终灰度图")
        seq_pv.addWidget(self.lbl_noise_result)

        # 高级预览折叠区
        self._adv_toggle_btn = QPushButton("▶ 高级预览（算法中间图）")
        self._adv_toggle_btn.setStyleSheet(
            "color:#585b70; font-size:9px; border:none; text-align:left; padding:2px 0;"
        )
        self._adv_toggle_btn.setFlat(True)
        seq_pv.addWidget(self._adv_toggle_btn)

        self._adv_preview_widget = QWidget()
        adv_pv = QVBoxLayout(self._adv_preview_widget)
        adv_pv.setContentsMargins(0, 0, 0, 0)
        adv_pv.setSpacing(4)

        adv_pv.addWidget(_section_title("首帧占位图", "#89b4fa"))
        _note1 = QLabel("⚠ 这是识别结果，不是原图")
        _note1.setStyleSheet("color:#a6adc8; font-size:9px;")
        adv_pv.addWidget(_note1)
        self.lbl_presence = PreviewLabel("（未生成）")
        adv_pv.addWidget(self.lbl_presence)

        adv_pv.addWidget(_section_title("末帧单调包络", "#cba6f7"))
        _note2 = QLabel("⚠ 这是识别结果，不是原图")
        _note2.setStyleSheet("color:#a6adc8; font-size:9px;")
        adv_pv.addWidget(_note2)
        self.lbl_envelope = PreviewLabel("（未生成）")
        adv_pv.addWidget(self.lbl_envelope)

        self._adv_preview_widget.setVisible(False)
        seq_pv.addWidget(self._adv_preview_widget)
        seq_pv.addStretch(1)

        left.addWidget(self._seq_preview_widget)

        # ── 单图模式预览区 ──
        self._single_preview_widget = QWidget()
        single_pv = QVBoxLayout(self._single_preview_widget)
        single_pv.setContentsMargins(0, 0, 0, 0)
        single_pv.setSpacing(4)

        single_pv.addWidget(_section_title("原图", "#89dceb"))
        self.lbl_src_single = PreviewLabel("（未导入）")
        single_pv.addWidget(self.lbl_src_single)

        single_pv.addWidget(_section_title("主体范围", "#a6e3a1"))
        self.lbl_mask_single = PreviewLabel("（未生成）", hint="白色=主体区域")
        single_pv.addWidget(self.lbl_mask_single)

        single_pv.addWidget(_section_title("生长灰度结果", "#f38ba8"))
        self.lbl_gray_single = PreviewLabel("（未生成）", hint="黑=先出现  白=后出现")
        single_pv.addWidget(self.lbl_gray_single)

        single_pv.addWidget(_section_title("噪波叠加结果", "#89b4fa"))
        self.lbl_noise_result_single = PreviewLabel("（未启用）", hint="叠加噪波后的最终灰度图")
        single_pv.addWidget(self.lbl_noise_result_single)
        single_pv.addStretch(1)

        self._single_preview_widget.setVisible(False)
        left.addWidget(self._single_preview_widget)

        # ===== 中列：主画布 =====
        mid_widget = QWidget()
        mid_widget.setMinimumWidth(300)
        mid = QVBoxLayout(mid_widget)
        mid.setContentsMargins(8, 0, 8, 0)
        mid.setSpacing(6)

        canvas_header = QHBoxLayout()
        canvas_lbl = QLabel("生长画布")
        canvas_lbl.setStyleSheet("color:#cdd6f4; font-size:12px; font-weight:700;")
        self.lbl_canvas_hint = QLabel("Ctrl+滚轮 缩放 · 中键拖拽 · F 适配")
        self.lbl_canvas_hint.setStyleSheet("color:#585b70; font-size:10px;")
        canvas_header.addWidget(canvas_lbl)
        canvas_header.addStretch()
        canvas_header.addWidget(self.lbl_canvas_hint)
        mid.addLayout(canvas_header)

        # 画布叠加模式切换
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("显示模式："))
        self.combo_overlay = QComboBox()
        self.combo_overlay.addItems([
            "原图",
            "原图 + 主体范围",
            "原图 + 灰度结果",
            "仅灰度结果",
            "仅叠加噪波后结果",
            "生长预览",
        ])
        self.combo_overlay.setToolTip("切换画布中间区域的显示内容")
        overlay_row.addWidget(self.combo_overlay, 1)
        mid.addLayout(overlay_row)

        self.canvas = GrowthCanvas()
        self.canvas.on_stroke_finished = self._on_stroke_finished
        self.canvas.on_drop_files = self._on_canvas_drop
        mid.addWidget(self.canvas, 1)

        # 笔刷控制（单图模式下显示，序列帧模式下隐藏）
        self._brush_group = QGroupBox("手绘路径控制")
        self._brush_group.setStyleSheet(
            "QGroupBox { border:1px solid #585b70; border-radius:8px;"
            "margin-top:14px; padding-top:8px; }"
            "QGroupBox::title { color:#a6adc8; left:10px; }"
        )
        bg = QGridLayout(self._brush_group)
        bg.setHorizontalSpacing(8)
        bg.setVerticalSpacing(4)
        bg.setColumnStretch(1, 1)

        # 笔刷大小
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(1, 200)
        self.slider_brush.setValue(12)
        self.edit_brush = QLineEdit("12")
        self.edit_brush.setFixedWidth(50)
        self.edit_brush.setAlignment(Qt.AlignCenter)
        self.edit_brush.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d{1,3}$"))
        )
        bg.addWidget(QLabel("笔刷大小："), 0, 0)
        bg.addWidget(self.slider_brush, 0, 1)
        bg.addWidget(self.edit_brush, 0, 2)

        # 写入进度
        self.slider_value = QSlider(Qt.Horizontal)
        self.slider_value.setRange(0, 100)
        self.slider_value.setValue(50)
        self.edit_value = QLineEdit("0.50")
        self.edit_value.setFixedWidth(50)
        self.edit_value.setAlignment(Qt.AlignCenter)
        self.edit_value.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d*$"))
        )
        bg.addWidget(QLabel("写入进度："), 1, 0)
        bg.addWidget(self.slider_value, 1, 1)
        bg.addWidget(self.edit_value, 1, 2)

        # 笔刷硬度
        self.slider_hardness = QSlider(Qt.Horizontal)
        self.slider_hardness.setRange(1, 100)
        self.slider_hardness.setValue(100)
        self.lbl_hardness = QLabel("硬度：100%")
        self.lbl_hardness.setFixedWidth(70)
        bg.addWidget(QLabel("笔刷硬度："), 2, 0)
        bg.addWidget(self.slider_hardness, 2, 1)
        bg.addWidget(self.lbl_hardness, 2, 2)

        # 清空路径
        self.btn_clear_seed = QPushButton("清空手绘路径")
        self.btn_clear_seed.setStyleSheet("color:#f38ba8;")
        bg.addWidget(self.btn_clear_seed, 3, 0, 1, 3)

        # 橡皮擦按钮
        self.btn_eraser = QPushButton("☐  橡皮擦")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setChecked(False)
        self.btn_eraser.setToolTip(
            "切换橡皮擦模式：左键拖拽可擦除已绘制的路径\n"
            "橡皮擦圆圈显示为浅绿色，绘制圆圈为白色"
        )
        self.btn_eraser.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; border-radius:6px;"
            "padding:5px; font-size:11px; }"
            "QPushButton:checked { background:#a6e3a1; color:#1e1e2e; font-weight:700; }"
        )
        self.btn_eraser.toggled.connect(self._on_eraser_toggled)
        bg.addWidget(self.btn_eraser, 4, 0, 1, 3)

        self._brush_group.setVisible(False)
        mid.addWidget(self._brush_group)

        # ===== 右列：控制面板 =====
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(260)
        right_scroll.setMaximumWidth(380)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")

        right_inner = QWidget()
        right_inner.setMinimumWidth(240)  # 防止内容被压缩折叠
        right = QVBoxLayout(right_inner)
        right.setContentsMargins(4, 4, 4, 4)
        right.setSpacing(8)
        right.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)

        # ── 模式切换（最顶部）──
        mode_group = CollapsibleGroupBox("工作模式", color="#89b4fa", border_color="#89b4fa")
        mode_layout = mode_group.content_layout
        mode_layout.setSpacing(4)

        self.btn_mode_seq = QPushButton("🎞  序列帧自动生成")
        self.btn_mode_seq.setCheckable(True)
        self.btn_mode_seq.setChecked(False)
        self.btn_mode_seq.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; border-radius:6px;"
            "padding:7px; font-size:11px; }"
            "QPushButton:checked { background:#cba6f7; color:#1e1e2e; font-weight:700; }"
        )
        mode_layout.addWidget(self.btn_mode_seq)

        self.btn_mode_single = QPushButton("✏  单图手绘生成")
        self.btn_mode_single.setCheckable(True)
        self.btn_mode_single.setChecked(False)
        self.btn_mode_single.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; border-radius:6px;"
            "padding:7px; font-size:11px; }"
            "QPushButton:checked { background:#a6e3a1; color:#1e1e2e; font-weight:700; }"
        )
        mode_layout.addWidget(self.btn_mode_single)

        self.lbl_mode_hint = QLabel("请先选择工作模式")
        self.lbl_mode_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_mode_hint.setAlignment(Qt.AlignCenter)
        mode_layout.addWidget(self.lbl_mode_hint)

        right.addWidget(mode_group)

        # ═══════════════════════════════════════════════════════════════
        # ── 序列帧模式控件区 ──
        # ═══════════════════════════════════════════════════════════════
        self._seq_controls = QWidget()
        self._seq_controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        seq_ctrl_layout = QVBoxLayout(self._seq_controls)
        seq_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        seq_ctrl_layout.setSpacing(8)

        # 导入
        seq_import_group = CollapsibleGroupBox("导入序列帧", color="#89b4fa", border_color="#89b4fa")
        sig = seq_import_group.content_layout
        sig.setSpacing(6)

        self.btn_import_seq = QPushButton("选择序列帧文件夹")
        self.btn_import_seq.setStyleSheet(
            "background:#313244; color:#cdd6f4; border-radius:5px; padding:6px;"
        )
        sig.addWidget(self.btn_import_seq)

        # 视频导入按钮
        self.btn_import_video = QPushButton("🎬 导入视频")
        self.btn_import_video.setStyleSheet(
            "background:#313244; color:#cdd6f4; border-radius:5px; padding:6px;"
        )
        self.btn_import_video.setToolTip(
            "从视频文件（MP4/MOV/AVI/WebM）中提取帧并生成灰度图\n"
            "省去手动导出序列帧的步骤"
        )
        sig.addWidget(self.btn_import_video)

        # 帧采样间隔控件（视频导入时可用）
        sample_row = QHBoxLayout()
        sample_row.setSpacing(4)
        sample_row.addWidget(QLabel("帧采样："))
        self.spin_video_sample = QSpinBox()
        self.spin_video_sample.setRange(1, 30)
        self.spin_video_sample.setValue(1)
        self.spin_video_sample.setSuffix(" 帧取1帧")
        self.spin_video_sample.setToolTip(
            "每隔 N 帧取一帧参与计算\n"
            "1 = 全部帧（最精确）\n"
            "2 = 隔一帧取一帧（速度翻倍）\n"
            "5 = 每5帧取一帧（适合长视频）"
        )
        sample_row.addWidget(self.spin_video_sample)
        self._video_sample_widget = QWidget()
        self._video_sample_widget.setLayout(sample_row)
        self._video_sample_widget.setVisible(False)  # 默认隐藏，导入视频后显示
        sig.addWidget(self._video_sample_widget)

        self.lbl_seq_import_info = QLabel("未导入")
        self.lbl_seq_import_info.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_seq_import_info.setWordWrap(True)
        sig.addWidget(self.lbl_seq_import_info)

        seq_ctrl_layout.addWidget(seq_import_group)

        # 内容识别设置
        seq_recog_group = CollapsibleGroupBox("内容识别设置", color="#cba6f7", border_color="#cba6f7")
        srg = seq_recog_group.content_layout
        srg.setSpacing(8)

        # 内容识别方式
        srg.addWidget(QLabel("内容识别方式："))
        self.combo_source_mode = QComboBox()
        self.combo_source_mode.addItems(["自动", "按透明度识别", "按亮度识别"])
        self.combo_source_mode.setToolTip(
            "自动：有透明通道时用透明度，否则用亮度\n"
            "按透明度识别：强制使用 Alpha 通道，适合有透明区域的贴图\n"
            "按亮度识别：强制使用亮度，适合无透明通道的亮色贴图"
        )
        srg.addWidget(self.combo_source_mode)

        # 动态状态标签：自动模式下显示实际使用的识别方式
        self.lbl_source_mode_actual = QLabel("自动模式：尚未生成")
        self.lbl_source_mode_actual.setStyleSheet(
            "color:#89b4fa; font-size:9px; padding:2px 4px;"
            "background:#1e1e2e; border-radius:3px;"
        )
        self.lbl_source_mode_actual.setWordWrap(True)
        self.lbl_source_mode_actual.setVisible(True)
        srg.addWidget(self.lbl_source_mode_actual)

        # 差异警告标签（默认隐藏，检测到 alpha/亮度差异大时显示）
        self.lbl_source_mode_warn = QLabel("")
        self.lbl_source_mode_warn.setStyleSheet(
            "color:#a6adc8; font-size:9px; padding:3px 4px;"
            "background:#1e1e2e; border:1px solid #585b70; border-radius:3px;"
        )
        self.lbl_source_mode_warn.setWordWrap(True)
        self.lbl_source_mode_warn.setVisible(False)
        srg.addWidget(self.lbl_source_mode_warn)

        _hint_mode = QLabel("手动选择时强制使用指定方式，不受自动检测影响")
        _hint_mode.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_mode.setWordWrap(True)
        srg.addWidget(_hint_mode)

        srg.addWidget(self._hline())

        # 边缘去抖
        srg.addWidget(QLabel("边缘去抖："))
        blur_row = QHBoxLayout()
        self.dspin_blur = QDoubleSpinBox()
        self.dspin_blur.setRange(0.0, 20.0)
        self.dspin_blur.setSingleStep(0.5)
        self.dspin_blur.setValue(0.0)
        self.dspin_blur.setDecimals(1)
        self.dspin_blur.setSuffix(" px")
        blur_row.addWidget(self.dspin_blur)
        btn_reset_blur = QPushButton("重置")
        btn_reset_blur.setFixedWidth(40)
        btn_reset_blur.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_blur.setToolTip("恢复默认值 0.0")
        btn_reset_blur.clicked.connect(lambda: self.dspin_blur.setValue(0.0))
        blur_row.addWidget(btn_reset_blur)
        blur_row.addStretch()
        srg.addLayout(blur_row)
        _hint_blur = QLabel("调大可消除边缘锯齿，但会让主体边界变模糊。0 = 不处理")
        _hint_blur.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_blur.setWordWrap(True)
        srg.addWidget(_hint_blur)

        seq_ctrl_layout.addWidget(seq_recog_group)

        # 主体保留范围
        seq_range_group = CollapsibleGroupBox("主体保留范围", color="#a6e3a1", border_color="#a6e3a1")
        srrg = seq_range_group.content_layout
        srrg.setSpacing(8)

        # 识别灵敏度
        srrg.addWidget(QLabel("识别灵敏度："))
        hit_row = QHBoxLayout()
        self.dspin_hit = QDoubleSpinBox()
        self.dspin_hit.setRange(0.01, 1.0)
        self.dspin_hit.setSingleStep(0.05)
        self.dspin_hit.setValue(0.2)
        self.dspin_hit.setDecimals(2)
        hit_row.addWidget(self.dspin_hit)
        btn_reset_hit = QPushButton("重置")
        btn_reset_hit.setFixedWidth(40)
        btn_reset_hit.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_hit.setToolTip("恢复默认值 0.20")
        btn_reset_hit.clicked.connect(lambda: self.dspin_hit.setValue(0.2))
        hit_row.addWidget(btn_reset_hit)
        hit_row.addStretch()
        srrg.addLayout(hit_row)
        _hint_hit = QLabel("调小=更容易识别到内容（可能误判背景）；调大=只识别明显内容")
        _hint_hit.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_hit.setWordWrap(True)
        srrg.addWidget(_hint_hit)

        srrg.addWidget(self._hline())

        # 主体保留范围
        srrg.addWidget(QLabel("主体保留范围："))
        mask_thresh_row = QHBoxLayout()
        self.dspin_mask_thresh = QDoubleSpinBox()
        self.dspin_mask_thresh.setRange(0.0, 1.0)
        self.dspin_mask_thresh.setSingleStep(0.01)
        self.dspin_mask_thresh.setValue(0.05)
        self.dspin_mask_thresh.setDecimals(2)
        mask_thresh_row.addWidget(self.dspin_mask_thresh)
        btn_reset_mask_thresh = QPushButton("重置")
        btn_reset_mask_thresh.setFixedWidth(40)
        btn_reset_mask_thresh.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_mask_thresh.setToolTip("恢复默认值 0.05")
        btn_reset_mask_thresh.clicked.connect(lambda: self.dspin_mask_thresh.setValue(0.05))
        mask_thresh_row.addWidget(btn_reset_mask_thresh)
        mask_thresh_row.addStretch()
        srrg.addLayout(mask_thresh_row)
        _hint_mask = QLabel("调大=只保留最明显的主体区域；调小=保留更多边缘细节")
        _hint_mask.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_mask.setWordWrap(True)
        srrg.addWidget(_hint_mask)

        seq_ctrl_layout.addWidget(seq_range_group)

        # 生成结果
        seq_gen_group = CollapsibleGroupBox("生成结果", color="#f38ba8", border_color="#f38ba8")
        sgg = seq_gen_group.content_layout
        sgg.setSpacing(8)

        # 反转生长顺序
        self.chk_invert = QCheckBox("反转生长顺序")
        self.chk_invert.setToolTip("勾选后：原本先出现的区域变为白色（后出现），反之亦然")
        sgg.addWidget(self.chk_invert)
        _hint_invert = QLabel("默认：先出现=黑，后出现=白。勾选后颠倒")
        _hint_invert.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_invert.setWordWrap(True)
        sgg.addWidget(_hint_invert)

        self.btn_generate_seq = QPushButton("▶  自动生成灰度图")
        self.btn_generate_seq.setStyleSheet(
            "background:#cba6f7; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        sgg.addWidget(self.btn_generate_seq)

        self.btn_cancel_seq = QPushButton("⏹  取消生成")
        self.btn_cancel_seq.setStyleSheet(
            "background:#f38ba8; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        self.btn_cancel_seq.setVisible(False)
        sgg.addWidget(self.btn_cancel_seq)

        self.lbl_seq_status = QLabel("尚未生成")
        self.lbl_seq_status.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_seq_status.setWordWrap(True)
        sgg.addWidget(self.lbl_seq_status)

        # 生长预览控件（序列帧模式）
        sgg.addWidget(self._hline())
        self._seq_preview_ctrl = self._build_growth_preview_ctrl("seq")
        sgg.addWidget(self._seq_preview_ctrl)

        seq_ctrl_layout.addWidget(seq_gen_group)
        # 注意：不在这里 addWidget，统一放入 _mode_stack

        # ═══════════════════════════════════════════════════════════════
        # ── 单图模式控件区 ──
        # ═══════════════════════════════════════════════════════════════
        self._single_controls = QWidget()
        self._single_controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        single_ctrl_layout = QVBoxLayout(self._single_controls)
        single_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        single_ctrl_layout.setSpacing(8)

        # 导入
        single_import_group = CollapsibleGroupBox("导入单图", color="#89b4fa", border_color="#89b4fa")
        siig = single_import_group.content_layout
        siig.setSpacing(6)

        self.btn_import_single = QPushButton("选择图像文件")
        self.btn_import_single.setStyleSheet(
            "background:#313244; color:#cdd6f4; border-radius:5px; padding:6px;"
        )
        siig.addWidget(self.btn_import_single)

        self.lbl_single_import_info = QLabel("未导入")
        self.lbl_single_import_info.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_single_import_info.setWordWrap(True)
        siig.addWidget(self.lbl_single_import_info)

        single_ctrl_layout.addWidget(single_import_group)

        # 主体范围生成
        single_mask_group = CollapsibleGroupBox("主体范围识别", color="#a6e3a1", border_color="#a6e3a1")
        smg = single_mask_group.content_layout
        smg.setSpacing(6)

        self.btn_mask_from_alpha = QPushButton("按透明度识别主体")
        self.btn_mask_from_alpha.setStyleSheet(
            "background:#313244; color:#cdd6f4; border-radius:5px; padding:5px;"
        )
        smg.addWidget(self.btn_mask_from_alpha)
        _hint_alpha = QLabel("适合有透明通道（Alpha）的贴图")
        _hint_alpha.setStyleSheet("color:#45475a; font-size:9px;")
        smg.addWidget(_hint_alpha)

        smg.addWidget(self._hline())

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("亮度阈值："))
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(128)
        self.lbl_thresh_val = QLabel("128")
        self.lbl_thresh_val.setFixedWidth(36)
        self.lbl_thresh_val.setAlignment(Qt.AlignCenter)
        self.lbl_thresh_val.setStyleSheet("color:#cdd6f4; font-size:11px;")
        # 保留 spin_thresh 作为内部数据接口（隐藏），与 slider 同步
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(0, 255)
        self.spin_thresh.setValue(128)
        self.spin_thresh.setVisible(False)
        self.slider_thresh.valueChanged.connect(self._on_thresh_slider_changed)
        thresh_row.addWidget(self.slider_thresh, 1)
        thresh_row.addWidget(self.lbl_thresh_val)
        smg.addLayout(thresh_row)
        _hint_thresh = QLabel("亮度高于此值的区域视为主体。调小=保留更多区域")
        _hint_thresh.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_thresh.setWordWrap(True)
        smg.addWidget(_hint_thresh)

        self.btn_mask_from_lum = QPushButton("按亮度识别主体")
        self.btn_mask_from_lum.setStyleSheet(
            "background:#313244; color:#cdd6f4; border-radius:5px; padding:5px;"
        )
        smg.addWidget(self.btn_mask_from_lum)

        self.btn_clear_mask = QPushButton("清除主体范围")
        self.btn_clear_mask.setStyleSheet("color:#f38ba8; font-size:10px;")
        smg.addWidget(self.btn_clear_mask)

        single_ctrl_layout.addWidget(single_mask_group)

        # 路径扩散设置
        prop_group = CollapsibleGroupBox("路径扩散设置", color="#89b4fa", border_color="#89b4fa")
        ppg = prop_group.content_layout
        ppg.setSpacing(8)

        # 扩散范围
        ppg.addWidget(QLabel("扩散范围："))
        radius_row = QHBoxLayout()
        self.spin_prop_radius = QSpinBox()
        self.spin_prop_radius.setRange(1, 2048)
        self.spin_prop_radius.setValue(64)
        self.spin_prop_radius.setSuffix(" px")
        radius_row.addWidget(self.spin_prop_radius)
        radius_row.addStretch()
        ppg.addLayout(radius_row)
        _hint_radius = QLabel("调大=路径影响更远的区域；调小=只影响路径附近")
        _hint_radius.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_radius.setWordWrap(True)
        ppg.addWidget(_hint_radius)

        ppg.addWidget(self._hline())

        # 路径影响衰减
        ppg.addWidget(QLabel("路径影响衰减："))
        power_row = QHBoxLayout()
        self.dspin_power = QDoubleSpinBox()
        self.dspin_power.setRange(0.1, 10.0)
        self.dspin_power.setSingleStep(0.5)
        self.dspin_power.setValue(2.0)
        self.dspin_power.setDecimals(1)
        power_row.addWidget(self.dspin_power)
        power_row.addStretch()
        ppg.addLayout(power_row)
        _hint_power = QLabel("调大=路径边缘过渡更硬，影响范围更集中；调小=过渡更柔和")
        _hint_power.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_power.setWordWrap(True)
        ppg.addWidget(_hint_power)

        ppg.addWidget(self._hline())

        # 平滑次数
        ppg.addWidget(QLabel("结果平滑次数："))
        smooth_row = QHBoxLayout()
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(0, 20)
        self.spin_smooth.setValue(0)
        smooth_row.addWidget(self.spin_smooth)
        smooth_row.addStretch()
        ppg.addLayout(smooth_row)
        _hint_smooth = QLabel("对结果做模糊平滑，消除锯齿感。0 = 不平滑")
        _hint_smooth.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_smooth.setWordWrap(True)
        ppg.addWidget(_hint_smooth)

        ppg.addWidget(self._hline())

        # 自动补全
        self.chk_fallback = QCheckBox("自动补全没画到的区域")
        self.chk_fallback.setChecked(True)
        ppg.addWidget(self.chk_fallback)
        _hint_fallback = QLabel("勾选后，扩散范围内没有路径的区域会用最近路径的值填充")
        _hint_fallback.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_fallback.setWordWrap(True)
        ppg.addWidget(_hint_fallback)

        # 生成按钮
        self.btn_generate_seed = QPushButton("▶  根据路径生成灰度图")
        self.btn_generate_seed.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        ppg.addWidget(self.btn_generate_seed)

        self.lbl_prop_status = QLabel("尚未生成")
        self.lbl_prop_status.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_prop_status.setWordWrap(True)
        ppg.addWidget(self.lbl_prop_status)

        # 生长预览控件（单图模式）
        ppg.addWidget(self._hline())
        self._single_preview_ctrl = self._build_growth_preview_ctrl("single")
        ppg.addWidget(self._single_preview_ctrl)

        single_ctrl_layout.addWidget(prop_group)
        # 注意：不在这里 addWidget，统一放入 _mode_stack

        # ── 模式切换堆叠区（序列帧 / 单图，同一时间只显示一个）──
        # 用自适应高度的 QStackedWidget，只取当前页面的 sizeHint，
        # 避免折叠后其他页面撑高导致间距过大
        from PySide6.QtWidgets import QStackedWidget

        class _AdaptiveStack(QStackedWidget):
            """sizeHint / minimumSizeHint 只反映当前可见页面，折叠后立即收缩。"""
            def sizeHint(self):
                w = self.currentWidget()
                if w:
                    return w.sizeHint()
                return super().sizeHint()
            def minimumSizeHint(self):
                w = self.currentWidget()
                if w:
                    return w.minimumSizeHint()
                return super().minimumSizeHint()

        self._mode_stack = _AdaptiveStack()
        self._mode_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._mode_stack.addWidget(self._seq_controls)     # index 0
        self._mode_stack.addWidget(self._single_controls)  # index 1
        self._mode_stack.setCurrentIndex(0)
        self._mode_stack.setVisible(False)  # 初始不显示任何一个，等待模式选择
        right.addWidget(self._mode_stack)

        # ── 路径细节叠加（噪波）──
        noise_group = CollapsibleGroupBox("路径细节叠加", color="#89b4fa", border_color="#89b4fa", collapsed=True)
        noise_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        ng = noise_group.content_layout
        ng.setSpacing(4)
        ng.setContentsMargins(0, 0, 0, 0)

        # 启用开关
        self.chk_noise_enable = QCheckBox("启用噪波叠加")
        self.chk_noise_enable.setToolTip(
            "在基础灰度图上叠加噪波，让生长边缘更有变化感\n"
            "类似材质中的 Dissolve / Edge Breakup 效果"
        )
        ng.addWidget(self.chk_noise_enable)

        # ── 折叠容器：勾选启用后才展开 ──
        self._noise_content_widget = QWidget()
        _ncw_layout = QVBoxLayout(self._noise_content_widget)
        _ncw_layout.setContentsMargins(0, 0, 0, 0)
        _ncw_layout.setSpacing(6)
        self._noise_content_widget.setVisible(False)

        _hint_noise_top = QLabel("在生长路径灰度图上叠加噪波，打散边缘，增加细节变化")
        _hint_noise_top.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_noise_top.setWordWrap(True)
        _ncw_layout.addWidget(_hint_noise_top)

        _ncw_layout.addWidget(self._hline())

        # 噪波来源
        _ncw_layout.addWidget(QLabel("噪波来源："))
        self.combo_noise_source = QComboBox()
        self.combo_noise_source.addItems(["内置噪波（Perlin）", "导入噪波贴图"])
        _ncw_layout.addWidget(self.combo_noise_source)

        # 导入噪波贴图按钮（仅"导入"模式下可用）
        self._noise_import_widget = QWidget()
        niw = QVBoxLayout(self._noise_import_widget)
        niw.setContentsMargins(0, 0, 0, 0)
        niw.setSpacing(4)
        self.btn_import_noise = QPushButton("选择噪波贴图文件")
        self.btn_import_noise.setMinimumHeight(28)
        self.btn_import_noise.setStyleSheet(
            "background:#313244; color:#cdd6f4; border-radius:5px; padding:6px 10px;"
        )
        niw.addWidget(self.btn_import_noise)
        self.lbl_noise_import_info = QLabel("未导入")
        self.lbl_noise_import_info.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_noise_import_info.setWordWrap(True)
        niw.addWidget(self.lbl_noise_import_info)
        self._noise_import_widget.setVisible(False)
        _ncw_layout.addWidget(self._noise_import_widget)

        _ncw_layout.addWidget(self._hline())

        # 叠加强度
        _ncw_layout.addWidget(QLabel("叠加强度："))
        noise_strength_row = QHBoxLayout()
        self.dspin_noise_strength = QDoubleSpinBox()
        self.dspin_noise_strength.setRange(0.0, 1.0)
        self.dspin_noise_strength.setSingleStep(0.05)
        self.dspin_noise_strength.setValue(0.3)
        self.dspin_noise_strength.setDecimals(2)
        noise_strength_row.addWidget(self.dspin_noise_strength)
        btn_reset_noise_strength = QPushButton("重置")
        btn_reset_noise_strength.setFixedWidth(40)
        btn_reset_noise_strength.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_noise_strength.clicked.connect(lambda: self.dspin_noise_strength.setValue(0.3))
        noise_strength_row.addWidget(btn_reset_noise_strength)
        noise_strength_row.addStretch()
        _ncw_layout.addLayout(noise_strength_row)
        _hint_ns = QLabel("调大=噪波影响更强，边缘打散更明显；调小=接近原始灰度图")
        _hint_ns.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_ns.setWordWrap(True)
        _ncw_layout.addWidget(_hint_ns)

        _ncw_layout.addWidget(self._hline())

        # 噪波旋转
        _ncw_layout.addWidget(QLabel("噪波旋转："))
        noise_rotate_row = QHBoxLayout()
        self.slider_noise_rotate = QSlider(Qt.Horizontal)
        self.slider_noise_rotate.setRange(0, 360)
        self.slider_noise_rotate.setValue(0)
        self.slider_noise_rotate.setTickPosition(QSlider.TicksBelow)
        self.slider_noise_rotate.setTickInterval(45)
        noise_rotate_row.addWidget(self.slider_noise_rotate)
        self.lbl_noise_rotate_val = QLabel("0°")
        self.lbl_noise_rotate_val.setFixedWidth(35)
        self.lbl_noise_rotate_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        noise_rotate_row.addWidget(self.lbl_noise_rotate_val)
        btn_reset_noise_rotate = QPushButton("重置")
        btn_reset_noise_rotate.setFixedWidth(40)
        btn_reset_noise_rotate.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_noise_rotate.clicked.connect(lambda: self.slider_noise_rotate.setValue(0))
        noise_rotate_row.addWidget(btn_reset_noise_rotate)
        _ncw_layout.addLayout(noise_rotate_row)
        _hint_nr = QLabel("旋转噪波贴图的采样角度（0°~360°）")
        _hint_nr.setStyleSheet("color:#45475a; font-size:9px;")
        _ncw_layout.addWidget(_hint_nr)

        _ncw_layout.addWidget(self._hline())

        # 噪波缩放 X / Y
        _ncw_layout.addWidget(QLabel("噪波缩放："))
        noise_scale_row = QHBoxLayout()
        noise_scale_row.addWidget(QLabel("X："))
        self.dspin_noise_scale_x = QLineEdit("4.0")
        self.dspin_noise_scale_x.setFixedWidth(50)
        noise_scale_row.addWidget(self.dspin_noise_scale_x)
        btn_reset_nsx = QPushButton("重置")
        btn_reset_nsx.setFixedWidth(40)
        btn_reset_nsx.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_nsx.clicked.connect(lambda: self.dspin_noise_scale_x.setText("4.0"))
        noise_scale_row.addWidget(btn_reset_nsx)
        noise_scale_row.addWidget(QLabel("  Y："))
        self.dspin_noise_scale_y = QLineEdit("4.0")
        self.dspin_noise_scale_y.setFixedWidth(50)
        noise_scale_row.addWidget(self.dspin_noise_scale_y)
        btn_reset_nsy = QPushButton("重置")
        btn_reset_nsy.setFixedWidth(40)
        btn_reset_nsy.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_nsy.clicked.connect(lambda: self.dspin_noise_scale_y.setText("4.0"))
        noise_scale_row.addWidget(btn_reset_nsy)
        noise_scale_row.addStretch()
        _ncw_layout.addLayout(noise_scale_row)
        _hint_nsc = QLabel("调大=噪波纹理更粗；调小=噪波纹理更细密（X/Y 可分别控制水平和垂直方向）")
        _hint_nsc.setStyleSheet("color:#45475a; font-size:9px;")
        _ncw_layout.addWidget(_hint_nsc)

        _ncw_layout.addWidget(self._hline())

        # 噪波偏移 X / Y
        _ncw_layout.addWidget(QLabel("噪波偏移："))
        noise_offset_row = QHBoxLayout()
        noise_offset_row.addWidget(QLabel("X："))
        self.dspin_noise_offset_x = QDoubleSpinBox()
        self.dspin_noise_offset_x.setRange(-100.0, 100.0)
        self.dspin_noise_offset_x.setSingleStep(0.5)
        self.dspin_noise_offset_x.setValue(0.0)
        self.dspin_noise_offset_x.setDecimals(1)
        noise_offset_row.addWidget(self.dspin_noise_offset_x)
        btn_reset_nx = QPushButton("重置")
        btn_reset_nx.setFixedWidth(40)
        btn_reset_nx.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_nx.clicked.connect(lambda: self.dspin_noise_offset_x.setValue(0.0))
        noise_offset_row.addWidget(btn_reset_nx)
        noise_offset_row.addStretch()
        _ncw_layout.addLayout(noise_offset_row)

        noise_offset_y_row = QHBoxLayout()
        noise_offset_y_row.addWidget(QLabel("Y："))
        self.dspin_noise_offset_y = QDoubleSpinBox()
        self.dspin_noise_offset_y.setRange(-100.0, 100.0)
        self.dspin_noise_offset_y.setSingleStep(0.5)
        self.dspin_noise_offset_y.setValue(0.0)
        self.dspin_noise_offset_y.setDecimals(1)
        noise_offset_y_row.addWidget(self.dspin_noise_offset_y)
        btn_reset_ny = QPushButton("重置")
        btn_reset_ny.setFixedWidth(40)
        btn_reset_ny.setStyleSheet("color:#6c7086; font-size:9px; padding:2px;")
        btn_reset_ny.clicked.connect(lambda: self.dspin_noise_offset_y.setValue(0.0))
        noise_offset_y_row.addWidget(btn_reset_ny)
        noise_offset_y_row.addStretch()
        _ncw_layout.addLayout(noise_offset_y_row)
        _hint_noff = QLabel("平移噪波采样位置，获得不同的噪波图案")
        _hint_noff.setStyleSheet("color:#45475a; font-size:9px;")
        _ncw_layout.addWidget(_hint_noff)

        _ncw_layout.addWidget(self._hline())

        # 反相噪波
        self.chk_noise_invert = QCheckBox("反相噪波")
        self.chk_noise_invert.setToolTip("将噪波图取反（1 - noise），改变叠加方向")
        _ncw_layout.addWidget(self.chk_noise_invert)

        _ncw_layout.addWidget(self._hline())

        # 应用按钮 + 状态
        self.btn_apply_noise = QPushButton("▶  应用噪波叠加")
        self.btn_apply_noise.setStyleSheet(
            "background:#89b4fa; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        _ncw_layout.addWidget(self.btn_apply_noise)

        self.lbl_noise_status = QLabel("未启用")
        self.lbl_noise_status.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_noise_status.setWordWrap(True)
        _ncw_layout.addWidget(self.lbl_noise_status)

        ng.addWidget(self._noise_content_widget)

        right.addWidget(noise_group)

        # ── 导出区（始终显示）──
        export_group = CollapsibleGroupBox("导出", color="#a6adc8", border_color="#585b70")
        eg = export_group.content_layout
        eg.setSpacing(6)

        eg.addWidget(QLabel("导出命名（tag）："))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如 fire → T_fire_Growth.png")
        self.name_input.setValidator(
            QRegularExpressionValidator(QRegularExpression("^[A-Za-z0-9_]*$"))
        )
        eg.addWidget(self.name_input)

        self.name_preview = QLabel("预览：-")
        self.name_preview.setStyleSheet("color:#a6e3a1; font-weight:700; font-size:11px;")
        eg.addWidget(self.name_preview)

        self.btn_apply_name = QPushButton("应用命名")
        eg.addWidget(self.btn_apply_name)

        eg.addWidget(self._hline())

        # 导出尺寸选择
        eg.addWidget(QLabel("导出尺寸："))
        _pow2 = ["原始尺寸", "32", "64", "128", "256", "512", "1024", "2048"]
        size_row = QHBoxLayout()
        size_row.setSpacing(2)
        _lbl_w = QLabel("宽:")
        _lbl_w.setFixedWidth(20)
        size_row.addWidget(_lbl_w)
        self.export_size_w = QComboBox()
        self.export_size_w.setEditable(True)
        self.export_size_w.addItems(_pow2)
        self.export_size_w.setCurrentText("原始尺寸")
        self.export_size_w.setMinimumWidth(90)
        size_row.addWidget(self.export_size_w, 1)
        size_row.addSpacing(4)
        _lbl_h = QLabel("高:")
        _lbl_h.setFixedWidth(20)
        size_row.addWidget(_lbl_h)
        self.export_size_h = QComboBox()
        self.export_size_h.setEditable(True)
        self.export_size_h.addItems(_pow2)
        self.export_size_h.setCurrentText("原始尺寸")
        self.export_size_h.setMinimumWidth(90)
        size_row.addWidget(self.export_size_h, 1)
        eg.addLayout(size_row)
        _hint_size = QLabel("选择\"原始尺寸\"则按原图分辨率导出，也可输入自定义数值")
        _hint_size.setStyleSheet("color:#45475a; font-size:9px;")
        _hint_size.setWordWrap(True)
        eg.addWidget(_hint_size)

        eg.addWidget(self._hline())

        self.btn_export = QPushButton("导出基础灰度图 PNG")
        self.btn_export.setStyleSheet(
            "background:#89b4fa; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        self.btn_export.setToolTip("导出原始生成的基础灰度图（不含噪波叠加）")
        eg.addWidget(self.btn_export)

        self.btn_export_final = QPushButton("导出叠加噪波后的灰度图 PNG")
        self.btn_export_final.setStyleSheet(
            "background:#313244; color:#cdd6f4; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        self.btn_export_final.setToolTip("导出叠加噪波后的最终灰度图（需先启用并应用噪波叠加）")
        eg.addWidget(self.btn_export_final)

        self.lbl_export_info = QLabel("尚未导出")
        self.lbl_export_info.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_export_info.setWordWrap(True)
        eg.addWidget(self.lbl_export_info)

        right.addWidget(export_group)

        right_scroll.setWidget(right_inner)

        # 左栏包入 QScrollArea，内容超出时可滚动
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        left_scroll.setMinimumWidth(140)
        left_scroll.setMaximumWidth(220)

        # ===== 组装到 QSplitter =====
        splitter.addWidget(left_scroll)
        splitter.addWidget(mid_widget)
        splitter.addWidget(right_scroll)
        # 初始比例：左160 / 中自适应 / 右300
        splitter.setSizes([160, 9999, 300])
        # 左栏和右栏不随窗口拉伸，中栏吸收所有多余空间
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color:#383850;")
        return f

    def _build_growth_preview_ctrl(self, tag: str) -> QWidget:
        """
        构建生长预览控件区（序列帧/单图共用同一套逻辑，tag 用于区分属性名）。
        返回一个 QWidget，包含：
          - 显示生长预览 开关
          - 生长进度 滑条 + 数值
          - 羽化 spinbox
        """
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        # 标题行 + 开关
        title_row = QHBoxLayout()
        title_lbl = QLabel("🎬 生长预览")
        title_lbl.setStyleSheet("color:#cdd6f4; font-size:11px; font-weight:700;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        chk = QCheckBox("显示生长预览")
        chk.setToolTip("勾选后画布切换到生长预览模式，拖动进度条实时查看显隐效果")
        title_row.addWidget(chk)
        layout.addLayout(title_row)

        _hint = QLabel("根据灰度图驱动显隐，判断生长路径顺序是否正确")
        _hint.setStyleSheet("color:#45475a; font-size:9px;")
        _hint.setWordWrap(True)
        layout.addWidget(_hint)

        # 进度条
        prog_row = QHBoxLayout()
        prog_lbl = QLabel("生长进度：")
        prog_lbl.setFixedWidth(64)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(0)
        slider.setEnabled(False)
        val_lbl = QLabel("0.00")
        val_lbl.setFixedWidth(36)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_row.addWidget(prog_lbl)
        prog_row.addWidget(slider, 1)
        prog_row.addWidget(val_lbl)
        layout.addLayout(prog_row)

        _hint_prog = QLabel("拖动查看不同时刻的显隐状态")
        _hint_prog.setStyleSheet("color:#45475a; font-size:9px;")
        layout.addWidget(_hint_prog)

        # 羽化控件（已隐藏，保留属性引用以兼容旧代码）
        feather_spin = QDoubleSpinBox()
        feather_spin.setValue(0.0)
        feather_spin.setVisible(False)  # 严格显隐模式，不再需要羽化

        # 把控件引用存到 self，方便后续访问
        if tag == "seq":
            self._seq_preview_chk = chk
            self._seq_preview_slider = slider
            self._seq_preview_val_lbl = val_lbl
            self._seq_preview_feather = feather_spin
        else:
            self._single_preview_chk = chk
            self._single_preview_slider = slider
            self._single_preview_val_lbl = val_lbl
            self._single_preview_feather = feather_spin

        # 开关联动：勾选时启用控件并切换画布模式
        def _on_chk(state, _slider=slider):
            enabled = bool(state)
            _slider.setEnabled(enabled)
            if enabled:
                self.combo_overlay.setCurrentIndex(5)  # 生长预览
            else:
                self.combo_overlay.setCurrentIndex(0)  # 原图

        chk.stateChanged.connect(_on_chk)

        # 进度条联动：拖动时实时刷新
        def _on_slider(v, _val_lbl=val_lbl):
            fv = v / 1000.0
            _val_lbl.setText(f"{fv:.2f}")
            self._refresh_canvas_overlay()

        slider.valueChanged.connect(_on_slider)

        # 羽化联动：已移除（严格显隐模式不需要）
        # feather_spin.valueChanged.connect(lambda _: self._refresh_canvas_overlay())

        return w

    # ── 信号连接 ──────────────────────────────────────────────────────
    def _connect_signals(self):
        # 模式切换
        self.btn_mode_seq.clicked.connect(lambda: self._switch_mode("seq"))
        self.btn_mode_single.clicked.connect(lambda: self._switch_mode("single"))
        self._adv_toggle_btn.clicked.connect(self._toggle_adv_preview)

        # 导入
        self.btn_import_single.clicked.connect(self._import_single)
        self.btn_import_seq.clicked.connect(self._import_sequence)
        self.btn_import_video.clicked.connect(self._import_video)

        # Mask
        self.btn_mask_from_alpha.clicked.connect(self._gen_mask_from_alpha)
        self.btn_mask_from_lum.clicked.connect(self._gen_mask_from_luminance)
        self.btn_clear_mask.clicked.connect(self._clear_mask)
        # 亮度阈值变化时自动刷新 mask（仅在已有 source_image 时触发）
        self.spin_thresh.valueChanged.connect(
            lambda: self._thresh_timer.start() if self.source_image is not None else None
        )

        # 笔刷
        self.slider_brush.valueChanged.connect(self._on_brush_slider)
        self.edit_brush.editingFinished.connect(self._on_brush_edit)
        self.slider_value.valueChanged.connect(self._on_value_slider)
        self.edit_value.editingFinished.connect(self._on_value_edit)
        self.slider_hardness.valueChanged.connect(self._on_hardness_slider)
        self.btn_clear_seed.clicked.connect(self._clear_seed)

        # 画布叠加
        self.combo_overlay.currentIndexChanged.connect(self._refresh_canvas_overlay)

        # 内容识别方式切换时更新状态标签
        self.combo_source_mode.currentIndexChanged.connect(self._on_source_mode_changed)

        # Ctrl+Z 撤销手绘路径
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.setContext(Qt.WidgetWithChildrenShortcut)
        undo_sc.activated.connect(self._undo_stroke)

        # 生成
        self.btn_generate_seq.clicked.connect(self._generate_from_sequence)
        self.btn_cancel_seq.clicked.connect(self._cancel_seq_generation)
        self.btn_generate_seed.clicked.connect(self._generate_from_seed)

        # ── 参数变化自动重新生成（防抖 600ms）──
        # 序列帧模式：内容识别设置 + 主体保留范围 + 反转
        self.combo_source_mode.currentIndexChanged.connect(self._schedule_auto_regen)
        self.dspin_blur.valueChanged.connect(self._schedule_auto_regen)
        self.dspin_hit.valueChanged.connect(self._schedule_auto_regen)
        self.dspin_mask_thresh.valueChanged.connect(self._schedule_auto_regen)
        self.chk_invert.stateChanged.connect(self._schedule_auto_regen)

        # 命名 & 导出
        self.name_input.textChanged.connect(self._on_name_changed)
        self.btn_apply_name.clicked.connect(self._apply_name)
        self.btn_export.clicked.connect(self._export_gray)
        self.btn_export_final.clicked.connect(self._export_final_gray)

        # 噪波叠加
        self.chk_noise_enable.stateChanged.connect(self._on_noise_enable_changed)
        self.combo_noise_source.currentIndexChanged.connect(self._on_noise_source_changed)
        self.btn_import_noise.clicked.connect(self._import_noise_texture)
        self.btn_apply_noise.clicked.connect(self._apply_noise_overlay)

        # 噪波参数实时预览（防抖 400ms）
        self._noise_timer = QTimer(self)
        self._noise_timer.setSingleShot(True)
        self._noise_timer.timeout.connect(self._apply_noise_overlay_silent)
        self.dspin_noise_strength.valueChanged.connect(self._schedule_noise_regen)
        self.dspin_noise_scale_x.textChanged.connect(self._schedule_noise_regen)
        self.dspin_noise_scale_y.textChanged.connect(self._schedule_noise_regen)
        self.dspin_noise_offset_x.valueChanged.connect(self._schedule_noise_regen)
        self.dspin_noise_offset_y.valueChanged.connect(self._schedule_noise_regen)
        self.chk_noise_invert.stateChanged.connect(self._schedule_noise_regen)
        self.combo_noise_source.currentIndexChanged.connect(self._schedule_noise_regen)
        self.slider_noise_rotate.valueChanged.connect(self._on_noise_rotate_changed)
        self.slider_noise_rotate.valueChanged.connect(self._schedule_noise_regen)

    def _on_source_mode_changed(self, idx: int):
        """内容识别方式下拉框切换时，更新状态标签显示。"""
        if idx == 0:  # 自动
            self.lbl_source_mode_actual.setText("自动模式：尚未生成（生成后显示实际使用方式）")
            self.lbl_source_mode_actual.setStyleSheet(
                "color:#89b4fa; font-size:9px; padding:2px 4px;"
                "background:#1e1e2e; border-radius:3px;"
            )
            self.lbl_source_mode_actual.setVisible(True)
        else:  # 手动指定
            mode_name = "透明度" if idx == 1 else "亮度"
            self.lbl_source_mode_actual.setText(
                f"✓ 已手动指定：强制使用{mode_name}识别，不受自动检测影响"
            )
            self.lbl_source_mode_actual.setStyleSheet(
                "color:#a6e3a1; font-size:9px; padding:2px 4px;"
                "background:#1e2a1e; border-radius:3px;"
            )
            self.lbl_source_mode_actual.setVisible(True)
            # 手动指定时隐藏差异警告
            self.lbl_source_mode_warn.setVisible(False)

    # ── 模式切换 ──────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        """切换工作模式：'seq' = 序列帧，'single' = 单图手绘。"""
        is_seq = (mode == "seq")
        is_single = (mode == "single")

        self.btn_mode_seq.setChecked(is_seq)
        self.btn_mode_single.setChecked(is_single)

        # 控件区切换（QStackedWidget，同一时间只有一个控件占位，避免 geometry 遮挡）
        if is_seq:
            self._mode_stack.setCurrentIndex(0)
            self._mode_stack.setVisible(True)
        elif is_single:
            self._mode_stack.setCurrentIndex(1)
            self._mode_stack.setVisible(True)
        else:
            self._mode_stack.setVisible(False)

        # 左栏预览区显隐
        self._seq_preview_widget.setVisible(is_seq)
        self._single_preview_widget.setVisible(is_single)

        # 画布下方笔刷控制区（仅单图模式显示）
        self._brush_group.setVisible(is_single)

        # 画布提示文字
        if is_single:
            self.lbl_canvas_hint.setText("滚轮缩放 \xb7 右键拖拽 \xb7 F 适配 \xb7 左键画路径")
            self.lbl_mode_hint.setText("在画布上左键拖拽绘制生长路径")
        elif is_seq:
            self.lbl_canvas_hint.setText("滚轮缩放 \xb7 右键拖拽 \xb7 F 适配")
            self.lbl_mode_hint.setText('导入序列帧后点击"自动生成灰度图"')
        else:
            self.lbl_canvas_hint.setText("滚轮缩放 \xb7 右键拖拽 \xb7 F 适配")
            self.lbl_mode_hint.setText("请先选择工作模式")

        # 画笔圆圈预览：仅单图模式下显示
        self.canvas._show_cursor = is_single
        # 切换模式时重置橡皮擦状态
        if not is_single:
            self.btn_eraser.setChecked(False)

    def _toggle_adv_preview(self):
        """折叠/展开高级预览区。"""
        visible = self._adv_preview_widget.isVisible()
        self._adv_preview_widget.setVisible(not visible)
        self._adv_toggle_btn.setText(
            "▼ 高级预览（算法中间图）" if not visible else "▶ 高级预览（算法中间图）"
        )

    # ── 导入 ──────────────────────────────────────────────────────────
    def _import_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入单图", "",
            "图像文件 (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
        )
        if not path:
            return
        self._load_single_from_path(path)

    def _load_single_from_path(self, path: str):
        """从给定路径加载单图（供按钮导入和拖拽导入共用）。"""
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as ex:
            QMessageBox.warning(self, "导入失败", str(ex))
            return

        self.source_image = img
        self.sequence_frames = []
        self._seq_file_paths = []
        self._src_path = path
        self._output_basename = os.path.splitext(os.path.basename(path))[0]

        h, w = img.height, img.width
        # 计算预览缩放因子
        self._preview_factor = _preview_scale_factor(w, h)
        self._checker_cache = None  # 清除缓存

        self.seed_map = np.full((h, w), -1.0, dtype=np.float32)
        self.gray_map = np.zeros((h, w), dtype=np.float32)
        self.mask_map = None

        # 预览使用缩放代理
        f = self._preview_factor
        preview_img = _downscale_pil(img, f) if f < 1.0 else img
        px = pil_to_qpixmap(preview_img)
        # 同步到单图预览区
        self.lbl_src_single.set_pixmap(px)
        self.lbl_mask_single.set_pixmap(None)
        self.lbl_gray_single.set_pixmap(None)
        # 也同步到序列帧预览区的原图（兼容）
        self.lbl_src.set_pixmap(px)
        self.canvas.set_source(px)
        self.canvas._fit_to_view()
        self.lbl_single_import_info.setText(
            f"{os.path.basename(path)}\n尺寸：{w}×{h}"
        )
        if not self.name_input.text():
            self.name_input.setText(self._output_basename)
        # 拖拽导入时自动切换到单图模式
        if self.btn_mode_seq.isChecked():
            self._switch_mode("single")
        elif not self._mode_stack.isVisible():
            self._switch_mode("single")
        self._refresh_canvas_overlay()

    def _on_canvas_drop(self, paths: list):
        """
        画布拖拽文件回调：
        - 单个视频文件 → 走视频导入流程
        - 单张图片 → 直接导入
        - 多张图片 → 提示用文件夹导入
        """
        # 先检查是否有视频文件
        video_paths = [
            p for p in paths
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_VIDEO_EXTS
        ]
        if video_paths:
            # 拖入了视频文件，取第一个
            self._load_video_from_path(video_paths[0])
            return

        # 过滤出支持的图片文件
        img_paths = [
            p for p in paths
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS
        ]
        if not img_paths:
            QMessageBox.information(
                self, "提示",
                "未检测到支持的文件。\n"
                "支持图片格式：PNG、JPG、JPEG、TGA、BMP、WEBP\n"
                "支持视频格式：MP4、AVI、MOV、WEBM、MKV、FLV"
            )
            return
        if len(img_paths) == 1:
            self._load_single_from_path(img_paths[0])
        else:
            QMessageBox.information(
                self, "提示",
                f"检测到 {len(img_paths)} 张图片。\n\n"
                "多图导入请使用「序列帧自动生成」模式，\n"
                "点击「选择序列帧文件夹」按钮导入整个文件夹。"
            )

    def _import_sequence(self):
        folder = QFileDialog.getExistingDirectory(self, "选择序列帧文件夹")
        if not folder:
            return
        files = _sorted_image_files(folder)
        if not files:
            QMessageBox.warning(self, "导入失败", "文件夹内未找到支持的图像文件。")
            return

        # 只加载首帧和尾帧用于预览，保存路径列表延迟加载
        try:
            first_frame = Image.open(files[0]).convert("RGBA")
            last_frame = Image.open(files[-1]).convert("RGBA") if len(files) > 1 else first_frame
        except Exception as ex:
            QMessageBox.warning(self, "导入失败", str(ex))
            return

        # 保存路径列表，实际帧在生成时按需加载
        self._seq_file_paths = files
        self._video_path = None          # 清空视频路径（序列帧与视频互斥）
        self._video_frame_count = 0
        self._video_sample_widget.setVisible(False)  # 隐藏视频帧采样控件
        self.sequence_frames = []  # 清空，生成时再按需加载
        # source_image 使用最后一帧（内容最完整），生长预览依赖此帧的 alpha 做纹理显隐
        self.source_image = last_frame
        self._src_path = folder
        self._output_basename = os.path.basename(folder.rstrip("/\\"))
        self._seq_generated = False  # 重置生成标记

        h, w = last_frame.height, last_frame.width
        # 计算预览缩放因子
        self._preview_factor = _preview_scale_factor(w, h)
        self._checker_cache = None  # 清除缓存

        self.seed_map = np.full((h, w), -1.0, dtype=np.float32)
        self.gray_map = np.zeros((h, w), dtype=np.float32)
        self.mask_map = None
        self.base_gray_map = None
        self.final_gray_map = None
        self.noise_map = None

        # 预览使用缩放代理
        f = self._preview_factor
        preview_first = _downscale_pil(first_frame, f) if f < 1.0 else first_frame
        preview_last = _downscale_pil(last_frame, f) if f < 1.0 else last_frame
        first_px = pil_to_qpixmap(preview_first)
        last_px = pil_to_qpixmap(preview_last)
        self.lbl_src.set_pixmap(first_px)
        self.lbl_last_frame.set_pixmap(last_px)
        self.lbl_mask.set_pixmap(None)
        self.lbl_gray.set_pixmap(None)
        self.lbl_presence.set_pixmap(None)
        self.lbl_envelope.set_pixmap(None)
        # 画布底图使用最后一帧（与 source_image 一致，内容最完整）
        self.canvas.set_source(last_px)
        self.canvas._fit_to_view()
        self.lbl_seq_import_info.setText(
            f"{len(files)} 帧 · {os.path.basename(folder)}\n尺寸：{w}×{h}"
        )
        self.lbl_seq_status.setText(f"已导入 {len(files)} 帧，点击按钮生成灰度图")
        if not self.name_input.text():
            self.name_input.setText(self._output_basename)
        self._refresh_canvas_overlay()

    # ── 视频导入 ───────────────────────────────────────────────────────
    def _import_video(self):
        """从视频文件导入帧（仅读取首尾帧预览 + 记录路径和帧数）。"""
        global _HAS_CV2
        if not _HAS_CV2:
            import sys
            if getattr(sys, 'frozen', False):
                # 打包环境：无法动态安装，提示用户该版本不支持
                QMessageBox.warning(
                    self, "缺少依赖",
                    "当前版本未包含视频导入所需的 opencv-python 库。\n\n"
                    "请联系开发者获取包含视频支持的新版本，\n"
                    "或将视频先转为序列帧图片后使用「选择序列帧文件夹」导入。"
                )
                return

            # 开发环境：尝试自动安装
            reply = QMessageBox.question(
                self, "缺少依赖",
                "视频导入需要 opencv-python 库，是否自动安装？\n\n"
                "点击「Yes」将自动执行安装\n"
                "点击「No」取消操作",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return

            # ── 非阻塞安装 opencv-python ──
            import subprocess
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "opencv-python"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except Exception as ex:
                QMessageBox.critical(self, "启动失败", f"无法启动 pip：\n{str(ex)}")
                return

            # 带取消按钮的进度对话框
            from PySide6.QtWidgets import QProgressDialog
            dlg = QProgressDialog("正在安装 opencv-python，请稍候...", "取消安装", 0, 0, self)
            dlg.setWindowTitle("正在安装")
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setValue(0)

            # 用户点取消 → 终止子进程
            def _on_cancel():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            dlg.canceled.connect(_on_cancel)

            # QTimer 轮询子进程状态（每 200ms）
            timer = QTimer(self)
            _cancel_handled = [False]  # 用列表包装以便闭包内修改

            def _poll_install():
                global _HAS_CV2
                if dlg.wasCanceled():
                    timer.stop()
                    if not _cancel_handled[0]:
                        _cancel_handled[0] = True
                        dlg.close()
                        QMessageBox.information(self, "已取消", "安装已取消，不影响其他功能使用。")
                    return
                if proc.poll() is not None:
                    # 子进程结束
                    timer.stop()
                    dlg.close()
                    if proc.returncode != 0:
                        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")[:500]
                        QMessageBox.critical(
                            self, "安装失败",
                            f"pip install opencv-python 失败：\n{stderr_text}"
                        )
                        return
                    # 安装完成，严格验证能否导入
                    try:
                        import importlib
                        cv2_mod = importlib.import_module("cv2")
                        # 验证核心功能可用
                        _ = cv2_mod.VideoCapture
                        import cv2  # noqa: F811 — 写入当前模块命名空间
                        _HAS_CV2 = True
                        QMessageBox.information(self, "安装成功",
                                                "opencv-python 安装成功！\n请再次点击「导入视频」按钮。")
                    except Exception as ex:
                        QMessageBox.critical(
                            self, "导入失败",
                            f"opencv-python 已安装但无法正常导入：\n{str(ex)}\n\n"
                            "建议手动执行：pip install --force-reinstall opencv-python"
                        )
                    return
            timer.timeout.connect(_poll_install)
            timer.start(200)
            return  # 先返回，等安装完成后用户再次点击按钮即可

        path, _ = QFileDialog.getOpenFileName(
            self, "导入视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.webm *.mkv *.flv);;所有文件 (*)"
        )
        if not path:
            return
        self._load_video_from_path(path)

    def _load_video_from_path(self, path: str):
        """从指定路径加载视频文件（供按钮导入和拖拽导入共用）。"""
        global _HAS_CV2
        if not _HAS_CV2:
            import sys
            if getattr(sys, 'frozen', False):
                QMessageBox.warning(
                    self, "缺少依赖",
                    "当前版本未包含视频导入所需的 opencv-python 库。\n\n"
                    "请联系开发者获取包含视频支持的新版本，\n"
                    "或将视频先转为序列帧图片后使用「选择序列帧文件夹」导入。"
                )
            else:
                QMessageBox.warning(
                    self, "缺少依赖",
                    "视频导入需要 opencv-python 库。\n"
                    "请先通过「导入视频」按钮安装 opencv-python 后再试。"
                )
            return
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("无法打开视频文件")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError("视频帧数为 0 或无法读取")

            # 读取首帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bgr_first = cap.read()
            if not ret:
                raise RuntimeError("无法读取视频首帧")
            first_frame = self._cv2_to_pil(bgr_first)

            # 读取尾帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, bgr_last = cap.read()
            if not ret:
                # 某些视频 seek 到最后一帧可能失败，尝试倒退几帧
                for offset in range(2, min(10, total_frames)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - offset)
                    ret, bgr_last = cap.read()
                    if ret:
                        break
                if not ret:
                    bgr_last = bgr_first  # fallback 到首帧
            last_frame = self._cv2_to_pil(bgr_last)

            cap.release()
        except Exception as ex:
            QMessageBox.warning(self, "导入失败", str(ex))
            return

        # 保存视频信息
        self._video_path = path
        self._video_frame_count = total_frames
        self._seq_file_paths = []  # 清空序列帧路径（互斥）
        self.sequence_frames = []
        self._seq_generated = False

        # 使用尾帧作为 source_image（内容最完整）
        self.source_image = last_frame
        self._src_path = path
        self._output_basename = os.path.splitext(os.path.basename(path))[0]

        h, w = last_frame.height, last_frame.width
        self._preview_factor = _preview_scale_factor(w, h)
        self._checker_cache = None

        self.seed_map = np.full((h, w), -1.0, dtype=np.float32)
        self.gray_map = np.zeros((h, w), dtype=np.float32)
        self.mask_map = None
        self.base_gray_map = None
        self.final_gray_map = None
        self.noise_map = None

        # 预览
        f = self._preview_factor
        preview_first = _downscale_pil(first_frame, f) if f < 1.0 else first_frame
        preview_last = _downscale_pil(last_frame, f) if f < 1.0 else last_frame
        first_px = pil_to_qpixmap(preview_first)
        last_px = pil_to_qpixmap(preview_last)
        self.lbl_src.set_pixmap(first_px)
        self.lbl_last_frame.set_pixmap(last_px)
        self.lbl_mask.set_pixmap(None)
        self.lbl_gray.set_pixmap(None)
        self.lbl_presence.set_pixmap(None)
        self.lbl_envelope.set_pixmap(None)
        self.canvas.set_source(last_px)
        self.canvas._fit_to_view()

        # 显示帧采样控件
        self._video_sample_widget.setVisible(True)
        # 根据帧数自动建议采样间隔
        if total_frames > 500:
            self.spin_video_sample.setValue(5)
        elif total_frames > 200:
            self.spin_video_sample.setValue(2)
        else:
            self.spin_video_sample.setValue(1)

        # 重新打开获取 fps（cap 已在 try 块中 release）
        fps = 0
        try:
            cap2 = cv2.VideoCapture(path)
            fps = cap2.get(cv2.CAP_PROP_FPS)
            cap2.release()
        except Exception:
            fps = 0

        duration = total_frames / fps if fps > 0 else 0
        info_parts = [f"🎬 {os.path.basename(path)}"]
        info_parts.append(f"{total_frames} 帧 · {w}×{h}")
        if fps > 0:
            info_parts.append(f"{fps:.1f}fps · {duration:.1f}s")
        info_parts.append("⚠ 视频无透明通道，建议使用亮度识别")
        self.lbl_seq_import_info.setText("\n".join(info_parts))
        self.lbl_seq_status.setText(f"已导入视频 {total_frames} 帧，点击按钮生成灰度图")

        if not self.name_input.text():
            self.name_input.setText(self._output_basename)
        self._refresh_canvas_overlay()
        # 拖拽导入视频时自动切换到序列帧模式
        if not self.btn_mode_seq.isChecked():
            self._switch_mode("seq")

    @staticmethod
    def _cv2_to_pil(bgr_frame) -> Image.Image:
        """将 OpenCV BGR 帧转为 PIL RGBA Image。"""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, "RGB").convert("RGBA")
        return pil_img

    def _iter_frames_from_video(self, video_path: str, sample_interval: int = 1):
        """视频帧迭代器：逐帧解码，支持跳帧采样，用完即释放。"""
        cap = cv2.VideoCapture(video_path)
        idx = 0
        try:
            while cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    break
                if idx % sample_interval == 0:
                    yield self._cv2_to_pil(bgr)
                idx += 1
        finally:
            cap.release()

    @staticmethod
    def _iter_frames_from_paths(file_paths: List[str]):
        """序列帧文件迭代器：逐张读取，用完即释放。"""
        for p in file_paths:
            yield Image.open(p).convert("RGBA")

    def _sample_frames_for_auto_detect(self, max_samples: int = 16) -> List[Image.Image]:
        """
        为跨帧 auto 判定采样若干帧（不全量加载）。
        支持序列帧路径和视频两种来源。
        """
        samples = []
        if self._video_path and _HAS_CV2:
            total = self._video_frame_count
            interval = self._video_sample_interval
            effective = total // max(interval, 1)
            step = max(1, effective // max_samples)
            cap = cv2.VideoCapture(self._video_path)
            try:
                for i in range(0, effective, step):
                    if len(samples) >= max_samples:
                        break
                    frame_idx = i * interval
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, bgr = cap.read()
                    if ret:
                        samples.append(self._cv2_to_pil(bgr))
            finally:
                cap.release()
        elif self._seq_file_paths:
            total = len(self._seq_file_paths)
            step = max(1, total // max_samples)
            indices = list(range(0, total, step))[:max_samples]
            for idx in indices:
                try:
                    samples.append(Image.open(self._seq_file_paths[idx]).convert("RGBA"))
                except Exception:
                    pass
        return samples

    # ── 序列帧自动生成 ─────────────────────────────────────────────────
    def _generate_from_sequence(self):
        # 如果已有后台线程在运行，忽略重复点击
        if self._seq_gen_worker is not None and self._seq_gen_worker.isRunning():
            return

        # 支持三种来源：视频文件 / 序列帧文件路径 / 已加载的帧列表
        has_video = bool(self._video_path) and _HAS_CV2
        has_paths = bool(self._seq_file_paths)
        has_frames = bool(self.sequence_frames)
        if not has_video and not has_paths and not has_frames:
            QMessageBox.information(self, "提示", "请先导入序列帧文件夹或视频。")
            return

        # 读取参数
        mode_map = {0: "auto", 1: "alpha", 2: "luminance"}
        source_mode = mode_map[self.combo_source_mode.currentIndex()]
        presence_blur = self.dspin_blur.value()
        hit_threshold = self.dspin_hit.value()
        mask_threshold = self.dspin_mask_thresh.value()
        invert = self.chk_invert.isChecked()

        # ── 跨帧 auto 判定（预扫描采样帧，数据量小，主线程完成即可）──
        force_mode = ""
        if source_mode == "auto":
            self.lbl_seq_status.setText("分析帧数据中…")
            QApplication.processEvents()
            sample_frames = self._sample_frames_for_auto_detect(16)
            if len(sample_frames) >= 2:
                force_mode, _ = cross_frame_auto_detect(sample_frames)
            del sample_frames  # 释放采样帧

        # ── 构建帧迭代器和帧数 ──
        if has_video:
            sample_interval = self.spin_video_sample.value()
            self._video_sample_interval = sample_interval
            frame_count = self._video_frame_count // max(sample_interval, 1)
            frame_iter = self._iter_frames_from_video(self._video_path, sample_interval)
        elif has_paths:
            frame_count = len(self._seq_file_paths)
            frame_iter = self._iter_frames_from_paths(self._seq_file_paths)
        else:
            frame_count = len(self.sequence_frames)
            frame_iter = iter(self.sequence_frames)

        # ── 切换 UI 为"生成中"状态 ──
        self.btn_generate_seq.setEnabled(False)
        self.btn_cancel_seq.setVisible(True)
        self.lbl_seq_status.setText(f"后台计算中（{frame_count} 帧），请稍候…")

        # ── 启动后台线程 ──
        worker = _SeqGenWorker(
            frame_iter=frame_iter,
            frame_count=frame_count,
            source_mode=source_mode,
            presence_blur=presence_blur,
            hit_threshold=hit_threshold,
            mask_threshold=mask_threshold,
            invert=invert,
            force_mode=force_mode,
            parent=self,
        )
        worker.progress.connect(self._on_seq_gen_progress)
        worker.finished_ok.connect(self._on_seq_gen_finished)
        worker.finished_err.connect(self._on_seq_gen_error)
        worker.cancelled.connect(self._on_seq_gen_cancelled)
        self._seq_gen_worker = worker
        worker.start()

    def _cancel_seq_generation(self):
        """用户点击取消按钮。"""
        if self._seq_gen_worker is not None and self._seq_gen_worker.isRunning():
            self._seq_gen_worker.cancel()
            self.lbl_seq_status.setText("正在取消…")
            self.btn_cancel_seq.setEnabled(False)

    def _on_seq_gen_progress(self, current: int, total: int):
        """后台线程进度回调（通过信号在主线程执行）。"""
        if current % max(1, total // 20) == 0:  # 每 5% 更新一次
            self.lbl_seq_status.setText(f"计算中… {current}/{total} 帧")

    def _on_seq_gen_finished(self, result: dict):
        """后台线程生成完成回调。"""
        # 恢复 UI 状态
        self.btn_generate_seq.setEnabled(True)
        self.btn_cancel_seq.setVisible(False)
        self.btn_cancel_seq.setEnabled(True)
        self._seq_gen_worker = None

        # 写入核心数据
        self.gray_map = result["gray_map"]
        self.base_gray_map = result["gray_map"].copy()  # 保留原始基础灰度图
        self.mask_map = result["mask_map"]
        # 重置噪波叠加结果（基础图已更新）
        self.final_gray_map = None
        self.noise_map = None

        # source_image 已在导入时设为尾帧，无需再更新
        # 释放可能残留的帧列表
        self.sequence_frames = []

        # 更新左栏预览
        bw_px = np_mask_to_qpixmap(self.mask_map)
        self.lbl_mask.set_pixmap(bw_px)
        self.lbl_mask_single.set_pixmap(bw_px)
        gp = np_gray_to_qpixmap(self.gray_map)
        self.lbl_gray.set_pixmap(gp)
        self.lbl_gray_single.set_pixmap(gp)
        # 重置噪波叠加结果预览
        self.lbl_noise_result.set_pixmap(None)
        self.lbl_noise_result_single.set_pixmap(None)
        # 高级预览（算法中间图）
        self.lbl_presence.set_pixmap(np_gray_to_qpixmap(result["first_presence"]))
        self.lbl_envelope.set_pixmap(np_gray_to_qpixmap(result["last_envelope"]))

        # 刷新画布叠加
        self._refresh_canvas_overlay()

        N = result["frame_count"]
        actual_mode = result["actual_mode"]
        alpha_lum_diff = result.get("alpha_lum_diff", 0.0)
        cross_frame_triggered = result.get("cross_frame_triggered", False)
        hit_threshold = self.dspin_hit.value()
        mask_threshold = self.dspin_mask_thresh.value()
        invert = self.chk_invert.isChecked()

        # ── 更新"内容识别方式"状态标签 ──
        is_auto = (self.combo_source_mode.currentIndex() == 0)
        if is_auto:
            mode_name = "透明度" if actual_mode == "alpha" else "亮度"
            mode_icon = "🔵" if actual_mode == "alpha" else "🟡"

            if cross_frame_triggered:
                # 跨帧分析强制切换：给出更明确的说明
                self.lbl_source_mode_actual.setText(
                    f"自动模式实际使用了：{mode_icon} {mode_name}识别"
                    f"（跨帧分析：亮度恒定 + 透明度变化，已强制切换）"
                )
                self.lbl_source_mode_actual.setStyleSheet(
                    "color:#a6e3a1; font-size:9px; padding:2px 4px;"
                    "background:#1a2e1a; border-radius:3px;"
                )
            else:
                self.lbl_source_mode_actual.setText(
                    f"自动模式实际使用了：{mode_icon} {mode_name}识别"
                )
                self.lbl_source_mode_actual.setStyleSheet(
                    "color:#89dceb; font-size:9px; padding:2px 4px;"
                    "background:#1a2a2e; border-radius:3px;"
                )
            self.lbl_source_mode_actual.setVisible(True)

            # 差异警告：alpha 与亮度均值差 > 0.25 时提示用户检查
            # 若已由跨帧分析强制切换，则不再重复警告
            if alpha_lum_diff > 0.25 and not cross_frame_triggered:
                other_mode = "亮度" if actual_mode == "alpha" else "透明度"
                self.lbl_source_mode_warn.setText(
                    f"⚠ 检测到透明度与亮度差异较大（{alpha_lum_diff:.2f}），"
                    f"当前使用{mode_name}识别。"
                    f"如果结果不理想，可尝试切换为「按{other_mode}识别」。"
                )
                self.lbl_source_mode_warn.setVisible(True)
            else:
                self.lbl_source_mode_warn.setVisible(False)
        else:
            # 手动指定模式，不更新状态标签，隐藏差异警告
            self.lbl_source_mode_warn.setVisible(False)

        mode_label = {"alpha": "透明度", "luminance": "亮度"}.get(actual_mode, actual_mode)
        self.lbl_seq_status.setText(
            f"完成！{N} 帧 · 识别方式={mode_label} · 命中={hit_threshold:.2f}"
            f" · Mask={mask_threshold:.2f}{'  [反转]' if invert else ''}"
        )
        self._seq_generated = True  # 标记已生成过，后续参数变化可自动刷新

    def _on_seq_gen_error(self, err_msg: str):
        """后台线程生成失败回调。"""
        self.btn_generate_seq.setEnabled(True)
        self.btn_cancel_seq.setVisible(False)
        self.btn_cancel_seq.setEnabled(True)
        self._seq_gen_worker = None
        QMessageBox.warning(self, "生成失败", err_msg)
        self.lbl_seq_status.setText("生成失败")

    def _on_seq_gen_cancelled(self):
        """后台线程被取消回调。"""
        self.btn_generate_seq.setEnabled(True)
        self.btn_cancel_seq.setVisible(False)
        self.btn_cancel_seq.setEnabled(True)
        self._seq_gen_worker = None
        self.lbl_seq_status.setText("已取消生成")

    def _schedule_auto_regen(self):
        """参数变化时启动防抖 Timer，600ms 后自动重新生成（仅在已生成过的情况下触发）。"""
        if self._seq_generated and (self.sequence_frames or self._seq_file_paths or self._video_path):
            self._auto_regen_timer.start()

    def _auto_regen_if_ready(self):
        """防抖 Timer 触发：自动重新生成序列帧灰度图。"""
        if self._seq_generated and (self.sequence_frames or self._seq_file_paths or self._video_path):
            # 如果后台线程正在运行，先取消旧的
            if self._seq_gen_worker is not None and self._seq_gen_worker.isRunning():
                self._seq_gen_worker.cancel()
                self._seq_gen_worker.wait()
                self._seq_gen_worker = None
                self.btn_generate_seq.setEnabled(True)
                self.btn_cancel_seq.setVisible(False)
                self.btn_cancel_seq.setEnabled(True)
            self._generate_from_sequence()

    # ── Mask 生成 ──────────────────────────────────────────────────────
    def _require_source(self) -> bool:
        if self.source_image is None:
            QMessageBox.information(self, "提示", "请先导入图像。")
            return False
        return True

    def _gen_mask_from_alpha(self):
        if not self._require_source():
            return
        arr = np.array(self.source_image, dtype=np.float32)  # H×W×4
        self.mask_map = arr[:, :, 3] / 255.0
        self._update_mask_preview()

    def _on_thresh_slider_changed(self, val: int):
        """亮度阈值滑块变化：同步数字显示和隐藏的 spin_thresh。"""
        self.lbl_thresh_val.setText(str(val))
        self.spin_thresh.setValue(val)

    def _gen_mask_from_luminance(self):
        if not self._require_source():
            return
        arr = np.array(self.source_image, dtype=np.float32)  # H×W×4
        # 亮度 = 0.299R + 0.587G + 0.114B
        lum = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114) / 255.0
        thresh = self.spin_thresh.value() / 255.0
        self.mask_map = (lum >= thresh).astype(np.float32)
        self._update_mask_preview()

    def _clear_mask(self):
        self.mask_map = None
        self.lbl_mask.set_pixmap(None)
        self._refresh_canvas_overlay()

    def _update_mask_preview(self):
        """更新 mask 预览：左栏显示黑白二值图，画布叠加用青色蒙层。"""
        if self.mask_map is not None:
            bw_px = np_mask_to_qpixmap(self.mask_map)  # 黑白
            self.lbl_mask.set_pixmap(bw_px)
            self.lbl_mask_single.set_pixmap(bw_px)
        self._refresh_canvas_overlay()

    # ── 手绘路径 → seed_map ───────────────────────────────────────────
    def _push_seed_history(self):
        """将当前 seed_map 的副本压入撤销栈。"""
        if self.seed_map is None:
            return
        self._seed_history.append(self.seed_map.copy())
        if len(self._seed_history) > self._MAX_HISTORY:
            self._seed_history.pop(0)

    def _undo_stroke(self):
        """Ctrl+Z：弹出上一步 seed_map 快照并恢复。"""
        if not self._seed_history:
            return
        self.seed_map = self._seed_history.pop()
        # 同步 gray_map 预览（仅显示 seed 有效区域）
        if self.gray_map is not None:
            self.gray_map[:] = 0.0
            valid = self.seed_map >= 0
            self.gray_map[valid] = self.seed_map[valid]
        # 同步 base_gray_map / final_gray_map，确保生长预览使用最新数据
        if self.base_gray_map is not None:
            self.base_gray_map[:] = 0.0
            valid = self.seed_map >= 0
            self.base_gray_map[valid] = self.seed_map[valid]
        if self.final_gray_map is not None:
            self.final_gray_map[:] = 0.0
            valid = self.seed_map >= 0
            self.final_gray_map[valid] = self.seed_map[valid]
        self._refresh_all_previews()

    def _on_eraser_toggled(self, checked: bool):
        """橡皮擦按钮切换：同步到画布的 _eraser_mode 标志。"""
        self.canvas._eraser_mode = checked
        self.btn_eraser.setText("✔  橡皮擦（激活）" if checked else "☐  橡皮擦")
        self.canvas.update()

    def _on_stroke_finished(self, pts: list, value: float):
        """
        笔触结束回调：调用 rasterize_stroke_to_seed 将路径写入 seed_map。
        笔触内部按采样顺序生成 0~1 时间进度（time_start=0, time_end=value）。
        """
        if self.seed_map is None:
            return

        # 若使用缩放代理预览，需将画布坐标映射回原始分辨率
        f = self._preview_factor
        if f < 1.0:
            inv = 1.0 / f
            pts = [(int(x * inv), int(y * inv)) for (x, y) in pts]

        # 橡皮擦模式：将笔触覆盖的区域重置为 -1（未赋值）
        if value == -2.0:
            if self.seed_map is None:
                return
            self._push_seed_history()
            hardness = self.slider_hardness.value() / 100.0
            r = self.canvas.brush_radius
            if f < 1.0:
                r = max(1, int(r / f))
            # 复用 rasterize_stroke_to_seed 获取覆盖像素，然后将其重置为 -1
            # 创建一个临时 seed_map 来识别被擦除的像素
            import numpy as np
            tmp = np.full_like(self.seed_map, -1.0)
            rasterize_stroke_to_seed(
                tmp, pts,
                brush_radius=r,
                brush_hardness=hardness,
                time_start=0.0,
                time_end=1.0,
            )
            erased = tmp >= 0
            self.seed_map[erased] = -1.0
            if self.gray_map is not None:
                self.gray_map[erased] = 0.0
            # 同步 base_gray_map / final_gray_map，确保生长预览使用最新数据
            if self.base_gray_map is not None:
                self.base_gray_map[erased] = 0.0
            if self.final_gray_map is not None:
                self.final_gray_map[erased] = 0.0
            self._refresh_all_previews()
            return

        # 写入前先保存快照，支持 Ctrl+Z 撤销
        self._push_seed_history()

        hardness = self.slider_hardness.value() / 100.0
        r = self.canvas.brush_radius
        if f < 1.0:
            r = max(1, int(r / f))

        # 笔触内部时间范围：从 0 到用户设定的 value
        # 这样单笔内部有渐变，多笔可以覆盖不同区间
        rasterize_stroke_to_seed(
            self.seed_map,
            pts,
            brush_radius=r,
            brush_hardness=hardness,
            time_start=0.0,
            time_end=value,
        )

        # 将 seed_map（有效区域）同步到 gray_map（仅用于即时预览）
        valid = self.seed_map >= 0
        if self.gray_map is not None:
            self.gray_map[valid] = self.seed_map[valid]
        # 同步 base_gray_map / final_gray_map，确保生长预览使用最新数据
        if self.base_gray_map is not None:
            self.base_gray_map[valid] = self.seed_map[valid]
        if self.final_gray_map is not None:
            self.final_gray_map[valid] = self.seed_map[valid]

        self._refresh_all_previews()

    def _clear_seed(self):
        # 清空前先保存快照，支持 Ctrl+Z 撤销
        self._push_seed_history()
        if self.seed_map is not None:
            self.seed_map[:] = -1.0
        if self.gray_map is not None:
            self.gray_map[:] = 0.0
        # 同步清除 base_gray_map / final_gray_map
        if self.base_gray_map is not None:
            self.base_gray_map[:] = 0.0
        if self.final_gray_map is not None:
            self.final_gray_map[:] = 0.0
        self.lbl_gray.set_pixmap(None)
        self.lbl_gray_single.set_pixmap(None)
        self.lbl_prop_status.setText("路径已清空")
        self._refresh_canvas_overlay()

    # ── 单图路径传播 ──────────────────────────────────────────────────
    def _generate_from_seed(self):
        """根据 seed_map + mask_map，用距离加权传播生成完整 gray_map。"""
        if self.source_image is None:
            QMessageBox.information(self, "提示", "请先导入单图。")
            return
        if self.seed_map is None:
            QMessageBox.information(self, "提示", "请先绘制路径。")
            return

        # 检查是否有有效 seed
        valid_count = int((self.seed_map >= 0).sum())
        if valid_count == 0:
            QMessageBox.information(self, "提示", "seed_map 中没有有效路径点，请先手绘路径。")
            return

        if self.mask_map is None:
            reply = QMessageBox.question(
                self, "无 Mask",
                "当前没有 Mask，将对整张图传播。\n建议先生成 Mask 以限制传播区域。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        radius = self.spin_prop_radius.value()
        power = self.dspin_power.value()
        fallback = self.chk_fallback.isChecked()
        smooth_iter = self.spin_smooth.value()

        self.lbl_prop_status.setText("传播计算中，请稍候…")
        QApplication.processEvents()

        try:
            gray = propagate_seed_to_gray(
                seed_map=self.seed_map,
                mask_map=self.mask_map,
                radius=radius,
                power=power,
                fallback_nearest=fallback,
            )

            if smooth_iter > 0:
                gray = smooth_gray_map(
                    gray,
                    self.mask_map,
                    iterations=smooth_iter,
                )

        except Exception as ex:
            QMessageBox.warning(self, "生成失败", str(ex))
            self.lbl_prop_status.setText("生成失败")
            return

        self.gray_map = gray
        self.base_gray_map = gray.copy()  # 保留原始基础灰度图
        # 重置噪波叠加结果（基础图已更新）
        self.final_gray_map = None
        self.noise_map = None

        # 更新预览
        gp = np_gray_to_qpixmap(self.gray_map)
        self.lbl_gray.set_pixmap(gp)
        self.lbl_gray_single.set_pixmap(gp)
        # 重置噪波叠加结果预览
        self.lbl_noise_result.set_pixmap(None)
        self.lbl_noise_result_single.set_pixmap(None)
        if self.mask_map is not None:
            bw_px = np_mask_to_qpixmap(self.mask_map)
            self.lbl_mask.set_pixmap(bw_px)
            self.lbl_mask_single.set_pixmap(bw_px)
        self._refresh_canvas_overlay()

        self.lbl_prop_status.setText(
            f"完成！seed={valid_count}px · 半径={radius} · 衰减={power:.1f}"
            f" · 平滑={smooth_iter}次"
        )

    # ── 笔刷控件同步 ──────────────────────────────────────────────────
    def _on_hardness_slider(self, v: int):
        self.lbl_hardness.setText(f"硬度：{v}%")

    def _on_brush_slider(self, v: int):
        self.edit_brush.setText(str(v))
        self.canvas.brush_radius = v

    def _on_brush_edit(self):
        try:
            v = int(self.edit_brush.text())
            v = max(1, min(200, v))
        except ValueError:
            v = self.slider_brush.value()
        self.slider_brush.setValue(v)
        self.canvas.brush_radius = v
        self.edit_brush.setText(str(v))

    def _on_value_slider(self, v: int):
        fv = v / 100.0
        self.edit_value.setText(f"{fv:.2f}")
        self.canvas.draw_value = fv

    def _on_value_edit(self):
        try:
            fv = float(self.edit_value.text())
            fv = max(0.0, min(1.0, fv))
        except ValueError:
            fv = self.slider_value.value() / 100.0
        self.slider_value.setValue(int(fv * 100))
        self.canvas.draw_value = fv
        self.edit_value.setText(f"{fv:.2f}")

    # ── 预览刷新 ──────────────────────────────────────────────────────
    def _get_checker_cached(self, w: int, h: int) -> QPixmap:
        """获取棋盘格 QPixmap（带缓存，避免重复生成）。"""
        if self._checker_cache is not None and self._checker_cache_size == (w, h):
            return self._checker_cache
        self._checker_cache = checkerboard_qpixmap(w, h, 16)
        self._checker_cache_size = (w, h)
        return self._checker_cache

    def _refresh_canvas_overlay(self):
        idx = self.combo_overlay.currentIndex()
        f = self._preview_factor  # 预览缩放因子

        # ── 生长预览模式（index 5）──
        if idx == 5:
            # 优先使用 final_gray_map（已叠加噪波），否则用 base_gray_map / gray_map
            gray_for_preview = (
                self.final_gray_map if self.final_gray_map is not None
                else (self.base_gray_map if self.base_gray_map is not None else self.gray_map)
            )
            if self.source_image is not None and gray_for_preview is not None:
                # 读取当前激活的预览控件参数
                is_seq = self.btn_mode_seq.isChecked()
                if is_seq and hasattr(self, "_seq_preview_slider"):
                    progress = self._seq_preview_slider.value() / 1000.0
                elif hasattr(self, "_single_preview_slider"):
                    progress = self._single_preview_slider.value() / 1000.0
                else:
                    progress = 0.5
                invert = self.chk_invert.isChecked() if hasattr(self, "chk_invert") else False

                # 使用缩放代理计算预览（大图加速）
                src_img = _downscale_pil(self.source_image, f) if f < 1.0 else self.source_image
                gray_ds = _downscale_array(gray_for_preview, f) if f < 1.0 else gray_for_preview
                mask_ds = _downscale_array(self.mask_map, f) if (f < 1.0 and self.mask_map is not None) else self.mask_map

                preview_px = compute_growth_preview_pixmap(
                    source_image=src_img,
                    gray_map=gray_ds,
                    mask_map=mask_ds,
                    progress=progress,
                    invert=invert,
                )
                ph, pw = src_img.height, src_img.width
                checker_px = self._get_checker_cached(pw, ph)
                self.canvas.set_source(checker_px)
                self.canvas.set_overlay(preview_px, opaque=True)
            else:
                self.canvas.set_overlay(None)
                if self.source_image is not None:
                    self.canvas.set_source(pil_to_qpixmap(
                        _downscale_pil(self.source_image, f) if f < 1.0 else self.source_image
                    ))
            return

        # 非生长预览模式：确保底图是原图（缩放代理）
        if self.source_image is not None:
            self.canvas.set_source(pil_to_qpixmap(
                _downscale_pil(self.source_image, f) if f < 1.0 else self.source_image
            ))

        if idx == 0 or self.source_image is None:
            # 原图
            self.canvas.set_overlay(None)
        elif idx == 1 and self.mask_map is not None:
            # 原图 + 主体范围（青色蒙层）
            self.canvas.set_overlay(np_mask_overlay_qpixmap(
                _downscale_array(self.mask_map, f) if f < 1.0 else self.mask_map
            ))
        elif idx == 2 and self.gray_map is not None:
            # 原图 + 灰度结果（半透明灰度叠加）
            self.canvas.set_overlay(np_gray_to_qpixmap(
                _downscale_array(self.gray_map, f) if f < 1.0 else self.gray_map
            ))
        elif idx == 3 and self.gray_map is not None:
            # 仅灰度结果：把底图换成灰度图
            self.canvas.set_source(np_gray_to_qpixmap(
                _downscale_array(self.gray_map, f) if f < 1.0 else self.gray_map
            ))
            self.canvas.set_overlay(None)
        elif idx == 4 and self.final_gray_map is not None:
            # 仅叠加噪波后结果
            self.canvas.set_source(np_gray_to_qpixmap(
                _downscale_array(self.final_gray_map, f) if f < 1.0 else self.final_gray_map
            ))
            self.canvas.set_overlay(None)
        else:
            self.canvas.set_overlay(None)

    def _refresh_all_previews(self):
        # 灰度图
        if self.gray_map is not None:
            gp = np_gray_to_qpixmap(self.gray_map)
            self.lbl_gray.set_pixmap(gp)
            self.lbl_gray_single.set_pixmap(gp)
        # mask
        if self.mask_map is not None:
            mp = np_mask_to_qpixmap(self.mask_map)
            self.lbl_mask.set_pixmap(mp)
            self.lbl_mask_single.set_pixmap(mp)
        # 噪波叠加结果（如果有）
        if self.final_gray_map is not None:
            fp = np_gray_to_qpixmap(self.final_gray_map)
            self.lbl_noise_result.set_pixmap(fp)
            self.lbl_noise_result_single.set_pixmap(fp)
        self._refresh_canvas_overlay()

    # ── 命名 ──────────────────────────────────────────────────────────
    def _on_name_changed(self, text: str):
        if text:
            self.name_preview.setText(f"预览：T_{text}_Growth.png")
        else:
            self.name_preview.setText("预览：-")

    def _apply_name(self):
        tag = self.name_input.text().strip()
        if not tag:
            QMessageBox.warning(self, "命名为空", "请输入导出命名 tag。")
            return
        self._output_basename = tag
        self.name_preview.setText(f"预览：T_{tag}_Growth.png")

    # ── 导出 ──────────────────────────────────────────────────────────
    def _get_export_size(self, orig_h: int, orig_w: int):
        """根据导出尺寸 combo 解析目标宽高，返回 (w, h)。
        如果选择"原始尺寸"或输入无效，返回 (orig_w, orig_h)。"""
        def _parse(combo_text: str, orig: int) -> int:
            t = combo_text.strip()
            if t == "原始尺寸" or t == "":
                return orig
            try:
                v = int(t)
                return max(1, v)
            except ValueError:
                return orig
        w = _parse(self.export_size_w.currentText(), orig_w)
        h = _parse(self.export_size_h.currentText(), orig_h)
        return w, h

    # ── 导出目录记忆 ─────────────────────────────────────────────────
    def _get_export_dir_cache_path(self) -> str:
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "growth_gray_last_export_dir.txt")

    def _load_last_export_dir(self) -> str:
        try:
            with open(self._get_export_dir_cache_path(), "r", encoding="utf-8") as f:
                d = f.read().strip()
                if d and os.path.isdir(d):
                    return d
        except Exception:
            pass
        if self._src_path and os.path.isfile(self._src_path):
            return os.path.dirname(self._src_path)
        elif self._src_path and os.path.isdir(self._src_path):
            return self._src_path
        return ""

    def _save_last_export_dir(self, path: str):
        try:
            with open(self._get_export_dir_cache_path(), "w", encoding="utf-8") as f:
                f.write(os.path.dirname(path))
        except Exception:
            pass

    def _export_gray(self):
        if self.gray_map is None or self.source_image is None:
            QMessageBox.information(self, "提示", "请先导入图像并绘制路径。")
            return

        # 确定默认文件名
        tag = self.name_input.text().strip()
        if tag:
            default_name = f"T_{tag}_Growth.png"
        elif self._output_basename:
            default_name = f"{self._output_basename}_Growth.png"
        else:
            default_name = "Growth.png"

        # 确定默认目录（优先使用上次导出目录）
        default_dir = self._load_last_export_dir()

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出灰度图",
            os.path.join(default_dir, default_name),
            "PNG 图像 (*.png)"
        )
        if not save_path:
            return

        try:
            u8 = (np.clip(self.gray_map, 0.0, 1.0) * 255).astype(np.uint8)
            out_img = Image.fromarray(u8, mode="L")
            # 按导出尺寸缩放
            orig_h, orig_w = self.gray_map.shape
            export_w, export_h = self._get_export_size(orig_h, orig_w)
            if export_w != orig_w or export_h != orig_h:
                out_img = out_img.resize((export_w, export_h), Image.BILINEAR)
            out_img.save(save_path)
            self._save_last_export_dir(save_path)
            size_info = f"{export_w}×{export_h}"
            self.lbl_export_info.setText(f"已导出：{size_info}\n{os.path.basename(save_path)}")
            QMessageBox.information(self, "导出成功", f"尺寸：{size_info}\n已保存到：\n{save_path}")
        except Exception as ex:
            QMessageBox.warning(self, "导出失败", str(ex))

    def _export_final_gray(self):
        """导出叠加噪波后的最终灰度图。"""
        if self.final_gray_map is None:
            QMessageBox.information(
                self, "提示",
                "尚未生成叠加噪波后的灰度图。\n请先启用噪波叠加并点击「应用噪波叠加」。"
            )
            return

        tag = self.name_input.text().strip()
        if tag:
            default_name = f"T_{tag}_Growth_Noise.png"
        elif self._output_basename:
            default_name = f"{self._output_basename}_Growth_Noise.png"
        else:
            default_name = "Growth_Noise.png"

        default_dir = self._load_last_export_dir()

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出叠加噪波后的灰度图",
            os.path.join(default_dir, default_name),
            "PNG 图像 (*.png)"
        )
        if not save_path:
            return

        try:
            u8 = (np.clip(self.final_gray_map, 0.0, 1.0) * 255).astype(np.uint8)
            out_img = Image.fromarray(u8, mode="L")
            # 按导出尺寸缩放
            orig_h, orig_w = self.final_gray_map.shape
            export_w, export_h = self._get_export_size(orig_h, orig_w)
            if export_w != orig_w or export_h != orig_h:
                out_img = out_img.resize((export_w, export_h), Image.BILINEAR)
            out_img.save(save_path)
            size_info = f"{export_w}×{export_h}"
            self._save_last_export_dir(save_path)
            self.lbl_export_info.setText(f"已导出（含噪波）：{size_info}\n{os.path.basename(save_path)}")
            QMessageBox.information(self, "导出成功", f"尺寸：{size_info}\n已保存到：\n{save_path}")
        except Exception as ex:
            QMessageBox.warning(self, "导出失败", str(ex))

    # ── 噪波叠加 ──────────────────────────────────────────────────────
    def _on_noise_enable_changed(self, state: int):
        """启用/禁用噪波叠加开关：折叠/展开噪波设置面板。"""
        enabled = bool(state)
        self._noise_content_widget.setVisible(enabled)
        if not enabled:
            self.lbl_noise_status.setText("未启用")
            # 禁用时清除 final_gray_map，恢复使用基础灰度图
            self.final_gray_map = None
            self.noise_map = None
            self.lbl_noise_result.set_pixmap(None)
            self.lbl_noise_result_single.set_pixmap(None)
            self._refresh_canvas_overlay()

    def _on_noise_rotate_changed(self, value: int):
        """旋转滑块值变化时更新标签。"""
        self.lbl_noise_rotate_val.setText(f"{value}°")

    def _on_noise_source_changed(self, idx: int):
        """噪波来源切换：显示/隐藏导入贴图控件。"""
        self._noise_import_widget.setVisible(idx == 1)

    def _import_noise_texture(self):
        """导入外部噪波贴图。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入噪波贴图", "",
            "图像文件 (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("L")  # 转灰度
            self._noise_image = img
            self.lbl_noise_import_info.setText(
                f"{os.path.basename(path)}\n尺寸：{img.width}×{img.height}"
            )
            # 导入成功后自动触发噪波叠加重新计算
            self._schedule_noise_regen()
        except Exception as ex:
            QMessageBox.warning(self, "导入失败", str(ex))

    def _parse_noise_scale(self, line_edit) -> float:
        """从 QLineEdit 中安全读取噪波缩放值，非法输入时返回默认值 4.0，并限制范围 0.1~32.0"""
        try:
            val = float(line_edit.text())
            return max(0.1, min(32.0, val))
        except (ValueError, TypeError):
            return 4.0

    def _generate_builtin_noise(self, h: int, w: int,
                                 scale_x: float, scale_y: float,
                                 offset_x: float, offset_y: float,
                                 rotate_deg: float = 0.0) -> np.ndarray:
        """
        生成内置 Perlin 风格噪波（使用 numpy 实现的简单分形噪波）。
        返回 H×W float32，范围 0~1。
        scale_x / scale_y 分别控制水平和垂直方向的缩放。
        rotate_deg 控制噪波采样坐标的旋转角度（0~360°）。
        """
        # 使用多层正弦/余弦叠加模拟 Perlin 风格噪波（不依赖额外库）
        # 生成坐标网格
        y_coords = (np.arange(h, dtype=np.float32) / h * scale_y + offset_y)
        x_coords = (np.arange(w, dtype=np.float32) / w * scale_x + offset_x)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # 旋转坐标
        if rotate_deg != 0.0:
            rad = np.radians(rotate_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            xx_r = xx * cos_a - yy * sin_a
            yy_r = xx * sin_a + yy * cos_a
            xx, yy = xx_r, yy_r

        # 多倍频叠加（4个倍频）
        noise = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        total_amplitude = 0.0
        freq = 1.0
        for _ in range(4):
            # 使用不同角度的正弦波叠加，模拟噪波效果
            n = (
                np.sin(xx * freq * 2.0 * np.pi + yy * freq * 1.3 * np.pi) * 0.5
                + np.sin(xx * freq * 1.7 * np.pi - yy * freq * 2.1 * np.pi) * 0.3
                + np.sin((xx + yy) * freq * 1.5 * np.pi) * 0.2
            )
            noise += n * amplitude
            total_amplitude += amplitude
            amplitude *= 0.5
            freq *= 2.0

        # 归一化到 0~1
        noise = noise / total_amplitude
        noise = (noise + 1.0) * 0.5  # -1~1 → 0~1
        return np.clip(noise, 0.0, 1.0)

    def _sample_noise_from_image(self, h: int, w: int,
                                  scale_x: float, scale_y: float,
                                  offset_x: float, offset_y: float,
                                  rotate_deg: float = 0.0) -> np.ndarray:
        """
        从导入的噪波贴图中采样，支持缩放、平铺和旋转。
        返回 H×W float32，范围 0~1。
        scale_x / scale_y 分别控制水平和垂直方向的缩放。
        rotate_deg 控制噪波采样坐标的旋转角度（0~360°）。
        """
        if self._noise_image is None:
            return self._generate_builtin_noise(h, w, scale_x, scale_y, offset_x, offset_y, rotate_deg)

        noise_arr = np.array(self._noise_image, dtype=np.float32) / 255.0  # H'×W'
        nh, nw = noise_arr.shape

        # 生成采样坐标（支持平铺 + 缩放 + 偏移）
        y_base = np.arange(h, dtype=np.float32) / h * scale_y + offset_y
        x_base = np.arange(w, dtype=np.float32) / w * scale_x + offset_x
        xx, yy = np.meshgrid(x_base, y_base)

        # 旋转坐标
        if rotate_deg != 0.0:
            rad = np.radians(rotate_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            # 围绕中心旋转
            cx = xx.mean()
            cy = yy.mean()
            xx_r = (xx - cx) * cos_a - (yy - cy) * sin_a + cx
            yy_r = (xx - cx) * sin_a + (yy - cy) * cos_a + cy
            xx, yy = xx_r, yy_r

        # 映射到噪波图像素坐标（取模实现平铺）
        xi = ((xx % 1.0) * nw).astype(np.int32) % nw
        yi = ((yy % 1.0) * nh).astype(np.int32) % nh
        # 最近邻采样
        sampled = noise_arr[yi, xi]
        return sampled.astype(np.float32)

    def _apply_noise_overlay(self):
        """
        应用噪波叠加：
        final_gray_map = clip(base_gray_map + n * noise_strength * mask_map, 0, 1)
        其中 n = noise_map * 2 - 1（映射到 -1~1）
        """
        if not self.chk_noise_enable.isChecked():
            QMessageBox.information(self, "提示", "请先勾选「启用噪波叠加」。")
            return

        if self.base_gray_map is None:
            QMessageBox.information(self, "提示", "请先生成基础灰度图。")
            return

        h, w = self.base_gray_map.shape
        scale_x = self._parse_noise_scale(self.dspin_noise_scale_x)
        scale_y = self._parse_noise_scale(self.dspin_noise_scale_y)
        offset_x = self.dspin_noise_offset_x.value()
        offset_y = self.dspin_noise_offset_y.value()
        strength = self.dspin_noise_strength.value()
        invert = self.chk_noise_invert.isChecked()
        rotate_deg = float(self.slider_noise_rotate.value())
        use_imported = (self.combo_noise_source.currentIndex() == 1)

        self.lbl_noise_status.setText("计算中…")
        QApplication.processEvents()

        try:
            # 生成噪波图
            if use_imported and self._noise_image is not None:
                noise = self._sample_noise_from_image(h, w, scale_x, scale_y, offset_x, offset_y, rotate_deg)
            else:
                noise = self._generate_builtin_noise(h, w, scale_x, scale_y, offset_x, offset_y, rotate_deg)

            if invert:
                noise = 1.0 - noise

            self.noise_map = noise

            # 映射到 -1~1
            n = noise * 2.0 - 1.0

            # mask 限制：只在 mask_map 内生效
            if self.mask_map is not None:
                mask = np.clip(self.mask_map, 0.0, 1.0)
            else:
                mask = np.ones((h, w), dtype=np.float32)

            # 叠加公式
            final = np.clip(self.base_gray_map + n * strength * mask, 0.0, 1.0)
            self.final_gray_map = final.astype(np.float32)

            # 更新预览
            fp = np_gray_to_qpixmap(self.final_gray_map)
            self.lbl_noise_result.set_pixmap(fp)
            self.lbl_noise_result_single.set_pixmap(fp)

            self.lbl_noise_status.setText(
                f"完成！强度={strength:.2f} · 缩放=({scale_x:.1f}, {scale_y:.1f})"
                f" · 偏移=({offset_x:.1f}, {offset_y:.1f})"
                f" · 旋转={rotate_deg:.0f}°"
                f"{'  [反相]' if invert else ''}"
            )
            self._refresh_canvas_overlay()

        except Exception as ex:
            QMessageBox.warning(self, "噪波叠加失败", str(ex))
            self.lbl_noise_status.setText("失败")

    def _schedule_noise_regen(self):
        """参数变化时触发防抖定时器，400ms 后自动重新计算噪波叠加。"""
        if self.chk_noise_enable.isChecked() and self.base_gray_map is not None:
            self._noise_timer.start(400)

    def _apply_noise_overlay_silent(self):
        """静默版噪波叠加（不弹对话框，用于实时预览）。"""
        if not self.chk_noise_enable.isChecked() or self.base_gray_map is None:
            return

        h, w = self.base_gray_map.shape
        scale_x = self._parse_noise_scale(self.dspin_noise_scale_x)
        scale_y = self._parse_noise_scale(self.dspin_noise_scale_y)
        offset_x = self.dspin_noise_offset_x.value()
        offset_y = self.dspin_noise_offset_y.value()
        strength = self.dspin_noise_strength.value()
        invert = self.chk_noise_invert.isChecked()
        rotate_deg = float(self.slider_noise_rotate.value())
        use_imported = (self.combo_noise_source.currentIndex() == 1)

        try:
            if use_imported and self._noise_image is not None:
                noise = self._sample_noise_from_image(h, w, scale_x, scale_y, offset_x, offset_y, rotate_deg)
            else:
                noise = self._generate_builtin_noise(h, w, scale_x, scale_y, offset_x, offset_y, rotate_deg)

            if invert:
                noise = 1.0 - noise

            self.noise_map = noise
            n = noise * 2.0 - 1.0

            if self.mask_map is not None:
                mask = np.clip(self.mask_map, 0.0, 1.0)
            else:
                mask = np.ones((h, w), dtype=np.float32)

            final = np.clip(self.base_gray_map + n * strength * mask, 0.0, 1.0)
            self.final_gray_map = final.astype(np.float32)

            fp = np_gray_to_qpixmap(self.final_gray_map)
            self.lbl_noise_result.set_pixmap(fp)
            self.lbl_noise_result_single.set_pixmap(fp)

            self.lbl_noise_status.setText(
                f"实时预览  强度={strength:.2f} · 缩放=({scale_x:.1f}, {scale_y:.1f})"
                f" · 偏移=({offset_x:.1f}, {offset_y:.1f})"
                f" · 旋转={rotate_deg:.0f}°"
                f"{'  [反相]' if invert else ''}"
            )
            self._refresh_canvas_overlay()

        except Exception:
            pass  # 实时预览静默失败
