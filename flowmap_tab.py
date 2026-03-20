# -*- coding: utf-8 -*-
"""
法线贴图 / Flow Control Map 编辑器
布局：
  左列  = 参考图显示区（上） + 法线结果观察区（下）
  中列  = 工具栏 + 主绘制区（UV offset 流动预览 + 绘制）
  右列  = 导出设置

核心数据逻辑：
  - 内部维护一张真正的 normal map（HxWx3 float32，xyz 法线向量）
  - 默认基准值：flat normal (0, 0, 1) → packed RGB(128, 128, 255)
  - 绘制链路：笔触方向 → 生成目标法线向量 → 与当前法线向量 lerp 混合 → normalize → 写回
  - 擦除 = 恢复到 flat normal (0, 0, 1)，不是透明清空
  - 结果区 = 直接显示当前 normal map（蓝紫底色）
  - 导出 = 直接导出 normal buffer，空白时就是标准 flat normal 蓝图
  - 主绘制区流动预览 = 用 normal map 的 xy 分量作为 UV offset 驱动参考图采样
"""

import os, math
import numpy as np
from typing import Optional, Tuple

from PIL import Image

from PySide6.QtCore import (
    Qt, QPoint, QRect, QTimer, QRegularExpression,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QCursor,
    QRegularExpressionValidator, QAction, QColor,
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QSlider, QComboBox,
    QLineEdit, QGroupBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFileDialog, QMessageBox,
    QSpinBox, QSizePolicy, QFrame, QMenu, QCheckBox,
)

# ── 常量 ──────────────────────────────────────────────────────────────
CANVAS_W = 512
CANVAS_H = 512
# flat normal 默认基准值：向量 (0,0,1) → packed RGB(128,128,255)
FLAT_NORMAL_VEC  = (0.0, 0.0, 1.0)   # float xyz
FLAT_NORMAL_RGB  = (128, 128, 255)    # packed uint8


# ── 工具函数 ──────────────────────────────────────────────────────────
def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qi = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi)


def np_rgba_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """将 HxWx4 uint8 numpy 数组转为 QPixmap（避免 PIL 中转）。"""
    h, w = arr.shape[:2]
    arr_c = np.ascontiguousarray(arr, dtype=np.uint8)
    qi = QImage(arr_c.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


def make_falloff_mask(size: int, hardness: float) -> np.ndarray:
    """
    生成 size×size 的 float32 falloff mask，值域 [0,1]。
    hardness=1 → 硬边，hardness=0 → 完全羽化。
    """
    s = max(4, size)
    cx = cy = (s - 1) / 2.0
    r = s / 2.0
    hard_r = r * max(0.0, min(1.0, hardness))
    ys, xs = np.mgrid[0:s, 0:s]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
    mask = np.where(
        dist >= r, 0.0,
        np.where(dist <= hard_r, 1.0,
                 1.0 - ((dist - hard_r) / max(r - hard_r, 1e-6)) ** 2)
    ).astype(np.float32)
    return mask


def _bilinear_sample_wrap(img: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """
    双线性插值 + Wrap（Repeat）采样，对齐 UE Sample(Texture) 默认行为。
    img : HxWxC uint8
    fx  : HxW float，列坐标（像素空间，可超出范围，自动 wrap）
    fy  : HxW float，行坐标（像素空间，可超出范围，自动 wrap）
    返回 HxWxC float32
    """
    h, w, c = img.shape
    img_f = img.astype(np.float32)

    # wrap 到 [0, w) / [0, h)
    fx_w = fx % w
    fy_w = fy % h

    x0 = np.floor(fx_w).astype(np.int32)
    y0 = np.floor(fy_w).astype(np.int32)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h

    # 双线性权重
    tx = (fx_w - x0).astype(np.float32)[:, :, np.newaxis]  # HxWx1
    ty = (fy_w - y0).astype(np.float32)[:, :, np.newaxis]

    # 四邻域采样
    c00 = img_f[y0, x0]   # HxWxC
    c10 = img_f[y0, x1]
    c01 = img_f[y1, x0]
    c11 = img_f[y1, x1]

    return (c00 * (1 - tx) * (1 - ty)
            + c10 * tx       * (1 - ty)
            + c01 * (1 - tx) * ty
            + c11 * tx       * ty)


# ── 左侧小预览区（paintEvent 绘制，无 resize 循环）────────────────────
class SidePreviewLabel(QWidget):
    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._src_img: Optional[Image.Image] = None
        self._cached_pix: Optional[QPixmap] = None
        self._cached_size = (0, 0)
        self.setStyleSheet(
            "background:#0d0d1a; border:1px solid #313244; border-radius:8px;"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_image(self, img: Optional[Image.Image]):
        self._src_img = img
        self._cached_pix = None
        self.update()

    def _get_scaled_pix(self) -> Optional[QPixmap]:
        if self._src_img is None:
            return None
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None
        if self._cached_pix is not None and self._cached_size == (w, h):
            return self._cached_pix
        pix = pil_to_qpixmap(self._src_img)
        self._cached_pix = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cached_size = (w, h)
        return self._cached_pix

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._cached_pix = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 26))
        pix = self._get_scaled_pix()
        if pix is not None:
            x = (self.width()  - pix.width())  // 2
            y = (self.height() - pix.height()) // 2
            p.drawPixmap(x, y, pix)
        else:
            p.setPen(QColor(69, 71, 90))
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
        p.end()


# ── 左上角参考图导入区（支持拖拽 + 点击导入）────────────────────────
class DropRefWidget(QWidget):
    """
    左上角参考图区域：
    - 支持拖拽图片文件进来
    - 支持点击弹出文件对话框
    - 导入后在此区域显示缩略图
    - 通过 on_image_loaded 回调通知外部
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src_img: Optional[Image.Image] = None
        self._cached_pix: Optional[QPixmap] = None
        self._cached_size = (0, 0)
        self.on_image_loaded: Optional[callable] = None  # 回调(img: Image.Image | None)
        self.setAcceptDrops(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet(
            "background:#0d0d1a; border:2px dashed #585b70; border-radius:8px;"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(80)

    # ── 拖拽 ──────────────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(
                "background:#1a1a2e; border:2px dashed #89b4fa; border-radius:8px;"
            )

    def dragLeaveEvent(self, e):
        self._reset_style()

    def dropEvent(self, e):
        self._reset_style()
        urls = e.mimeData().urls()
        if urls:
            self._load(urls[0].toLocalFile())

    # ── 点击导入 ──────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择参考图", "",
                "Images (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
            )
            if path:
                self._load(path)

    def _load(self, path: str):
        try:
            img = Image.open(path).convert("RGBA")
            self._src_img = img
            self._cached_pix = None
            self.update()
            if self.on_image_loaded:
                self.on_image_loaded(img)
        except Exception as ex:
            QMessageBox.critical(self, "错误", f"加载参考图失败：\n{ex}")

    def clear_image(self):
        self._src_img = None
        self._cached_pix = None
        self.update()
        if self.on_image_loaded:
            self.on_image_loaded(None)

    def _reset_style(self):
        if self._src_img is None:
            self.setStyleSheet(
                "background:#0d0d1a; border:2px dashed #585b70; border-radius:8px;"
            )
        else:
            self.setStyleSheet(
                "background:#0d0d1a; border:1px solid #313244; border-radius:8px;"
            )

    # ── 绘制 ──────────────────────────────────────────────────────────
    def _get_scaled_pix(self) -> Optional[QPixmap]:
        if self._src_img is None:
            return None
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None
        if self._cached_pix is not None and self._cached_size == (w, h):
            return self._cached_pix
        pix = pil_to_qpixmap(self._src_img)
        self._cached_pix = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cached_size = (w, h)
        return self._cached_pix

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._cached_pix = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 26))
        pix = self._get_scaled_pix()
        if pix is not None:
            x = (self.width()  - pix.width())  // 2
            y = (self.height() - pix.height()) // 2
            p.drawPixmap(x, y, pix)
            # 右上角显示清除按钮提示
            p.setPen(QColor(100, 100, 120))
            p.setFont(p.font())
            p.drawText(self.rect().adjusted(0, 4, -6, 0),
                       Qt.AlignTop | Qt.AlignRight, "✕")
        else:
            p.setPen(QColor(89, 91, 112))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "📂 点击或拖拽\n导入参考图")
        p.end()

    def contextMenuEvent(self, e):
        if self._src_img is not None:
            menu = QMenu(self)
            act_clear = QAction("清除参考图", self)
            act_clear.triggered.connect(self.clear_image)
            menu.addAction(act_clear)
            menu.exec(e.globalPos())


# ── 主绘制画布（Normal Map 编辑器 + UV offset 流动预览）──────────────────
class VectorMapCanvas(QWidget):
    """
    核心画布：
    - 数据层：normal_map，HxWx3 float32，存储 xyz 法线向量，默认 flat normal (0,0,1)
    - 绘制链路：笔触方向 → 生成目标法线向量 → lerp 混合 → normalize → 写回
    - 擦除：恢复到 flat normal (0,0,1)，不是透明清空
    - 显示层：参考图经 normal map xy 分量 UV offset 驱动后的动态采样结果
    - Follow Stroke Direction：写入方向跟随笔触切线
    - Invert Direction：写入方向反向
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#0d0d1a; border-radius:10px;")
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        # ── 画布逻辑尺寸
        self._cw = CANVAS_W
        self._ch = CANVAS_H

        # ── 缩放状态（鼠标滚轮缩放）
        self._zoom        = 1.0          # 当前缩放倍数
        self._zoom_min    = 0.25
        self._zoom_max    = 8.0
        self._pan_offset  = QPoint(0, 0) # 平移偏移（暂不开放拖动，仅缩放居中）

        # ── 鼠标位置（用于笔刷预览圆）
        self._mouse_widget_pos: Optional[QPoint] = None

        # ── Normal map 数据：HxWx3 float32，xyz 法线向量，默认 flat normal (0,0,1)
        self.normal_map: np.ndarray = self._make_flat_normal()

        # ── 参考图（RGBA numpy HxWx4 uint8）
        self._ref_np: Optional[np.ndarray] = None

        # ── 笔刷参数
        self.brush_size       = 80   # UI显示40，实际大小 = UI × 2
        self.brush_hardness   = 0.5 * 0.5   # 实际硬度，初始0.25
        self.brush_strength   = 0.45 * 0.5  # 实际强度，初始0.225（UI显示45%）
        self.brush_opacity    = 0.05         # 笔刷透明度，默认5%
        self.brush_spacing    = 5.0          # 笔刷连续性（spacing %），默认5%
        self.mode             = "draw"   # "draw" | "erase"
        self.follow_stroke    = True     # Follow Stroke Direction

        self.mode_dx          = True     # True=DirectX（G通道翻转），False=OpenGL

        # ── 流动动画（双相位交叉混合，无缝循环）
        self.flow_speed        = 0.211563   # 对应 UE 材质实例 Speed 参数（每秒相位推进量）
        self.preview_strength = 0.538304   # 对应 UE 材质实例 strangth 参数（UV 偏移强度）
        self._flow_time   = 0.0          # 秒为单位的累计时间
        self._fps         = 30
        self._flow_timer  = QTimer(self)
        self._flow_timer.timeout.connect(self._tick_flow)
        # Timer 由 FlowMapTab.showEvent/hideEvent 控制启停，避免 Tab 不可见时持续消耗 CPU

        # ── 鼠标状态
        self._drawing   = False
        self._last_cpos: Optional[Tuple[float, float]] = None

        # ── 右键拖动状态
        self._panning        = False
        self._pan_start_pos: Optional[QPoint] = None   # 右键按下时的 widget 坐标
        self._pan_start_off: Optional[QPoint] = None   # 右键按下时的 _pan_offset

        # ── Undo 历史栈（每次落笔前保存快照）
        self._undo_stack: list = []          # list of np.ndarray (HxWx3 float32)
        self._undo_max   = 30                # 最多保留 30 步

        # ── 显示缓存
        self._pix_rect: Optional[QRect] = None

        # ── 流动预览性能缓存
        # flow_xy 只在 normal_map 改变时重算（_flow_cache_dirty=True 触发）
        self._flow_cache_dirty: bool = True
        self._flow_x_cache: Optional[np.ndarray] = None  # calc_h x calc_w float32，UV 空间 X 偏移
        self._flow_y_cache: Optional[np.ndarray] = None  # calc_h x calc_w float32，UV 空间 Y 偏移
        self._flow_calc_size = (0, 0)  # 当前流动计算分辨率 (calc_h, calc_w)
        # 下采样后的参考图缓存（与 _flow_calc_size 对应）
        self._ref_calc_np: Optional[np.ndarray] = None
        # xs/ys 网格缓存（只在计算分辨率变化时重建）
        self._grid_xs: Optional[np.ndarray] = None
        self._grid_ys: Optional[np.ndarray] = None
        self._grid_res = (0, 0)
        # normal_vis 缓存（normal_map 不变时直接复用）
        self._normal_vis_pix: Optional[QPixmap] = None
        self._normal_vis_dirty: bool = True
        # 原图预览模式：True=用原图分辨率计算，False=用256分辨率（流畅）
        self._hq_preview: bool = False

        # ── 外部回调
        self.on_ref_updated:    Optional[callable] = None
        self.on_normal_updated: Optional[callable] = None  # 通知左侧结果区

        self._update_pix_rect()

    def _make_flat_normal(self) -> np.ndarray:
        """Create a flat normal map: all pixels = (0, 0, 1)."""
        nm = np.zeros((self._ch, self._cw, 3), dtype=np.float32)
        nm[:, :, 2] = 1.0  # z = 1
        return nm
    def set_ref(self, img: Optional[Image.Image]):
        """由外部（DropRefWidget 回调）设置参考图。"""
        if img is None:
            self._ref_np = None
        else:
            resized = img.convert("RGBA").resize(
                (self._cw, self._ch), Image.LANCZOS
            )
            self._ref_np = np.array(resized, dtype=np.uint8)
        self._flow_cache_dirty = True
        self._ref_calc_np = None  # 参考图换了，清除下采样缓存
        self.update()

    # ── 尺寸 / 缩放 ──────────────────────────────────────────────
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_pix_rect()

    def _update_pix_rect(self):
        """根据当前 zoom 和 _pan_offset 计算画布在 widget 中的显示矩形。"""
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        # 基础缩放：让画布尽量充满 widget
        base_scale = min(lw / self._cw, lh / self._ch)
        scale = base_scale * self._zoom
        pw = int(self._cw * scale)
        ph = int(self._ch * scale)
        # 居中 + 平移偏移
        x = (lw - pw) // 2 + self._pan_offset.x()
        y = (lh - ph) // 2 + self._pan_offset.y()
        self._pix_rect = QRect(x, y, pw, ph)

    def wheelEvent(self, e):
        """Ctrl+滚轮或直接滚轮缩放画布，笔刷大小不变。"""
        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(self._zoom_min, min(self._zoom_max, self._zoom * factor))
        self._zoom = new_zoom
        self._update_pix_rect()
        self.update()
        e.accept()

    def _widget_to_canvas(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        r = self._pix_rect
        if r is None or r.width() <= 0 or r.height() <= 0:
            return None
        rx = (pos.x() - r.x()) / r.width()  * self._cw
        ry = (pos.y() - r.y()) / r.height() * self._ch
        return (rx, ry)
    # ── 流动动画 tick ──────────────────────────────────────────────
    def _tick_flow(self):
        """
        每帧推进真实时间。
        flow_speed 单位：每秒推进的画布坐标单位数（相对于画布宽度的比例）。
        例：flow_speed=0.3 → 每秒推进 0.3 幅图像宽度的距离。
        """
        dt = 1.0 / self._fps
        self._flow_time += dt
        self.update()
    # ── paintEvent：UV offset 流动预览 + 笔刷预览圆 ──────────────────
    def paintEvent(self, e):
        self._update_pix_rect()
        r = self._pix_rect
        if r is None:
            return

        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 26))

        if self._ref_np is None:
            # 无参考图：显示 normal map 可视化（蓝紫底色）
            self._paint_normal_vis(p, r)
        else:
            # 有参考图：整张图都参与双相位 UV 偏移采样，直接输出最终结果
            # 不保留静止底图，无笔迹区域偏移量为 0，采样结果即原图本身
            self._paint_uv_flow_overlay(p, r)

        # 笔刷预览圆（始终显示，不受缩放影响）
        self._paint_brush_cursor(p)

        p.end()

    def _paint_brush_cursor(self, p: QPainter):
        """在鼠标位置画笔刷大小预览圆。"""
        if self._mouse_widget_pos is None:
            return
        r = self._pix_rect
        if r is None or r.width() <= 0:
            return

        # 笔刷大小在画布坐标系中的半径，转换到 widget 坐标系
        canvas_to_widget_scale = r.width() / self._cw
        radius_w = (self.brush_size / 2.0) * canvas_to_widget_scale

        mx = self._mouse_widget_pos.x()
        my = self._mouse_widget_pos.y()

        from PySide6.QtGui import QPen
        from PySide6.QtCore import QRectF
        # 外圈：白色半透明
        pen = QPen(QColor(255, 255, 255, 180))
        pen.setWidth(1)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(mx - radius_w, my - radius_w,
                             radius_w * 2, radius_w * 2))
        # 内圈：黑色半透明（增强对比度）
        pen2 = QPen(QColor(0, 0, 0, 120))
        pen2.setWidth(1)
        p.setPen(pen2)
        p.drawEllipse(QRectF(mx - radius_w + 1, my - radius_w + 1,
                             radius_w * 2 - 2, radius_w * 2 - 2))
    def _paint_normal_vis(self, p: QPainter, r: QRect):
        """将 normal map 可视化为颜色图（蓝紫底色，即 flat normal 外观）。
        G 通道翻转对齐导出时的 DirectX 模式（mode_dx=True），
        确保法线结果显示区和实际导出图颜色一致。
        缓存优化：normal_map 不变时直接复用上次的 QPixmap。
        """
        if self._normal_vis_dirty or self._normal_vis_pix is None:
            nm = self.normal_map  # HxWx3
            h, w = nm.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = np.clip((nm[:, :, 0] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            # G 通道根据当前导出模式决定是否翻转
            if self.mode_dx:
                rgba[:, :, 1] = np.clip((-nm[:, :, 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            else:
                rgba[:, :, 1] = np.clip((nm[:, :, 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            rgba[:, :, 2] = np.clip((nm[:, :, 2] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            rgba[:, :, 3] = 255
            self._normal_vis_pix = np_rgba_to_qpixmap(rgba)
            self._normal_vis_dirty = False
        scaled = self._normal_vis_pix.scaled(r.width(), r.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        p.drawPixmap(r.topLeft(), scaled)

    def _rebuild_flow_cache(self):
        """重建 flow_xy 缓存和 xs/ys 网格。只在 normal_map 改变时调用。
        计算分辨率动态取 min(原图分辨率, 512)，兼顾性能与效果。
        """
        nm = self.normal_map  # HxWx3 float32，原图分辨率
        src_h, src_w = nm.shape[:2]

        # 动态计算分辨率：原图预览开启时用原图分辨率，否则限制到 256
        max_res = src_w if getattr(self, '_hq_preview', False) else 256
        calc_w = min(src_w, max_res)
        calc_h = min(src_h, max_res)

        # 如果需要下采样，用 float32 精度做缩放，避免 uint8 量化损失
        if calc_w != src_w or calc_h != src_h:
            # 纯 numpy 双线性插值下采样（保持 float32 精度）
            gy = np.linspace(0, src_h - 1, calc_h, dtype=np.float32)
            gx = np.linspace(0, src_w - 1, calc_w, dtype=np.float32)
            gy_grid, gx_grid = np.meshgrid(gy, gx, indexing='ij')
            y0 = np.floor(gy_grid).astype(np.int32)
            x0 = np.floor(gx_grid).astype(np.int32)
            y1 = np.minimum(y0 + 1, src_h - 1)
            x1 = np.minimum(x0 + 1, src_w - 1)
            wy = (gy_grid - y0).astype(np.float32)[:, :, np.newaxis]
            wx = (gx_grid - x0).astype(np.float32)[:, :, np.newaxis]
            nm_small = (nm[y0, x0] * (1 - wy) * (1 - wx)
                      + nm[y0, x1] * (1 - wy) * wx
                      + nm[y1, x0] * wy * (1 - wx)
                      + nm[y1, x1] * wy * wx)
        else:
            nm_small = nm

        strength = self.preview_strength
        # X 方向：_apply_brush 中已取反写入，直接使用
        self._flow_x_cache = nm_small[:, :, 0] * strength
        # Y 方向：_apply_brush 中未取反（由 DirectX 导出 G 通道翻转保证），
        # 预览需要模拟 DirectX 翻转效果，所以这里取反
        self._flow_y_cache = -nm_small[:, :, 1] * strength
        self._flow_cache_dirty = False
        self._flow_calc_size = (calc_h, calc_w)  # 记录当前计算分辨率

        # 重建 xs/ys 网格（只在计算分辨率变化时重建）
        if self._grid_res != (calc_h, calc_w):
            self._grid_ys, self._grid_xs = np.mgrid[0:calc_h, 0:calc_w]
            self._grid_res = (calc_h, calc_w)

    def _paint_uv_flow_overlay(self, p: QPainter, r: QRect):
        """
        严格对齐 UE FlowMap 材质链路的双相位预览。
        性能优化：
          - flow_xy 缓存：只在 normal_map 改变时重算
        - 动态计算分辨率：min(原图分辨率, 512)，兼顾性能与效果
          - xs/ys 网格缓存：只在计算分辨率变化时重建
        """
        ref = self._ref_np  # HxWx4 uint8，原图分辨率
        if ref is None:
            return

        # 重建 flow 缓存（只在 normal_map 改变时执行）
        if self._flow_cache_dirty or self._flow_x_cache is None:
            self._rebuild_flow_cache()

        flow_x = self._flow_x_cache  # calc_h x calc_w float32
        flow_y = self._flow_y_cache
        xs = self._grid_xs
        ys = self._grid_ys
        calc_h, calc_w = self._flow_calc_size

        # 参考图也下采样到计算分辨率（缓存，避免每帧 resize）
        ref_h, ref_w = ref.shape[:2]
        if calc_w != ref_w or calc_h != ref_h:
            if self._ref_calc_np is None or \
               self._ref_calc_np.shape[:2] != (calc_h, calc_w):
                from PIL import Image as _PILImage
                ref_pil = _PILImage.fromarray(ref).resize(
                    (calc_w, calc_h), _PILImage.BILINEAR
                )
                self._ref_calc_np = np.array(ref_pil, dtype=np.uint8)
            ref_calc = self._ref_calc_np
        else:
            ref_calc = ref

        # ── 对应 UE：phase0 = frac(time * Speed) ──────────────────────
        speed = max(self.flow_speed, 0.0001)
        raw_t = self._flow_time * speed
        phase_a = math.fmod(raw_t, 1.0)        # phase0：[0, 1)
        phase_b = math.fmod(raw_t + 0.5, 1.0)  # phase1：[0, 1)，错开半个周期

        # ── 对应 UE：uv0 = uv + flow * phase0，转换到像素空间 ──────────
        ox_a = flow_x * phase_a * calc_w
        oy_a = flow_y * phase_a * calc_h
        ox_b = flow_x * phase_b * calc_w
        oy_b = flow_y * phase_b * calc_h

        samp_a = _bilinear_sample_wrap(ref_calc, xs + ox_a, ys + oy_a)  # sampleA
        samp_b = _bilinear_sample_wrap(ref_calc, xs + ox_b, ys + oy_b)  # sampleB

        # ── 对应 UE：base = Texture(UV)（原始无偏移采样）──────────────
        base = ref_calc.astype(np.float32)  # HxWx4 float32

        # ── 对应 UE：blend = abs(phaseA * 2 - 1) ─────────────────────
        blend = abs(phase_a * 2.0 - 1.0)

        # ── 对应 UE：maskValue（暂时无 mask 输入，默认 mask = 1）──────
        mask_val = 1.0

        # ── 对应 UE：colorA = sampleA * mask + base * (1 - mask) ──────
        color_a = samp_a[:, :, :3] * mask_val + base[:, :, :3] * (1.0 - mask_val)
        color_b = samp_b[:, :, :3] * mask_val + base[:, :, :3] * (1.0 - mask_val)

        # ── 对应 UE：finalRGB = lerp(colorA, colorB, blend) ──────────
        # ── 对应 UE：finalA   = lerp(sampleA.a, sampleB.a, blend) ────
        mixed = np.empty_like(samp_a)
        mixed[:, :, :3] = color_a * (1.0 - blend) + color_b * blend
        mixed[:, :, 3] = samp_a[:, :, 3] * (1.0 - blend) + samp_b[:, :, 3] * blend

        result = np.clip(mixed, 0, 255).astype(np.uint8)
        pix = np_rgba_to_qpixmap(result)
        # 放大到显示尺寸（SmoothTransformation 保证质量）
        scaled = pix.scaled(r.width(), r.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        p.drawPixmap(r.topLeft(), scaled)

    # ── 笔刷写入 normal map ──────────────────────────────────────────────
    def _apply_brush(self, cx: float, cy: float,
                     stroke_dx: float = 0.0, stroke_dy: float = 0.0):
        """
        正确的绘制链路：
          1. 根据笔触方向/参数生成目标法线向量 (tx, ty, tz)
          2. 与当前 normal map 区域做 lerp 混合
          3. normalize 混合结果
          4. 写回 normal_map
        擦除模式：目标向量 = flat normal (0, 0, 1)
        """
        s = self.brush_size
        x0 = int(cx - s / 2)
        y0 = int(cy - s / 2)

        # 生成 falloff mask
        mask = make_falloff_mask(s, self.brush_hardness)  # sxs float32

        # ── 确定目标法线向量 (tx, ty, tz) ──────────────────────────────
        if self.mode == "draw":
            if self.follow_stroke:
                # 跟随笔触方向模式
                if stroke_dx != 0.0 or stroke_dy != 0.0:
                    # 有有效方向：用笔触切线方向作为 xy 分量
                    length = math.sqrt(stroke_dx ** 2 + stroke_dy ** 2)
                    tx = stroke_dx / length
                    ty = stroke_dy / length
                else:
                    # 无方向信息（落笔第一点）：跳过绘制，等有方向后再画
                    return
            else:
                # 非跟随模式：默认方向 +Y
                tx, ty = 0.0, 1.0

            # 只取反 tx：UE shader 中 uvA = UV + flow * phase，
            # flow 正值让 UV 正偏→纹理反向移动，X 方向写入取反值使 UE 方向正确。
            # Y 方向不取反：因为 DirectX 导出时 G 通道会翻转（255-G），
            # 已经等价于对 Y 做了取反，如果这里再取反就会双重翻转导致方向错误。
            tx = -tx

            # z 分量：由 x, y 反推，确保法线单位化且朝上
            # 限制 xy 长度，保证 z > 0
            xy_len = math.sqrt(tx * tx + ty * ty)
            if xy_len > 0.99:
                tx = tx / xy_len * 0.99
                ty = ty / xy_len * 0.99
                xy_len = 0.99
            tz = math.sqrt(max(0.0, 1.0 - tx * tx - ty * ty))
        else:
            # 擦除模式：目标向量 = flat normal (0, 0, 1)
            tx, ty, tz = 0.0, 0.0, 1.0

        # ── 计算在 normal_map 中的有效区域 ──────────────────────────────
        vm_x0 = max(0, x0)
        vm_y0 = max(0, y0)
        vm_x1 = min(self._cw, x0 + s)
        vm_y1 = min(self._ch, y0 + s)

        if vm_x0 >= vm_x1 or vm_y0 >= vm_y1:
            return

        mk_x0 = vm_x0 - x0
        mk_y0 = vm_y0 - y0
        mk_x1 = mk_x0 + (vm_x1 - vm_x0)
        mk_y1 = mk_y0 + (vm_y1 - vm_y0)

        sub_mask = mask[mk_y0:mk_y1, mk_x0:mk_x1]  # HxW float32
        # 最终混合权重 = falloff × 强度 × 透明度
        alpha = sub_mask * self.brush_strength * self.brush_opacity  # HxW

        # ── 读取当前区域的法线向量 ──────────────────────────────────
        cur_x = self.normal_map[vm_y0:vm_y1, vm_x0:vm_x1, 0].copy()
        cur_y = self.normal_map[vm_y0:vm_y1, vm_x0:vm_x1, 1].copy()
        cur_z = self.normal_map[vm_y0:vm_y1, vm_x0:vm_x1, 2].copy()

        # ── lerp 混合：current → target ──────────────────────────────────
        mix_x = cur_x * (1.0 - alpha) + tx * alpha
        mix_y = cur_y * (1.0 - alpha) + ty * alpha
        mix_z = cur_z * (1.0 - alpha) + tz * alpha

        # ── normalize 混合结果 ────────────────────────────────────────
        length = np.sqrt(mix_x ** 2 + mix_y ** 2 + mix_z ** 2)
        length = np.maximum(length, 1e-8)  # 防止除以零
        mix_x /= length
        mix_y /= length
        mix_z /= length

        # ── 写回 normal_map ────────────────────────────────────────────
        self.normal_map[vm_y0:vm_y1, vm_x0:vm_x1, 0] = mix_x
        self.normal_map[vm_y0:vm_y1, vm_x0:vm_x1, 1] = mix_y
        self.normal_map[vm_y0:vm_y1, vm_x0:vm_x1, 2] = mix_z

        # 标记缓存失效（normal_map 已改变）
        self._flow_cache_dirty = True
        self._normal_vis_dirty = True

        self.update()
        if self.on_normal_updated:
            self.on_normal_updated(self.normal_map)
    # ── Undo ──────────────────────────────────────────────
    def _push_undo(self):
        """落笔前保存当前 normal_map 快照到 undo 栈。"""
        snapshot = self.normal_map.copy()
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._undo_max:
            self._undo_stack.pop(0)

    def undo(self):
        """撤回上一笔。"""
        if not self._undo_stack:
            return
        self.normal_map = self._undo_stack.pop()
        # 标记缓存失效
        self._flow_cache_dirty = True
        self._normal_vis_dirty = True
        self.update()
        if self.on_normal_updated:
            self.on_normal_updated(self.normal_map)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Z and (e.modifiers() & Qt.ControlModifier):
            self.undo()
        else:
            super().keyPressEvent(e)

    # ── 右键拖动边界限制 ────────────────────────────────────────
    def _clamp_pan(self):
        """
        限制 _pan_offset，确保画布至少有 1/4 宽度和 1/4 高度仍在 widget 内可见。
        即：画布不能完全滑出边界。
        """
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        base_scale = min(lw / self._cw, lh / self._ch)
        scale = base_scale * self._zoom
        pw = int(self._cw * scale)
        ph = int(self._ch * scale)

        # 居中时的基准位置
        cx = (lw - pw) // 2
        cy = (lh - ph) // 2

        # 允许的最大偏移：画布少有 1/4 宽度在可视区内
        margin_x = max(pw // 4, 40)
        margin_y = max(ph // 4, 40)

        # 左边界：画布右边不能超过 widget 左边 + margin
        max_ox = lw - cx - margin_x          # 向右最大移动
        min_ox = -(cx + pw - margin_x)       # 向左最大移动
        max_oy = lh - cy - margin_y
        min_oy = -(cy + ph - margin_y)

        ox = max(min_ox, min(max_ox, self._pan_offset.x()))
        oy = max(min_oy, min(max_oy, self._pan_offset.y()))
        self._pan_offset = QPoint(ox, oy)

    # ── 鼠标事件 ──────────────────────────────────────────────
    def mousePressEvent(self, e):
        self.setFocus()  # 确保能接收键盘事件
        if e.button() == Qt.LeftButton:
            self._drawing = True
            self._push_undo()  # 落笔前保存快照
            cp = self._widget_to_canvas(e.pos())
            if cp:
                if not self.follow_stroke:
                    # 非跟随模式：落笔即绘制（使用默认方向）
                    self._apply_brush(cp[0], cp[1], 0.0, 0.0)
                # 跟随模式：落笔不绘制，仅记录位置，等 mouseMoveEvent 有方向后再绘制
                self._last_cpos = cp
        elif e.button() == Qt.RightButton:
            self._panning = True
            self._pan_start_pos = e.pos()
            self._pan_start_off = QPoint(self._pan_offset)
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, e):
        # 更新鼠标位置（笔刷预览圆用）
        self._mouse_widget_pos = e.pos()

        # 右键拖动画布
        if self._panning and self._pan_start_pos is not None:
            delta = e.pos() - self._pan_start_pos
            self._pan_offset = self._pan_start_off + delta
            self._clamp_pan()
            self._update_pix_rect()
            self.update()
            return

        # 只更新笔刷预览圆区域，避免每次鼠标移动都触发全量重绘
        # 注意：流动动画由定时器驱动，不需要鼠标移动触发
        if not self._drawing:
            # 非绘制状态：只需重绘笔刷圆附近区域
            if self._mouse_widget_pos is not None:
                r_px = self._pix_rect
                if r_px is not None and r_px.width() > 0:
                    scale = r_px.width() / self._cw
                    rad = int(self.brush_size / 2.0 * scale) + 4
                    mx, my = e.pos().x(), e.pos().y()
                    self.update(QRect(mx - rad, my - rad, rad * 2, rad * 2))
                else:
                    self.update()
            return

        if self._drawing and (e.buttons() & Qt.LeftButton):
            cp = self._widget_to_canvas(e.pos())
            if cp:
                if self._last_cpos:
                    lx, ly = self._last_cpos
                    cx2, cy2 = cp
                    dx = cx2 - lx
                    dy = cy2 - ly
                    dist = math.sqrt(dx * dx + dy * dy)
                    # spacing = brush_size × spacing%，连续性参数控制密度
                    spacing = max(1.0, self.brush_size * (self.brush_spacing / 100.0))
                    steps = max(1, int(dist / spacing))
                    for i in range(1, steps + 1):
                        t = i / steps
                        self._apply_brush(
                            lx + dx * t, ly + dy * t,
                            dx, dy  # 笔触方向
                        )
                else:
                    self._apply_brush(cp[0], cp[1], 0.0, 0.0)
                self._last_cpos = cp

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drawing = False
            self._last_cpos = None
        elif e.button() == Qt.RightButton:
            self._panning = False
            self._pan_start_pos = None
            self._pan_start_off = None
            self.setCursor(QCursor(Qt.CrossCursor))

    def leaveEvent(self, e):
        """鼠标离开绘制区时隐藏笔刷预览圆圈。"""
        self._mouse_widget_pos = None
        self.update()
    # ── 清空 ──────────────────────────────────────────────────────────────
    def clear_canvas(self):
        """Reset normal map to flat normal (0,0,1) = RGB(128,128,255)."""
        self.normal_map = self._make_flat_normal()
        self._flow_cache_dirty = True
        self._normal_vis_dirty = True
        self.update()
        if self.on_normal_updated:
            self.on_normal_updated(self.normal_map)

    # ── 获取 packed normal map（用于导出和左侧结果区）────────────────────
    def get_packed_map(self) -> np.ndarray:
        """
        返回 HxWx3 uint8，packed normal map。
        r = x*0.5+0.5, g = y*0.5+0.5, b = z*0.5+0.5
        默认空白画布导出结果就是标准 flat normal RGB(128,128,255)。
        """
        nm = self.normal_map
        out = np.zeros((self._ch, self._cw, 3), dtype=np.uint8)
        out[:, :, 0] = np.clip((nm[:, :, 0] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        out[:, :, 1] = np.clip((nm[:, :, 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        out[:, :, 2] = np.clip((nm[:, :, 2] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        return out

    def get_export_image(self, mode_dx: bool = True,
                         target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        导出 normal map 为 PIL Image（RGB）。
        mode_dx=True：DirectX 模式，翻转 G 通道（y 轴反向）。
        空白画布导出就是标准 flat normal：
          OpenGL: RGB(128, 128, 255)
          DirectX: RGB(128, 127, 255)
        """
        packed = self.get_packed_map()  # HxWx3 uint8
        r_ch = packed[:, :, 0].copy()
        g_ch = packed[:, :, 1].copy()
        b_ch = packed[:, :, 2].copy()

        if mode_dx:
            # DirectX：G 通道反转（y 轴朝下）
            g_ch = 255 - g_ch

        rgb = np.stack([r_ch, g_ch, b_ch], axis=2)
        img = Image.fromarray(rgb, "RGB")

        if target_size:
            img = img.resize(target_size, Image.LANCZOS)

        return img

# ── 法线结果观察区 ──────────────────────────────────────────────────────────────
class VectorResultWidget(QWidget):
    """
    左侧下方：显示当前 normal map 的观察结果。
    - 显示全图：直接显示完整 normal map（蓝紫底色）
    - 仅显示变化区域：只显示相对于 flat normal 发生变化的部分
    使用 paintEvent 绘制，无 resize 循环。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._normal_map: Optional[np.ndarray] = None  # HxWx3 float32
        self._show_all   = False  # False=仅显示变化区域，True=显示完整结果
        self._mode_dx    = True   # True=DirectX（翻转G），False=OpenGL
        self._cached_pix: Optional[QPixmap] = None
        self._cached_key  = None
        self.setStyleSheet(
            "background:#0d0d1a; border:1px solid #313244; border-radius:8px;"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_normal(self, nm: np.ndarray):
        self._normal_map = nm
        self._cached_pix = None
        self.update()

    def set_show_all(self, v: bool):
        self._show_all = v
        self._cached_pix = None
        self.update()

    def set_mode_dx(self, dx: bool):
        if self._mode_dx != dx:
            self._mode_dx = dx
            self._cached_pix = None
            self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._cached_pix = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 26))

        if self._normal_map is None:
            p.setPen(QColor(69, 71, 90))
            p.drawText(self.rect(), Qt.AlignCenter, "法线结果\n绘制后此处显示")
            p.end()
            return

        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            p.end()
            return

        key = (self._show_all, id(self._normal_map), w, h)
        if self._cached_pix is None or self._cached_key != key:
            self._cached_pix = self._build_pix(w, h)
            self._cached_key = key

        if self._cached_pix:
            px = (w - self._cached_pix.width())  // 2
            py = (h - self._cached_pix.height()) // 2
            p.drawPixmap(px, py, self._cached_pix)

        p.end()

    def _build_pix(self, widget_w: int, widget_h: int) -> Optional[QPixmap]:
        nm = self._normal_map
        if nm is None:
            return None

        ch, cw = nm.shape[:2]
        rgba = np.zeros((ch, cw, 4), dtype=np.uint8)

        # 直接将 normal map 打包为可视化颜色
        # 默认 flat normal (0,0,1) → packed RGB(128,128,255) → 蓝紫底色
        r_ch = np.clip((nm[:, :, 0] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        if self._mode_dx:
            # DirectX 模式：G 通道翻转（与导出一致）
            g_ch = np.clip((-nm[:, :, 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        else:
            # OpenGL 模式：G 通道不翻转
            g_ch = np.clip((nm[:, :, 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        b_ch = np.clip((nm[:, :, 2] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

        if self._show_all:
            # 显示完整结果：包括 flat normal 底色区域
            rgba[:, :, 0] = r_ch
            rgba[:, :, 1] = g_ch
            rgba[:, :, 2] = b_ch
            rgba[:, :, 3] = 255
        else:
            # 仅显示变化区域：只有相对于 flat normal 发生变化的像素才可见
            # flat normal = (0,0,1)，对应 xy 分量均为 0
            # 用 xy 分量的 magnitude 判断是否发生变化
            mag_xy = np.sqrt(nm[:, :, 0] ** 2 + nm[:, :, 1] ** 2)
            threshold = 0.02
            changed = mag_xy > threshold

            rgba[:, :, 0] = r_ch
            rgba[:, :, 1] = g_ch
            rgba[:, :, 2] = b_ch
            rgba[:, :, 3] = np.where(
                changed,
                np.clip(mag_xy * 255 * 2, 60, 255).astype(np.uint8),
                0
            ).astype(np.uint8)

        pix = np_rgba_to_qpixmap(rgba)
        scale = min(widget_w / cw, widget_h / ch)
        pw = int(cw * scale)
        ph = int(ch * scale)
        return pix.scaled(pw, ph, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

# ── 主 Tab ────────────────────────────────────────────────────────────
class FlowMapTab(QWidget):
    """向量场贴图编辑器 Tab（法线绘制）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target_size:     Optional[Tuple[int, int]] = None
        self._output_basename: Optional[str]             = None
        self._src_path:        Optional[str]             = None
        self._build_ui()
        self._connect_signals()

    # ── 构建 UI ───────────────────────────────────────────────
    def keyPressEvent(self, e):
        """Tab 层转发 Ctrl+Z 到 canvas。"""
        if e.key() == Qt.Key_Z and (e.modifiers() & Qt.ControlModifier):
            self.canvas.undo()
        else:
            super().keyPressEvent(e)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== 左列 =====
        left = QVBoxLayout()
        left.setSpacing(4)

        # 参考图导入区（上）—— 支持拖拽 + 点击导入，导入后显示缩略图
        ref_header = QHBoxLayout()
        ref_title = QLabel("参考图")
        ref_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ref_title.setStyleSheet("color:#89dceb; font-size:11px; font-weight:600;")
        ref_header.addWidget(ref_title)
        ref_header.addStretch()
        left.addLayout(ref_header)
        self.drop_ref = DropRefWidget()
        self.drop_ref.on_image_loaded = self._on_ref_drop_loaded
        left.addWidget(self.drop_ref, 1)

        # 向量场结果观察区（下）
        result_header = QHBoxLayout()
        normal_title = QLabel("法线结果")
        normal_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        normal_title.setStyleSheet("color:#f38ba8; font-size:11px; font-weight:600;")
        self.chk_show_all = QCheckBox("显示全图")
        self.chk_show_all.setStyleSheet("color:#a6adc8; font-size:10px;")
        self.chk_show_all.setChecked(False)
        result_header.addWidget(normal_title)
        result_header.addStretch()
        result_header.addWidget(self.chk_show_all)
        self.vector_result = VectorResultWidget()
        left.addLayout(result_header)
        left.addWidget(self.vector_result, 1)

        # ===== 中列：工具栏 + 主绘制区 =====
        mid = QVBoxLayout()
        mid.setSpacing(6)

        # 工具栏
        toolbar_group = QGroupBox("向量场绘制工具")
        toolbar_group.setStyleSheet(
            "QGroupBox { border:1px solid #89b4fa; border-radius:8px;"
            "margin-top:16px; padding-top:10px; }"
            "QGroupBox::title { color:#89b4fa; left:10px; }"
        )
        tb = QGridLayout(toolbar_group)
        tb.setHorizontalSpacing(10)
        tb.setVerticalSpacing(5)
        tb.setColumnMinimumWidth(0, 40)
        tb.setColumnStretch(1, 1)

        # 模式切换
        self.btn_draw  = QPushButton("✏ 绘制")
        self.btn_erase = QPushButton("◻ 擦除")
        self.btn_draw.setCheckable(True)
        self.btn_erase.setCheckable(True)
        self.btn_draw.setChecked(True)
        self.btn_draw.setStyleSheet(
            "QPushButton:checked { background:#89b4fa; color:#1e1e2e; }"
        )
        self.btn_erase.setStyleSheet(
            "QPushButton:checked { background:#f38ba8; color:#1e1e2e; }"
        )
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.btn_draw)
        mode_row.addWidget(self.btn_erase)
        tb.addWidget(QLabel("模式："), 0, 0)
        tb.addLayout(mode_row, 0, 1, 1, 3)

        # 笔刷大小（UI 1-200）
        self.slider_size = QSlider(Qt.Horizontal)
        self.slider_size.setRange(1, 200)
        self.slider_size.setValue(40)
        self.edit_size = QLineEdit("40")
        self.edit_size.setFixedWidth(55)
        self.edit_size.setAlignment(Qt.AlignCenter)
        self.edit_size.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d{1,3}$")))
        tb.addWidget(QLabel("笔刷大小："), 1, 0)
        tb.addWidget(self.slider_size, 1, 1, 1, 2)
        tb.addWidget(self.edit_size, 1, 3)

        # 笔刷强度（UI 0-100，实际值 × 0.35 缩小65%）
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(1, 100)
        self.slider_strength.setValue(45)
        self.edit_strength = QLineEdit("45%")
        self.edit_strength.setFixedWidth(55)
        self.edit_strength.setAlignment(Qt.AlignCenter)
        self.edit_strength.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d{1,3}%?$")))
        tb.addWidget(QLabel("笔刷强度："), 2, 0)
        tb.addWidget(self.slider_strength, 2, 1, 1, 2)
        tb.addWidget(self.edit_strength, 2, 3)

        # 笔刷硬度（UI 0-100，实际值 × 0.35 缩小65%）
        self.slider_hardness = QSlider(Qt.Horizontal)
        self.slider_hardness.setRange(0, 100)
        self.slider_hardness.setValue(50)
        self.edit_hardness = QLineEdit("50%")
        self.edit_hardness.setFixedWidth(55)
        self.edit_hardness.setAlignment(Qt.AlignCenter)
        self.edit_hardness.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d{1,3}%?$")))
        tb.addWidget(QLabel("笔刷硬度："), 3, 0)
        tb.addWidget(self.slider_hardness, 3, 1, 1, 2)
        tb.addWidget(self.edit_hardness, 3, 3)

        # 笔刷透明度（默认5%）
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(1, 100)
        self.slider_opacity.setValue(5)
        self.edit_opacity = QLineEdit("5%")
        self.edit_opacity.setFixedWidth(55)
        self.edit_opacity.setAlignment(Qt.AlignCenter)
        self.edit_opacity.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d{1,3}%?$")))
        tb.addWidget(QLabel("笔刷透明度："), 4, 0)
        tb.addWidget(self.slider_opacity, 4, 1, 1, 2)
        tb.addWidget(self.edit_opacity, 4, 3)

        # 笔刷连续性（spacing，默认5%）
        self.slider_spacing = QSlider(Qt.Horizontal)
        self.slider_spacing.setRange(1, 100)
        self.slider_spacing.setValue(5)
        self.edit_spacing = QLineEdit("5%")
        self.edit_spacing.setFixedWidth(55)
        self.edit_spacing.setAlignment(Qt.AlignCenter)
        self.edit_spacing.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d{1,3}%?$")))
        tb.addWidget(QLabel("笔刷连续性："), 5, 0)
        tb.addWidget(self.slider_spacing, 5, 1, 1, 2)
        tb.addWidget(self.edit_spacing, 5, 3)

        # Follow Stroke Direction + Invert Direction
        dir_row = QHBoxLayout()
        self.chk_follow = QCheckBox("跟随笔触方向")
        self.chk_follow.setChecked(True)
        self.chk_follow.setStyleSheet("color:#a6e3a1; font-size:11px;")
        dir_row.addWidget(self.chk_follow)
        tb.addWidget(QLabel("方向控制："), 6, 0)
        tb.addLayout(dir_row, 6, 1, 1, 3)

        # 清空按鈕
        self.btn_clear = QPushButton("清空法线图")
        self.btn_clear.setStyleSheet("color:#f38ba8;")
        tb.addWidget(self.btn_clear, 7, 0, 1, 4)

        # ===== UE 参数预览 GroupBox =====
        ue_group = QGroupBox("UE 参数预览")
        ue_group.setStyleSheet(
            "QGroupBox { border:1px solid #89b4fa; border-radius:8px;"
            "margin-top:16px; padding-top:12px; }"
            "QGroupBox::title { color:#89b4fa; left:10px; font-weight:700; }"
        )
        ue_grid = QGridLayout(ue_group)
        ue_grid.setSpacing(6)
        ue_grid.setContentsMargins(8, 4, 8, 8)

        # 流动速度（对应 UE Speed 参数，默认 0.21，范围 0~1）
        self.slider_flow_speed = QSlider(Qt.Horizontal)
        self.slider_flow_speed.setRange(0, 100)
        self.slider_flow_speed.setValue(21)
        self.edit_flow_speed = QLineEdit("0.21")
        self.edit_flow_speed.setFixedWidth(46)
        self.edit_flow_speed.setAlignment(Qt.AlignCenter)
        self.edit_flow_speed.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d*\\.?\\d*$")))
        ue_grid.addWidget(QLabel("Speed："), 0, 0)
        ue_grid.addWidget(self.slider_flow_speed, 0, 1)
        ue_grid.addWidget(self.edit_flow_speed, 0, 2)

        # 预览强度（对应 UE strangth 参数，默认 0.54，范围 -1~1）
        self.slider_preview_strength = QSlider(Qt.Horizontal)
        self.slider_preview_strength.setRange(-100, 100)
        self.slider_preview_strength.setValue(54)
        self.edit_preview_strength = QLineEdit("0.54")
        self.edit_preview_strength.setFixedWidth(46)
        self.edit_preview_strength.setAlignment(Qt.AlignCenter)
        self.edit_preview_strength.setValidator(QRegularExpressionValidator(QRegularExpression("^-?\\d*\\.?\\d*$")))
        ue_grid.addWidget(QLabel("Strength："), 1, 0)
        ue_grid.addWidget(self.slider_preview_strength, 1, 1)
        ue_grid.addWidget(self.edit_preview_strength, 1, 2)
        # 主绘制区
        self.canvas = VectorMapCanvas()
        self.canvas.on_normal_updated = self._on_normal_updated

        # 原图预览开关
        preview_header = QHBoxLayout()
        preview_header.setContentsMargins(0, 0, 0, 0)
        self.chk_hq_preview = QCheckBox("原图预览")
        self.chk_hq_preview.setChecked(False)
        self.chk_hq_preview.setToolTip("开启：使用原图分辨率计算流动预览（效果准确但较慢）\n关闭：使用 256 分辨率计算（流畅但轻微模糊）")
        self.chk_hq_preview.setStyleSheet(
            "QCheckBox { color:#89b4fa; font-size:12px; font-weight:600; }"
            "QCheckBox::indicator:checked { background-color:#89b4fa; border-color:#89b4fa; }"
        )
        self.chk_hq_preview.stateChanged.connect(self._on_hq_preview_toggled)
        preview_header.addStretch(1)
        preview_header.addWidget(self.chk_hq_preview)

        mid.addWidget(toolbar_group)
        mid.addWidget(ue_group)
        mid.addLayout(preview_header)
        mid.addWidget(self.canvas, 1)

        # ===== 右列：导出设置 =====
        right = QVBoxLayout()
        right.setSpacing(8)

        export_group = QGroupBox("导出设置")
        export_group.setStyleSheet(
            "QGroupBox { border:1px solid #585b70; border-radius:8px;"
            "margin-top:16px; padding-top:12px; }"
            "QGroupBox::title { color:#a6adc8; left:10px; }"
        )
        eg = QVBoxLayout(export_group)
        eg.setSpacing(8)

        eg.addWidget(QLabel("导出命名："))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如 rock（导出为 T_rock_N）")
        self.name_input.setValidator(
            QRegularExpressionValidator(QRegularExpression("^[A-Za-z0-9_]*$"))
        )
        eg.addWidget(self.name_input)

        self.name_preview = QLabel("预览：-")
        self.name_preview.setStyleSheet("color:#a6e3a1; font-weight:700;")
        eg.addWidget(self.name_preview)

        self.btn_apply_name = QPushButton("应用命名")
        eg.addWidget(self.btn_apply_name)

        eg.addWidget(self._hline())

        eg.addWidget(QLabel("一键尺寸："))
        size_row1 = QHBoxLayout()
        size_row1.setSpacing(6)
        size_row1.setContentsMargins(0, 2, 0, 2)
        for s in [32, 64, 128]:
            b = QPushButton(f"{s}x{s}")
            b.setMinimumWidth(72)
            b.setMinimumHeight(30)
            b.clicked.connect(lambda _, sz=s: self._set_size(sz, sz))
            size_row1.addWidget(b)
        eg.addLayout(size_row1)
        size_row2 = QHBoxLayout()
        size_row2.setSpacing(6)
        size_row2.setContentsMargins(0, 0, 0, 2)
        for s in [256, 512]:
            b = QPushButton(f"{s}x{s}")
            b.setMinimumWidth(100)
            b.setMinimumHeight(30)
            b.clicked.connect(lambda _, sz=s: self._set_size(sz, sz))
            size_row2.addWidget(b)
        eg.addLayout(size_row2)

        self.btn_reset_size = QPushButton("重置尺寸")
        eg.addWidget(self.btn_reset_size)

        eg.addWidget(self._hline())

        eg.addWidget(QLabel("导出模式："))
        self.combo_normal_mode = QComboBox()
        self.combo_normal_mode.addItems(["DirectX（默认/UE）", "OpenGL"])
        eg.addWidget(self.combo_normal_mode)

        eg.addWidget(self._hline())

        self.btn_export = QPushButton("导出向量场贴图")
        self.btn_export.setStyleSheet(
            "background:#89b4fa; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        eg.addWidget(self.btn_export)

        self.info_label = QLabel("尺寸：512×512\n模式：DirectX")
        self.info_label.setStyleSheet("color:#6c7086; font-size:11px;")
        self.info_label.setWordWrap(True)
        eg.addWidget(self.info_label)

        eg.addStretch(1)
        right.addWidget(export_group)
        right.addStretch(1)

        # ===== 组装 =====
        root.addLayout(left, 2)
        root.addLayout(mid, 4)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(270)
        root.addWidget(right_widget)

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color:#383850;")
        return f

    # ── 回调 ──────────────────────────────────────────────────────────
    def _on_ref_drop_loaded(self, img: Optional[Image.Image]):
        """DropRefWidget 导入/清除参考图后的回调。"""
        # 同步到画布（绘制区底图）
        self.canvas.set_ref(img)
        # drop_ref 本身已经显示缩略图，无需额外操作

    def _on_normal_updated(self, nm: np.ndarray):
        self.vector_result.update_normal(nm)

    # ── Tab 可见性控制（节省 CPU：不可见时暂停流动动画）──────────────
    def showEvent(self, e):
        super().showEvent(e)
        self.canvas._flow_timer.start(1000 // self.canvas._fps)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.canvas._flow_timer.stop()

    # ── 信号连接 ──────────────────────────────────────────────────────
    def _connect_signals(self):
        self.btn_draw.clicked.connect(lambda: self._set_mode("draw"))
        self.btn_erase.clicked.connect(lambda: self._set_mode("erase"))

        self.slider_size.valueChanged.connect(self._on_size_slider)
        self.edit_size.editingFinished.connect(self._on_size_edit)
        self.slider_strength.valueChanged.connect(self._on_strength_slider)
        self.edit_strength.editingFinished.connect(self._on_strength_edit)
        self.slider_hardness.valueChanged.connect(self._on_hardness_slider)
        self.edit_hardness.editingFinished.connect(self._on_hardness_edit)
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider)
        self.edit_opacity.editingFinished.connect(self._on_opacity_edit)
        self.slider_spacing.valueChanged.connect(self._on_spacing_slider)
        self.edit_spacing.editingFinished.connect(self._on_spacing_edit)
        self.slider_flow_speed.valueChanged.connect(self._on_flow_speed_slider)
        self.edit_flow_speed.editingFinished.connect(self._on_flow_speed_edit)
        self.slider_preview_strength.valueChanged.connect(self._on_preview_strength_slider)
        self.edit_preview_strength.editingFinished.connect(self._on_preview_strength_edit)

        self.chk_follow.toggled.connect(lambda v: setattr(self.canvas, "follow_stroke", v))

        self.chk_show_all.toggled.connect(self.vector_result.set_show_all)

        self.btn_clear.clicked.connect(self._clear_canvas)
        self.btn_apply_name.clicked.connect(self._apply_name)
        self.btn_reset_size.clicked.connect(self._reset_size)
        self.btn_export.clicked.connect(self._export)
        self.name_input.textChanged.connect(self._update_name_preview)
        self.combo_normal_mode.currentIndexChanged.connect(self._update_info)
        self.combo_normal_mode.currentIndexChanged.connect(self._on_normal_mode_changed)

    # ── 模式 ──────────────────────────────────────────────────────────
    def _set_mode(self, mode: str):
        self.canvas.mode = mode
        self.btn_draw.setChecked(mode == "draw")
        self.btn_erase.setChecked(mode == "erase")

    # ── 笔刷参数 ──────────────────────────────────────────────────────
    def _on_size_slider(self, v):
        self.edit_size.blockSignals(True)
        self.edit_size.setText(str(v))
        self.edit_size.blockSignals(False)
        self.canvas.brush_size = v * 2  # 实际大小 = UI值 × 2

    def _on_size_edit(self):
        try:
            v = max(1, min(200, int(self.edit_size.text().strip())))
        except ValueError:
            v = 40
        self.edit_size.setText(str(v))
        self.slider_size.blockSignals(True)
        self.slider_size.setValue(v)
        self.slider_size.blockSignals(False)
        self.canvas.brush_size = v * 2  # 实际大小 = UI值 × 2

    def _on_strength_slider(self, v):
        self.edit_strength.blockSignals(True)
        self.edit_strength.setText(f"{v}%")
        self.edit_strength.blockSignals(False)
        # 实际强度 = UI值 / 100 × 0.5
        self.canvas.brush_strength = v / 100.0 * 0.5

    def _on_strength_edit(self):
        try:
            v = max(1, min(100, int(self.edit_strength.text().replace("%", "").strip())))
        except ValueError:
            v = 45
        self.edit_strength.setText(f"{v}%")
        self.slider_strength.blockSignals(True)
        self.slider_strength.setValue(v)
        self.slider_strength.blockSignals(False)
        self.canvas.brush_strength = v / 100.0 * 0.5

    def _on_hardness_slider(self, v):
        self.edit_hardness.blockSignals(True)
        self.edit_hardness.setText(f"{v}%")
        self.edit_hardness.blockSignals(False)
        # 实际硬度 = UI值 / 100 × 0.5
        self.canvas.brush_hardness = v / 100.0 * 0.5

    def _on_hardness_edit(self):
        try:
            v = max(0, min(100, int(self.edit_hardness.text().replace("%", "").strip())))
        except ValueError:
            v = 50
        self.edit_hardness.setText(f"{v}%")
        self.slider_hardness.blockSignals(True)
        self.slider_hardness.setValue(v)
        self.slider_hardness.blockSignals(False)
        self.canvas.brush_hardness = v / 100.0 * 0.5

    def _on_opacity_slider(self, v):
        self.edit_opacity.blockSignals(True)
        self.edit_opacity.setText(f"{v}%")
        self.edit_opacity.blockSignals(False)
        self.canvas.brush_opacity = v / 100.0

    def _on_opacity_edit(self):
        try:
            v = max(1, min(100, int(self.edit_opacity.text().replace("%", "").strip())))
        except ValueError:
            v = 5
        self.edit_opacity.setText(f"{v}%")
        self.slider_opacity.blockSignals(True)
        self.slider_opacity.setValue(v)
        self.slider_opacity.blockSignals(False)
        self.canvas.brush_opacity = v / 100.0

    def _on_spacing_slider(self, v):
        self.edit_spacing.blockSignals(True)
        self.edit_spacing.setText(f"{v}%")
        self.edit_spacing.blockSignals(False)
        # spacing 值越小 → 笔触越密集；值越大 → 间距越大
        self.canvas.brush_spacing = float(v)

    def _on_spacing_edit(self):
        try:
            v = max(1, min(100, int(self.edit_spacing.text().replace("%", "").strip())))
        except ValueError:
            v = 5
        self.edit_spacing.setText(f"{v}%")
        self.slider_spacing.blockSignals(True)
        self.slider_spacing.setValue(v)
        self.slider_spacing.blockSignals(False)
        self.canvas.brush_spacing = float(v)

    def _on_flow_speed_slider(self, v):
        val = v / 100.0
        self.edit_flow_speed.blockSignals(True)
        self.edit_flow_speed.setText(f"{val:.2f}")
        self.edit_flow_speed.blockSignals(False)
        # 对应 UE 材质实例 Speed 参数（0~1.0）
        self.canvas.flow_speed = val

    def _on_flow_speed_edit(self):
        try:
            val = float(self.edit_flow_speed.text())
            val = max(0.0, min(1.0, val))
        except ValueError:
            val = 0.21
        self.edit_flow_speed.setText(f"{val:.2f}")
        self.slider_flow_speed.blockSignals(True)
        self.slider_flow_speed.setValue(int(round(val * 100)))
        self.slider_flow_speed.blockSignals(False)
        self.canvas.flow_speed = val

    def _on_preview_strength_slider(self, v):
        val = v / 100.0
        self.edit_preview_strength.blockSignals(True)
        self.edit_preview_strength.setText(f"{val:.2f}")
        self.edit_preview_strength.blockSignals(False)
        # 对应 UE 材质实例 strangth 参数（UV 偏移强度，-1.0~1.0）
        self.canvas.preview_strength = val
        # strength 改变时 flow_xy 需要重算
        self.canvas._flow_cache_dirty = True

    def _on_preview_strength_edit(self):
        try:
            val = float(self.edit_preview_strength.text())
            val = max(-1.0, min(1.0, val))
        except ValueError:
            val = 0.54
        self.edit_preview_strength.setText(f"{val:.2f}")
        self.slider_preview_strength.blockSignals(True)
        self.slider_preview_strength.setValue(int(round(val * 100)))
        self.slider_preview_strength.blockSignals(False)
        self.canvas.preview_strength = val
        self.canvas._flow_cache_dirty = True

    def _on_hq_preview_toggled(self, state):
        """切换原图预览模式：开启时用原图分辨率计算，关闭时用 256 分辨率。"""
        self.canvas._hq_preview = bool(state)
        # 分辨率变了，缓存全部失效
        self.canvas._flow_cache_dirty = True
        self.canvas._ref_calc_np = None

    # ── 清空 ──────────────────────────────────────────────────────────
    def _clear_canvas(self):
        reply = QMessageBox.question(
            self, "确认清空", "确定要清空法线图吗？此操作可通过 Ctrl+Z 撤回。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.canvas._push_undo()  # 清空前保存快照，支持撤回
            self.canvas.clear_canvas()

    # ── 尺寸 ──────────────────────────────────────────────────────────
    def _set_size(self, w: int, h: int):
        self._target_size = (w, h)
        self._update_info()

    def _reset_size(self):
        self._target_size = None
        self._update_info()

    def _update_info(self):
        sz = (f"{self._target_size[0]}×{self._target_size[1]}"
              if self._target_size else "512×512（默认）")
        mode_txt = self.combo_normal_mode.currentText()
        self.info_label.setText(f"尺寸：{sz}\n模式：{mode_txt}")

    def _on_normal_mode_changed(self):
        """用户切换 DirectX/OpenGL 模式时，同步更新画布预览和左侧结果区。"""
        dx = self.combo_normal_mode.currentIndex() == 0
        self.canvas.mode_dx = dx
        self.canvas._normal_vis_dirty = True
        self.canvas.update()
        self.vector_result.set_mode_dx(dx)

    # ── 命名 ──────────────────────────────────────────────────────────
    def _update_name_preview(self):
        tag = self.name_input.text().strip()
        self.name_preview.setText(f"预览：T_{tag}_N" if tag else "预览：-")

    def _apply_name(self):
        tag = self.name_input.text().strip()
        if not tag:
            QMessageBox.warning(self, "命名为空", "请先输入导出名称。")
            return
        self._output_basename = f"T_{tag}_N"
        self.name_preview.setText(f"已应用：{self._output_basename}")
        QMessageBox.information(self, "命名已应用",
                                f"导出时将使用：{self._output_basename}")

    def _get_export_name(self) -> str:
        if self._output_basename:
            return self._output_basename
        if self._src_path:
            return os.path.splitext(os.path.basename(self._src_path))[0] + "_N"
        return "FlowMap"

    # ── 导出路径记忆 ──────────────────────────────────────────────────
    def _get_export_dir_cache_path(self) -> str:
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "flowmap_last_export_dir.txt")

    def _load_last_export_dir(self) -> str:
        try:
            with open(self._get_export_dir_cache_path(), "r", encoding="utf-8") as f:
                d = f.read().strip()
                if d and os.path.isdir(d):
                    return d
        except Exception:
            pass
        return os.path.dirname(self._src_path) if self._src_path else ""

    def _save_last_export_dir(self, path: str):
        try:
            with open(self._get_export_dir_cache_path(), "w", encoding="utf-8") as f:
                f.write(os.path.dirname(path))
        except Exception:
            pass

    # ── 导出 ──────────────────────────────────────────────────────────
    def _export(self):
        mode_dx = self.combo_normal_mode.currentIndex() == 0
        export_img = self.canvas.get_export_image(
            mode_dx=mode_dx, target_size=self._target_size
        )
        name = self._get_export_name()
        default_dir = self._load_last_export_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "导出向量场贴图",
            os.path.join(default_dir, f"{name}.png"),
            "PNG (*.png)"
        )
        if not path:
            return
        try:
            export_img.save(path)
            self._save_last_export_dir(path)
            QMessageBox.information(self, "导出完成", f"已导出：\n{path}")
        except Exception as ex:
            QMessageBox.critical(self, "导出失败", str(ex))
