# -*- coding: utf-8 -*-
"""
全能看图 Tab — 支持多种图片格式的浏览器
支持格式：
  Tier 1 (Pillow内置): PNG, JPG, JPEG, BMP, GIF, WebP, ICO
  Tier 2 (Pillow内置): TGA, TIFF, PPM, PCX, DIB
  Tier 3 (需要 psd-tools): PSD
  Tier 4 (需要 PyMuPDF): PDF
  Tier 5 (Pillow + numpy 色调映射): HDR
"""
import os
import math
import struct
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image, ImageSequence

from PySide6.QtCore import Qt, QTimer, QSize, QRectF, QPointF, Signal, QThread, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QWheelEvent,
    QMouseEvent, QPen, QFont, QTransform, QMovie,
    QPalette, QDragEnterEvent, QDropEvent,
    QCursor, QPainterPath, QBrush,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QMessageBox, QScrollArea, QSplitter, QFrame,
    QSizePolicy, QComboBox, QSlider, QToolButton, QSpacerItem,
    QProgressBar, QGraphicsOpacityEffect, QApplication,
)

# =========================================================================
#  常量
# =========================================================================
# Pillow 原生支持的格式
PILLOW_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".ico",
    ".tga", ".tiff", ".tif", ".ppm", ".pcx", ".dib",
}
# 需要 psd-tools 的格式
PSD_EXTS = {".psd", ".psb"}
# 需要 PyMuPDF 的格式
PDF_EXTS = {".pdf"}
# SVG 格式（Qt 内置 QSvgRenderer）
SVG_EXTS = {".svg"}
# HDR 高动态范围格式（Pillow 读取 + Reinhard 色调映射）
HDR_EXTS = {".hdr"}

ALL_SUPPORTED_EXTS = PILLOW_EXTS | PSD_EXTS | PDF_EXTS | SVG_EXTS | HDR_EXTS

# 文件对话框过滤器
_ext_str = " ".join(f"*{e}" for e in sorted(ALL_SUPPORTED_EXTS))
FILE_FILTER = f"所有支持的图片 ({_ext_str});;所有文件 (*)"

# 文件大小上限提醒 (300 MB)
SIZE_WARN_BYTES = 300 * 1024 * 1024


# =========================================================================
#  辅助：可选依赖的延迟导入
# =========================================================================
def _try_import_psd():
    """尝试导入 psd-tools，返回 PSDImage 类或 None"""
    try:
        from psd_tools import PSDImage
        return PSDImage
    except ImportError:
        return None


def _try_import_fitz():
    """尝试导入 PyMuPDF (fitz)，返回 fitz 模块或 None"""
    try:
        import fitz
        return fitz
    except ImportError:
        return None


# =========================================================================
#  辅助：PIL Image → QPixmap
# =========================================================================
def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, 4 * img.width, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# =========================================================================
#  后台加载线程
# =========================================================================
class _LoadWorker(QThread):
    """后台线程加载图片，通过信号报告进度"""
    progress = Signal(int, str)          # (百分比 0-100, 阶段描述)
    finished = Signal(object, dict)       # (加载结果, 附加信息)
    error = Signal(str)                   # 错误消息

    def __init__(self, filepath: str, ext: str, parent=None):
        super().__init__(parent)
        self._filepath = filepath
        self._ext = ext

    def run(self):
        try:
            ext = self._ext
            filepath = self._filepath

            if ext == ".gif":
                # GIF 由主线程处理（QMovie 必须在主线程创建）
                self.progress.emit(50, "正在读取 GIF...")
                # 预读取文件数据以验证文件可读性
                with open(filepath, "rb") as f:
                    _ = f.read()
                self.progress.emit(100, "准备显示...")
                self.finished.emit({"type": "gif", "path": filepath}, {})
                return

            elif ext in SVG_EXTS:
                # SVG 由主线程处理（QSvgRenderer 需要在主线程创建）
                self.progress.emit(50, "正在读取 SVG...")
                with open(filepath, "rb") as f:
                    _ = f.read()  # 验证可读性
                self.progress.emit(100, "准备显示...")
                self.finished.emit({"type": "svg", "path": filepath}, {})
                return

            elif ext in PDF_EXTS:
                self._load_pdf(filepath)
            elif ext in PSD_EXTS:
                self._load_psd(filepath)
            elif ext in HDR_EXTS:
                self._load_hdr(filepath)
            else:
                self._load_pillow(filepath)

        except Exception as e:
            self.error.emit(str(e))

    def _load_pillow(self, filepath: str):
        """后台使用 Pillow 加载图片"""
        self.progress.emit(10, "正在读取文件...")

        img = Image.open(filepath)

        self.progress.emit(30, "正在解码图片...")
        img.load()  # 强制解码到内存

        self.progress.emit(60, "正在转换像素数据...")
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        self.progress.emit(75, "正在生成显示数据...")
        data = img.tobytes("raw", "RGBA")
        w, h = img.width, img.height

        self.progress.emit(90, "即将完成...")
        # 将 bytes、尺寸传回主线程构造 QPixmap（QPixmap 必须在主线程创建）
        self.finished.emit(
            {"type": "pillow", "data": data, "width": w, "height": h},
            {"pil_image": img}
        )

    def _load_hdr(self, filepath: str):
        """后台加载 HDR (Radiance) 格式，使用 Reinhard 色调映射转为 8bit"""
        self.progress.emit(10, "正在读取 HDR 文件...")

        img = Image.open(filepath)
        img.load()  # 强制解码到内存

        self.progress.emit(30, "正在转换为浮点数据...")
        # 转为 RGB 浮点模式
        if img.mode == "F":
            # 单通道浮点 -> 灰度 -> RGB
            arr = np.array(img, dtype=np.float32)
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            img_rgb = img.convert("RGB") if img.mode != "RGB" else img
            arr = np.array(img_rgb, dtype=np.float32)

        self.progress.emit(50, "正在进行色调映射...")
        # Reinhard 色调映射: pixel / (pixel + 1)
        arr = np.maximum(arr, 0.0)  # 去除负值
        arr = arr / (arr + 1.0)

        # Gamma 校正 (sRGB)
        arr = np.power(arr, 1.0 / 2.2)

        # 转为 8bit
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        self.progress.emit(70, "正在生成显示数据...")
        # 转回 PIL Image 再加 Alpha 通道
        result = Image.fromarray(arr, mode="RGB").convert("RGBA")
        data = result.tobytes("raw", "RGBA")
        w, h = result.width, result.height

        self.progress.emit(90, "即将完成...")
        self.finished.emit(
            {"type": "hdr", "data": data, "width": w, "height": h},
            {"pil_image": result}
        )

    def _load_psd(self, filepath: str):
        """后台使用 psd-tools 加载 PSD"""
        PSDImage = _try_import_psd()
        if PSDImage is None:
            self.error.emit("PSD 格式支持需要安装 psd-tools:\n\npip install psd-tools")
            return

        self.progress.emit(10, "正在打开 PSD 文件...")
        psd = PSDImage.open(filepath)

        self.progress.emit(30, "正在合成 PSD 图层（可能较慢）...")
        composite = psd.composite()  # 这一步最耗时

        self.progress.emit(70, "正在转换像素数据...")
        if composite.mode != "RGBA":
            composite = composite.convert("RGBA")
        data = composite.tobytes("raw", "RGBA")
        w, h = composite.width, composite.height

        layer_count = len(psd) if hasattr(psd, '__len__') else 0

        self.progress.emit(90, "即将完成...")
        self.finished.emit(
            {"type": "psd", "data": data, "width": w, "height": h},
            {"pil_image": composite, "psd_layer_count": layer_count}
        )

    def _load_pdf(self, filepath: str):
        """后台使用 PyMuPDF 加载 PDF"""
        fitz = _try_import_fitz()
        if fitz is None:
            self.error.emit("PDF 格式支持需要安装 PyMuPDF:\n\npip install PyMuPDF")
            return

        self.progress.emit(10, "正在打开 PDF 文件...")
        doc = fitz.open(filepath)
        total = len(doc)

        self.progress.emit(40, "正在渲染第 1 页...")
        page = doc.load_page(0)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)

        self.progress.emit(75, "正在转换像素数据...")
        samples = bytes(pix.samples)
        has_alpha = pix.alpha
        pw, ph, stride = pix.width, pix.height, pix.stride

        self.progress.emit(90, "即将完成...")
        # 注意：doc 对象必须在主线程保留引用，所以传回去
        self.finished.emit(
            {"type": "pdf", "samples": samples, "width": pw, "height": ph,
             "stride": stride, "has_alpha": has_alpha, "doc": doc, "total": total},
            {}
        )


# =========================================================================
#  加载进度条叠加控件
# =========================================================================
class _LoadingOverlay(QWidget):
    """叠加在 viewer 中央的加载进度条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # 整体布局
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 容器（带圆角背景）
        self._container = QWidget()
        self._container.setFixedSize(320, 90)
        self._container.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 40, 220);
                border-radius: 12px;
            }
        """)
        c_layout = QVBoxLayout(self._container)
        c_layout.setContentsMargins(20, 14, 20, 14)
        c_layout.setSpacing(8)

        # 阶段文字
        self._stage_label = QLabel("正在加载...")
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stage_label.setStyleSheet("color: #ddd; font-size: 13px; background: transparent;")
        c_layout.addWidget(self._stage_label)

        # 进度条
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 8px;
                background-color: #2a2a35;
                text-align: center;
                color: #ccc;
                font-size: 11px;
            }
            QProgressBar::chunk {
                border-radius: 7px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a9eff, stop:1 #67d4ff
                );
            }
        """)
        c_layout.addWidget(self._progress_bar)

        layout.addWidget(self._container)
        self.hide()

    def start(self):
        """显示进度条并重置"""
        self._progress_bar.setValue(0)
        self._stage_label.setText("正在加载...")
        self.show()
        self.raise_()

    def update_progress(self, value: int, stage: str):
        """更新进度和阶段文字"""
        self._progress_bar.setValue(value)
        if stage:
            self._stage_label.setText(stage)

    def finish(self):
        """隐藏进度条"""
        self._progress_bar.setValue(100)
        self._stage_label.setText("加载完成")
        # 短暂延迟后隐藏，让用户看到100%
        QTimer.singleShot(200, self.hide)

    def reposition(self, parent_w: int, parent_h: int):
        """重新定位到父控件中央"""
        self.setGeometry(0, 0, parent_w, parent_h)


# =========================================================================
#  棋盘格背景 Label（用于透明图片预览）
# =========================================================================
class _CheckerWidget(QWidget):
    """显示图片的核心控件，支持棋盘格背景 + 缩放 + 右键平移"""
    # 视口发生变化时发出（缩放、平移、resize 都会触发）
    view_changed = Signal()



    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._movie: Optional[QMovie] = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._last_mouse = QPointF()
        self._cell = 12  # 棋盘格子大小
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setAcceptDrops(False)  # 由外层处理拖拽
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)  # 屏蔽右键菜单

        # SVG 矢量渲染相关
        self._svg_renderer: Optional[QSvgRenderer] = None
        self._svg_default_size: Optional[QSize] = None  # SVG 的逻辑尺寸，作为 "虚拟 pixmap" 参与缩放计算

        # ---- 吸管取色 ----
        self._eyedropper_mode = False          # 是否处于吸管模式
        self._eyedropper_mouse_pos: Optional[QPointF] = None  # 当前鼠标在控件中的位置
        self._eyedropper_show_info = False     # 是否显示颜色信息文字
        self._eyedropper_color: Optional[QColor] = None  # 当前吸取到的颜色

        # 鼠标停顿 0.5 秒后显示色值信息的定时器
        self._eyedropper_hover_timer = QTimer(self)
        self._eyedropper_hover_timer.setSingleShot(True)
        self._eyedropper_hover_timer.setInterval(500)
        self._eyedropper_hover_timer.timeout.connect(self._on_eyedropper_hover_timeout)

        # 吸取成功动效状态
        self._eyedropper_pick_anim_progress = 0.0  # 0.0~1.0，0 表示无动效
        self._eyedropper_pick_anim_pos: Optional[QPointF] = None  # 动效中心位置
        self._eyedropper_pick_anim_color: Optional[QColor] = None  # 动效颜色
        self._eyedropper_pick_anim_timer = QTimer(self)
        self._eyedropper_pick_anim_timer.setInterval(16)  # ~60fps
        self._eyedropper_pick_anim_timer.timeout.connect(self._on_pick_anim_tick)

    # --- 公开接口 ---
    def set_pixmap(self, pm: Optional[QPixmap]):
        self.stop_movie()
        self.clear_svg()
        self._pixmap = pm
        self._reset_view()
        self.update()

    def set_svg(self, renderer: QSvgRenderer, default_size: QSize):
        """设置 SVG 源：传入渲染器和 SVG 逻辑尺寸

        SVG 模式下不依赖 _pixmap 来绘制，而是在 paintEvent 中
        直接调用 renderer.render() 进行矢量绘制。
        _pixmap 仅用于 minimap 缩略图和「转移至贴图修改」。
        """
        self.stop_movie()
        self._svg_renderer = renderer
        self._svg_default_size = default_size
        self._pixmap = None  # 清除旧 pixmap，SVG 绘制走 renderer
        self._reset_view()
        self.update()

    def set_movie(self, movie: QMovie):
        self.stop_movie()
        self._movie = movie
        self._pixmap = None
        self._movie.frameChanged.connect(self._on_movie_frame)
        self._movie.start()
        self._reset_view()

    def stop_movie(self):
        if self._movie:
            self._movie.stop()
            try:
                self._movie.frameChanged.disconnect(self._on_movie_frame)
            except RuntimeError:
                pass
            self._movie = None

    def clear_svg(self):
        """清理 SVG 渲染器"""
        self._svg_renderer = None
        self._svg_default_size = None

    def current_pixmap(self) -> Optional[QPixmap]:
        if self._pixmap:
            return self._pixmap
        if self._movie:
            return self._movie.currentPixmap()
        # SVG 模式：按当前 default_size 生成一个静态位图快照
        # 供 minimap、转移至贴图修改 等外部调用使用
        if self._svg_renderer and self._svg_default_size:
            return self._render_svg_to_pixmap(self._svg_default_size)
        return None

    def _render_svg_to_pixmap(self, size: QSize) -> QPixmap:
        """将 SVG 渲染为指定尺寸的 QPixmap（辅助方法）"""
        img = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(img)
        self._svg_renderer.render(painter, QRectF(0, 0, size.width(), size.height()))
        painter.end()
        return QPixmap.fromImage(img)

    def _logical_size(self) -> Optional[tuple]:
        """获取当前图片的逻辑像素尺寸 (w, h)

        SVG 模式返回 default_size，其他模式返回 pixmap/movie 尺寸。
        所有缩放/偏移计算都基于此逻辑尺寸。
        """
        if self._svg_renderer and self._svg_default_size:
            return (self._svg_default_size.width(), self._svg_default_size.height())
        pm = None
        if self._pixmap:
            pm = self._pixmap
        elif self._movie:
            pm = self._movie.currentPixmap()
        if pm and not pm.isNull() and pm.width() > 0 and pm.height() > 0:
            return (pm.width(), pm.height())
        return None

    def fit_in_view(self):
        dims = self._logical_size()
        if not dims:
            return
        pw, ph = dims
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return
        sx = ww / pw
        sy = wh / ph
        self._scale = min(sx, sy) * 0.95
        self._offset = QPointF(
            (ww - pw * self._scale) / 2,
            (wh - ph * self._scale) / 2,
        )
        self.update()
        self.view_changed.emit()

    def zoom_1to1(self):
        dims = self._logical_size()
        if not dims:
            return
        self._scale = 1.0
        pw, ph = dims
        ww, wh = self.width(), self.height()
        self._offset = QPointF((ww - pw) / 2, (wh - ph) / 2)
        self.update()
        self.view_changed.emit()

    def get_scale(self) -> float:
        return self._scale

    def get_offset(self) -> QPointF:
        return QPointF(self._offset)

    def set_offset(self, offset: QPointF):
        """由缩略图导航调用，设置偏移量"""
        self._offset = offset
        self.update()
        self.view_changed.emit()

    def get_viewport_rect_in_image(self) -> QRectF:
        """返回当前视口在图片像素坐标系中的可见区域"""
        dims = self._logical_size()
        if not dims or self._scale == 0:
            return QRectF()
        # 视口左上角在图片坐标中的位置
        x = -self._offset.x() / self._scale
        y = -self._offset.y() / self._scale
        w = self.width() / self._scale
        h = self.height() / self._scale
        return QRectF(x, y, w, h)

    # --- 内部 ---
    def _reset_view(self):
        # 延迟一帧让控件获得正确尺寸
        QTimer.singleShot(0, self.fit_in_view)

    def _on_movie_frame(self):
        self.update()

    # --- 绘制 ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 背景
        painter.fillRect(self.rect(), QColor(40, 40, 50))

        # 判断当前是 SVG 矢量模式还是位图模式
        is_svg = bool(self._svg_renderer and self._svg_default_size)

        if is_svg:
            pw = self._svg_default_size.width()
            ph = self._svg_default_size.height()
        else:
            pm = self.current_pixmap()
            if not pm or pm.isNull():
                # 提示文字
                painter.setPen(QColor(140, 140, 160))
                painter.setFont(QFont("Microsoft YaHei", 14))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                                 "拖拽图片到此处\n或点击「打开文件」按钮")
                painter.end()
                return
            pw, ph = pm.width(), pm.height()

        dst = QRectF(self._offset.x(), self._offset.y(),
                     pw * self._scale, ph * self._scale)

        # 棋盘格（只在图片区域绘制）
        clip = dst.intersected(QRectF(self.rect()))
        if not clip.isEmpty():
            c1, c2 = QColor(200, 200, 200), QColor(255, 255, 255)
            cell = max(4, int(self._cell * min(self._scale, 1.0)))
            x0 = int(clip.left())
            y0 = int(clip.top())
            x1 = int(clip.right())
            y1 = int(clip.bottom())
            for y in range(y0, y1, cell):
                for x in range(x0, x1, cell):
                    row = (y - int(dst.y())) // cell
                    col = (x - int(dst.x())) // cell
                    painter.fillRect(x, y,
                                     min(cell, x1 - x), min(cell, y1 - y),
                                     c1 if (row + col) % 2 == 0 else c2)

        # 绘制图片
        if is_svg:
            # SVG 矢量模式：直接用 renderer 绘制到目标矩形
            # 每帧都是矢量渲染，天然无限清晰，无需替换 pixmap
            self._svg_renderer.render(painter, dst)
        else:
            painter.drawPixmap(dst, pm, QRectF(0, 0, pw, ph))

        # 吸管模式：绘制放大镜
        if self._eyedropper_mode and self._eyedropper_mouse_pos is not None:
            self._paint_magnifier(painter)

        # 吸取成功动效（独立于吸管模式，可在动效播放中即使已关闭吸管）
        if self._eyedropper_pick_anim_progress > 0:
            self._paint_pick_animation(painter)

        painter.end()

    # --- 鼠标交互 ---
    def wheelEvent(self, event: QWheelEvent):
        # SVG 模式下也需要响应缩放，用 _logical_size 判断
        if not self._logical_size():
            return
        pos = event.position()
        old_scale = self._scale
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = max(0.01, min(old_scale * factor, 50.0))

        # 以鼠标位置为缩放中心
        self._offset = pos - (pos - self._offset) * (new_scale / old_scale)
        self._scale = new_scale

        # 吸管模式下：缩放也算"动作"，重置停顿定时器
        if self._eyedropper_mode:
            self._eyedropper_show_info = False
            self._eyedropper_hover_timer.stop()
            self._eyedropper_hover_timer.start()
            self._eyedropper_mouse_pos = pos
            self._eyedropper_color = self._get_pixel_color(pos)

        self.update()
        self.view_changed.emit()

    def mousePressEvent(self, event: QMouseEvent):
        # 吸管模式：左键取色
        if self._eyedropper_mode and event.button() == Qt.MouseButton.LeftButton:
            color = self._get_pixel_color(event.position())
            if color and color.isValid() and color.alpha() > 0:
                hex_text = f"{color.red():02X}{color.green():02X}{color.blue():02X}"
                clipboard = QApplication.clipboard()
                clipboard.setText(hex_text)
                # 触发吸取成功动效
                self._start_pick_animation(event.position(), color)
            return
        # 右键 或 中键 拖拽平移
        if event.button() == Qt.MouseButton.RightButton or event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = True
            self._last_mouse = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.position() - self._last_mouse
            self._offset += delta
            self._last_mouse = event.position()
            self.update()
            self.view_changed.emit()
            return
        # 吸管模式：更新放大镜位置，重置停顿定时器
        if self._eyedropper_mode:
            self._eyedropper_mouse_pos = event.position()
            self._eyedropper_show_info = False  # 移动时隐藏色值
            self._eyedropper_hover_timer.stop()
            self._eyedropper_hover_timer.start()  # 重新开始 0.5s 倒计时
            self._eyedropper_color = self._get_pixel_color(event.position())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging:
            self._dragging = False
            if self._eyedropper_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event):
        """鼠标离开控件时隐藏放大镜"""
        if self._eyedropper_mode:
            self._eyedropper_mouse_pos = None
            self._eyedropper_show_info = False
            self._eyedropper_hover_timer.stop()
            self.update()
        super().leaveEvent(event)

    # ---- 吸管取色核心方法 ----
    def set_eyedropper_mode(self, enabled: bool):
        """启用/禁用吸管取色模式"""
        self._eyedropper_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._eyedropper_mouse_pos = None
            self._eyedropper_show_info = False
            self._eyedropper_color = None
            self._eyedropper_hover_timer.stop()
            self.update()

    def _widget_to_image_pos(self, widget_pos: QPointF) -> Optional[tuple]:
        """将控件坐标转换为图片像素坐标 (x, y)"""
        dims = self._logical_size()
        if not dims or self._scale == 0:
            return None
        img_x = (widget_pos.x() - self._offset.x()) / self._scale
        img_y = (widget_pos.y() - self._offset.y()) / self._scale
        pw, ph = dims
        if 0 <= img_x < pw and 0 <= img_y < ph:
            return (int(img_x), int(img_y))
        return None

    def _get_pixel_color(self, widget_pos: QPointF) -> Optional[QColor]:
        """获取控件坐标处对应图片像素的颜色"""
        img_pos = self._widget_to_image_pos(widget_pos)
        if img_pos is None:
            return None
        ix, iy = img_pos

        # SVG 模式：从 default_size 的位图中采样
        if self._svg_renderer and self._svg_default_size:
            pm = self._render_svg_to_pixmap(self._svg_default_size)
            img = pm.toImage()
        elif self._pixmap:
            img = self._pixmap.toImage()
        elif self._movie:
            pm = self._movie.currentPixmap()
            if pm and not pm.isNull():
                img = pm.toImage()
            else:
                return None
        else:
            return None

        if 0 <= ix < img.width() and 0 <= iy < img.height():
            return img.pixelColor(ix, iy)
        return None

    def _on_eyedropper_hover_timeout(self):
        """鼠标停顿 0.5 秒后显示色值信息"""
        if self._eyedropper_mode and self._eyedropper_mouse_pos is not None:
            self._eyedropper_show_info = True
            self.update()

    def _start_pick_animation(self, pos: QPointF, color: QColor):
        """启动吸取成功的扩散光环动效"""
        self._eyedropper_pick_anim_progress = 0.01  # 起始值 > 0 表示动效进行中
        self._eyedropper_pick_anim_pos = QPointF(pos)
        self._eyedropper_pick_anim_color = QColor(color)
        self._eyedropper_pick_anim_timer.start()

    def _on_pick_anim_tick(self):
        """动效帧更新 (~60fps)"""
        self._eyedropper_pick_anim_progress += 0.04  # 约 25 帧完成 ≈ 0.4 秒
        if self._eyedropper_pick_anim_progress >= 1.0:
            self._eyedropper_pick_anim_progress = 0.0
            self._eyedropper_pick_anim_pos = None
            self._eyedropper_pick_anim_color = None
            self._eyedropper_pick_anim_timer.stop()
        self.update()

    def _paint_pick_animation(self, painter: QPainter):
        """绘制吸取成功的扩散光环动效"""
        if self._eyedropper_pick_anim_progress <= 0 or self._eyedropper_pick_anim_pos is None:
            return

        t = self._eyedropper_pick_anim_progress
        pos = self._eyedropper_pick_anim_pos
        color = self._eyedropper_pick_anim_color or QColor(255, 255, 255)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 第一层：吸取颜色的扩散光环
        ring_radius = 8 + t * 35  # 从 8 扩散到 43
        ring_alpha = int(200 * (1.0 - t))  # 淡出
        ring_width = 3.0 * (1.0 - t * 0.5)  # 线宽渐变
        ring_color = QColor(color.red(), color.green(), color.blue(), ring_alpha)
        painter.setPen(QPen(ring_color, ring_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(pos, ring_radius, ring_radius)

        # 第二层：白色内圈（稍快扩散）
        inner_radius = 5 + t * 20
        inner_alpha = int(160 * (1.0 - t))
        inner_color = QColor(255, 255, 255, inner_alpha)
        painter.setPen(QPen(inner_color, 1.5))
        painter.drawEllipse(pos, inner_radius, inner_radius)

        # 中心小圆点（迅速淡出）
        if t < 0.3:
            dot_alpha = int(255 * (1.0 - t / 0.3))
            dot_radius = 3 * (1.0 - t / 0.3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, dot_alpha))
            painter.drawEllipse(pos, dot_radius, dot_radius)

        # "吸取成功" 文字提示（动效中期开始显现）
        if t > 0.2:
            text_alpha = int(255 * min(1.0, (t - 0.2) / 0.3) * (1.0 - max(0, (t - 0.7)) / 0.3))
            text_font = QFont("Microsoft YaHei", 9)
            text_font.setBold(True)
            painter.setFont(text_font)
            fm = painter.fontMetrics()
            tip_text = "吸取成功"
            text_w = fm.horizontalAdvance(tip_text)
            text_h = fm.height()
            text_x = int(pos.x() - text_w / 2)
            text_y = int(pos.y() - ring_radius - 8)
            # 背景圆角矩形
            bg_rect = QRectF(text_x - 4, text_y - text_h + 2, text_w + 8, text_h + 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, int(160 * text_alpha / 255)))
            painter.drawRoundedRect(bg_rect, 4, 4)
            # 文字
            painter.setPen(QColor(255, 255, 255, text_alpha))
            painter.drawText(text_x, text_y, tip_text)
        painter.restore()

    def _paint_magnifier(self, painter: QPainter):
        """绘制圆形放大镜 + 可选色值信息"""
        mouse_pos = self._eyedropper_mouse_pos
        if mouse_pos is None:
            return

        # ---- 参数 ----
        mag_radius = 60       # 放大镜半径
        mag_zoom = 8          # 放大倍数
        border_width = 2
        crosshair_size = 6
        info_margin = 6       # 色值文字与放大镜的间距

        # 放大镜圆心位置：鼠标右下方偏移
        cx = mouse_pos.x() + mag_radius + 20
        cy = mouse_pos.y() + mag_radius + 20

        # 防止超出控件边界
        w, h = self.width(), self.height()
        if cx + mag_radius + 4 > w:
            cx = mouse_pos.x() - mag_radius - 20
        if cy + mag_radius + 4 > h:
            cy = mouse_pos.y() - mag_radius - 20
        # 二次修正（左/上边界）
        if cx - mag_radius < 0:
            cx = mag_radius + 4
        if cy - mag_radius < 0:
            cy = mag_radius + 4

        # ---- 获取放大镜区域的源图像 ----
        # 鼠标在图片坐标中的位置
        img_pos = self._widget_to_image_pos(mouse_pos)

        # 构造一个以鼠标为中心的采样区域
        sample_half = mag_radius / mag_zoom  # 图片坐标系中的半径
        sample_size = int(sample_half * 2)
        if sample_size < 2:
            sample_size = 2

        # 创建放大镜内容图
        mag_img = QImage(mag_radius * 2, mag_radius * 2, QImage.Format.Format_ARGB32_Premultiplied)
        mag_img.fill(QColor(40, 40, 50))
        mag_painter = QPainter(mag_img)
        mag_painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # 像素级，不做平滑

        if img_pos is not None:
            ix, iy = img_pos
            # 采样区域在图片坐标中的范围
            src_x = ix - sample_size // 2
            src_y = iy - sample_size // 2
            src_rect = QRectF(src_x, src_y, sample_size, sample_size)
            dst_rect = QRectF(0, 0, mag_radius * 2, mag_radius * 2)

            # 根据模式获取源图
            if self._svg_renderer and self._svg_default_size:
                pm = self._render_svg_to_pixmap(self._svg_default_size)
                mag_painter.drawPixmap(dst_rect, pm, src_rect)
            elif self._pixmap:
                mag_painter.drawPixmap(dst_rect, self._pixmap, src_rect)
            elif self._movie:
                frame_pm = self._movie.currentPixmap()
                if frame_pm and not frame_pm.isNull():
                    mag_painter.drawPixmap(dst_rect, frame_pm, src_rect)

        # 画十字准线
        center = mag_radius
        pen = QPen(QColor(255, 50, 50, 180), 1)
        mag_painter.setPen(pen)
        mag_painter.drawLine(center - crosshair_size, center, center + crosshair_size, center)
        mag_painter.drawLine(center, center - crosshair_size, center, center + crosshair_size)

        # 画像素网格线（当缩放足够大时）
        if mag_zoom >= 6:
            grid_pen = QPen(QColor(255, 255, 255, 40), 1)
            mag_painter.setPen(grid_pen)
            pixel_size = mag_zoom  # 每个像素在放大镜中占的像素数 ≈ mag_zoom
            # 从中心对齐画网格
            offset_x = center % pixel_size
            offset_y = center % pixel_size
            for gx in range(int(offset_x), mag_radius * 2, pixel_size):
                mag_painter.drawLine(gx, 0, gx, mag_radius * 2)
            for gy in range(int(offset_y), mag_radius * 2, pixel_size):
                mag_painter.drawLine(0, gy, mag_radius * 2, gy)

        mag_painter.end()

        # ---- 将放大镜内容画到主 painter 上（圆形裁剪）----
        painter.save()

        # 圆形裁剪路径
        clip_path = QPainterPath()
        clip_path.addEllipse(QPointF(cx, cy), mag_radius, mag_radius)
        painter.setClipPath(clip_path)
        painter.drawImage(
            QRectF(cx - mag_radius, cy - mag_radius, mag_radius * 2, mag_radius * 2),
            mag_img
        )
        painter.setClipping(False)

        # 圆形边框
        painter.setPen(QPen(QColor(255, 255, 255, 200), border_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), mag_radius, mag_radius)

        # 外环阴影
        painter.setPen(QPen(QColor(0, 0, 0, 80), border_width + 2))
        painter.drawEllipse(QPointF(cx, cy), mag_radius + 1, mag_radius + 1)

        # ---- 色值信息文字（0.5 秒停顿后显示）----
        if self._eyedropper_show_info and self._eyedropper_color and self._eyedropper_color.isValid():
            c = self._eyedropper_color
            alpha = c.alpha()

            if alpha == 0:
                # 完全透明像素
                lines = ["该区域为透明"]
            elif alpha < 255:
                # 半透明像素：显示带 alpha 的色值
                r, g, b = c.red(), c.green(), c.blue()
                h_hsl, s_hsl, l_hsl = c.hslHue(), c.hslSaturation(), c.lightness()
                h_hsv, s_hsv, v_hsv = c.hsvHue(), c.hsvSaturation(), c.value()
                lines = [
                    f"RGBA: {r}, {g}, {b}, {alpha}",
                    f"HSL: {h_hsl}°, {s_hsl}, {l_hsl}",
                    f"HSV: {h_hsv}°, {s_hsv}, {v_hsv}",
                    f"HEX: #{r:02X}{g:02X}{b:02X}{alpha:02X}",
                ]
            else:
                # 完全不透明像素
                r, g, b = c.red(), c.green(), c.blue()
                h_hsl, s_hsl, l_hsl = c.hslHue(), c.hslSaturation(), c.lightness()
                h_hsv, s_hsv, v_hsv = c.hsvHue(), c.hsvSaturation(), c.value()
                lines = [
                    f"RGB: {r}, {g}, {b}",
                    f"HSL: {h_hsl}°, {s_hsl}, {l_hsl}",
                    f"HSV: {h_hsv}°, {s_hsv}, {v_hsv}",
                    f"HEX: #{r:02X}{g:02X}{b:02X}",
                ]

            font = QFont("Consolas", 10)
            font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            line_h = fm.height() + 2
            max_text_w = max(fm.horizontalAdvance(line) for line in lines)

            # 文字背景矩形位置（放大镜正下方）
            info_x = cx - max_text_w / 2 - 8
            info_y = cy + mag_radius + info_margin
            info_w = max_text_w + 16
            info_h = line_h * len(lines) + 10

            # 防止信息框超出控件底部
            if info_y + info_h > h:
                info_y = cy - mag_radius - info_margin - info_h

            # 防止信息框超出控件右侧
            if info_x + info_w > w:
                info_x = w - info_w - 4
            if info_x < 0:
                info_x = 4

            # 色块 + 背景
            bg_rect = QRectF(info_x, info_y, info_w, info_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(30, 30, 40, 220))
            painter.drawRoundedRect(bg_rect, 6, 6)

            # 文字
            painter.setPen(QColor(230, 230, 230))
            text_x = info_x + 8
            text_y = info_y + 6
            for line in lines:
                painter.drawText(int(text_x), int(text_y + fm.ascent()), line)
                text_y += line_h

        painter.restore()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 窗口resize时重新适配
        if self._logical_size():
            self.fit_in_view()




# =========================================================================
#  右下角缩略图导航（MiniMap）
# =========================================================================
class _MiniMapWidget(QWidget):
    """右下角缩略图导航控件
    - 显示完整图片的缩略图
    - 方框标示当前视口区域
    - 方框外有半透明黑色蒙版
    - 可拖拽方框改变聚焦区域
    """
    # 拖拽方框时发出，参数是图片坐标系中的视口中心点
    navigate_to = Signal(float, float)

    MINIMAP_SIZE = 180  # 缩略图控件的最大边长

    _OPACITY_IDLE = 0.3   # 平时不透明度（30%）
    _OPACITY_HOVER = 1.0   # 鼠标悬停时不透明度
    _FADE_DURATION = 200   # 渐变动画时长(ms)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumbnail: Optional[QPixmap] = None
        self._viewer: Optional[_CheckerWidget] = None
        self._dragging = False
        self._img_w = 0  # 原图像素宽
        self._img_h = 0  # 原图像素高
        self.setFixedSize(self.MINIMAP_SIZE, self.MINIMAP_SIZE)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # 透明度效果：平时保持 30% 不透明度，不影响看图
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(self._OPACITY_IDLE)
        self.setGraphicsEffect(self._opacity_effect)

        # 渐变动画
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._fade_anim.setDuration(self._FADE_DURATION)

        self.hide()

    def bind_viewer(self, viewer: '_CheckerWidget'):
        """绑定主视图控件，以获取视口信息"""
        self._viewer = viewer
        viewer.view_changed.connect(self.update)

    def set_image(self, pm: Optional[QPixmap]):
        """设置要显示缩略图的原图"""
        if pm is None or pm.isNull():
            self._thumbnail = None
            self._img_w = 0
            self._img_h = 0
            self.hide()
            return

        self._img_w = pm.width()
        self._img_h = pm.height()

        # 按比例缩放到 MINIMAP_SIZE 内
        thumb = pm.scaled(
            QSize(self.MINIMAP_SIZE - 4, self.MINIMAP_SIZE - 4),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail = thumb

        # 调整控件尺寸与缩略图匹配（加 4px 边距）
        self.setFixedSize(thumb.width() + 4, thumb.height() + 4)
        self.show()
        self.update()

    def _thumb_rect(self) -> QRectF:
        """缩略图在控件中的绘制区域"""
        if not self._thumbnail:
            return QRectF()
        return QRectF(2, 2, self._thumbnail.width(), self._thumbnail.height())

    def _viewport_rect_in_thumb(self) -> QRectF:
        """将主视图的视口区域映射到缩略图坐标系"""
        if not self._viewer or not self._thumbnail or self._img_w == 0 or self._img_h == 0:
            return QRectF()

        vr = self._viewer.get_viewport_rect_in_image()
        if vr.isEmpty():
            return QRectF()

        tr = self._thumb_rect()
        sx = tr.width() / self._img_w
        sy = tr.height() / self._img_h

        return QRectF(
            tr.x() + vr.x() * sx,
            tr.y() + vr.y() * sy,
            vr.width() * sx,
            vr.height() * sy,
        )

    def paintEvent(self, event):
        if not self._thumbnail:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        tr = self._thumb_rect()

        # 1. 绘制缩略图
        painter.drawPixmap(tr, self._thumbnail, QRectF(self._thumbnail.rect()))

        # 2. 视口方框
        vr = self._viewport_rect_in_thumb()
        if not vr.isEmpty():
            # 将方框限制在缩略图范围内（用于蒙版计算）
            clipped_vr = vr.intersected(tr)

            # 3. 蒙版：方框外的区域覆盖半透明黑色
            overlay = QColor(0, 0, 0, 120)
            # 上方蒙版
            if clipped_vr.top() > tr.top():
                painter.fillRect(QRectF(tr.left(), tr.top(), tr.width(), clipped_vr.top() - tr.top()), overlay)
            # 下方蒙版
            if clipped_vr.bottom() < tr.bottom():
                painter.fillRect(QRectF(tr.left(), clipped_vr.bottom(), tr.width(), tr.bottom() - clipped_vr.bottom()), overlay)
            # 左侧蒙版（仅在方框高度范围内）
            if clipped_vr.left() > tr.left():
                painter.fillRect(QRectF(tr.left(), clipped_vr.top(), clipped_vr.left() - tr.left(), clipped_vr.height()), overlay)
            # 右侧蒙版
            if clipped_vr.right() < tr.right():
                painter.fillRect(QRectF(clipped_vr.right(), clipped_vr.top(), tr.right() - clipped_vr.right(), clipped_vr.height()), overlay)

            # 4. 绘制视口方框边框
            pen = QPen(QColor(255, 200, 50), 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(vr)

        # 5. 控件边框
        painter.setPen(QPen(QColor(80, 80, 100), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        painter.end()

    # --- 透明度：鼠标悬停时变不透明 ---
    def enterEvent(self, event):
        self._animate_opacity(self._OPACITY_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event):
        # 拖拽过程中不恢复透明，避免拖到控件外时突然变透明
        if not self._dragging:
            self._animate_opacity(self._OPACITY_IDLE)
        super().leaveEvent(event)

    def _animate_opacity(self, target: float):
        """平滑过渡到目标不透明度"""
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(target)
        self._fade_anim.start()

    # --- 鼠标交互：拖拽方框导航 ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._navigate_to_pos(event.position())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._navigate_to_pos(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            # 如果鼠标已在控件外，恢复透明
            if not self.rect().contains(event.position().toPoint()):
                self._animate_opacity(self._OPACITY_IDLE)

    def _navigate_to_pos(self, pos: QPointF):
        """将缩略图上的点击位置转换为图片坐标，通知主视图导航"""
        if not self._thumbnail or self._img_w == 0 or self._img_h == 0:
            return
        tr = self._thumb_rect()
        # 缩略图坐标 → 图片坐标
        img_x = (pos.x() - tr.x()) / tr.width() * self._img_w
        img_y = (pos.y() - tr.y()) / tr.height() * self._img_h
        self.navigate_to.emit(img_x, img_y)


# =========================================================================
#  图片信息面板
# =========================================================================
class _InfoPanel(QGroupBox):
    """右侧的图片元数据信息面板"""

    def __init__(self, parent=None):
        super().__init__("图片信息", parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 16px;
                color: #ddd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QLabel {
                font-size: 12px;
                color: #ccc;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self._labels = {}
        fields = [
            ("file_name", "文件名"),
            ("file_path", "路径"),
            ("file_size", "文件大小"),
            ("format", "格式"),
            ("dimensions", "尺寸"),
            ("color_mode", "色彩模式"),
            ("bit_depth", "位深度"),
            ("dpi", "DPI"),
            ("frames", "帧数"),
            ("layers", "图层数"),
            ("pdf_pages", "PDF页数"),
            ("modified_time", "修改时间"),
        ]
        for key, title in fields:
            row = QHBoxLayout()
            title_label = QLabel(f"{title}：")
            title_label.setFixedWidth(70)
            title_label.setStyleSheet("color: #999; font-size: 12px;")
            value_label = QLabel("—")
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(title_label)
            row.addWidget(value_label, 1)
            layout.addLayout(row)
            self._labels[key] = value_label

        layout.addStretch()

    def clear_info(self):
        for lbl in self._labels.values():
            lbl.setText("—")

    def set_info(self, info: dict):
        self.clear_info()
        for key, value in info.items():
            if key in self._labels and value:
                self._labels[key].setText(str(value))


# =========================================================================
#  PDF 页面控制栏
# =========================================================================
class _PdfControlBar(QWidget):
    """PDF分页浏览的控制栏"""
    page_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._btn_prev = QPushButton("◀ 上一页")
        self._btn_prev.setFixedWidth(90)
        self._page_label = QLabel("第 1 / 1 页")
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_label.setFixedWidth(120)
        self._btn_next = QPushButton("下一页 ▶")
        self._btn_next.setFixedWidth(90)

        layout.addStretch()
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._page_label)
        layout.addWidget(self._btn_next)
        layout.addStretch()

        self._btn_prev.clicked.connect(lambda: self._go(-1))
        self._btn_next.clicked.connect(lambda: self._go(1))

        self._current = 0
        self._total = 0
        self.hide()

    def setup(self, total_pages: int, start_page: int = 0):
        self._total = total_pages
        self._current = start_page
        self._update_label()
        self.setVisible(total_pages > 1)

    def _go(self, delta: int):
        new_page = self._current + delta
        if 0 <= new_page < self._total:
            self._current = new_page
            self._update_label()
            self.page_changed.emit(self._current)

    def _update_label(self):
        self._page_label.setText(f"第 {self._current + 1} / {self._total} 页")
        self._btn_prev.setEnabled(self._current > 0)
        self._btn_next.setEnabled(self._current < self._total - 1)

    def current_page(self) -> int:
        return self._current


# =========================================================================
#  GIF 控制栏
# =========================================================================
class _GifControlBar(QWidget):
    """GIF动图的播放控制栏"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._btn_play = QPushButton("⏸ 暂停")
        self._btn_play.setFixedWidth(80)
        self._frame_label = QLabel("帧: 0/0")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setFixedWidth(100)
        self._speed_label = QLabel("速度:")
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.setFixedWidth(70)

        layout.addStretch()
        layout.addWidget(self._btn_play)
        layout.addWidget(self._frame_label)
        layout.addWidget(self._speed_label)
        layout.addWidget(self._speed_combo)
        layout.addStretch()

        self._playing = True
        self._movie: Optional[QMovie] = None
        self._btn_play.clicked.connect(self._toggle_play)
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self.hide()

    def bind_movie(self, movie: QMovie, frame_count: int):
        self._movie = movie
        self._frame_count = frame_count
        self._playing = True
        self._btn_play.setText("⏸ 暂停")
        self._frame_label.setText(f"帧: 1/{frame_count}")
        self._speed_combo.setCurrentText("1x")
        movie.frameChanged.connect(self._on_frame_changed)
        self.show()

    def unbind(self):
        if self._movie:
            try:
                self._movie.frameChanged.disconnect(self._on_frame_changed)
            except RuntimeError:
                pass
        self._movie = None
        self.hide()

    def _toggle_play(self):
        if not self._movie:
            return
        if self._playing:
            self._movie.setPaused(True)
            self._btn_play.setText("▶ 播放")
        else:
            self._movie.setPaused(False)
            self._btn_play.setText("⏸ 暂停")
        self._playing = not self._playing

    def _on_speed_changed(self, text: str):
        if not self._movie:
            return
        speed_map = {"0.25x": 400, "0.5x": 200, "1x": 100, "2x": 50, "4x": 25}
        speed = speed_map.get(text, 100)
        self._movie.setSpeed(speed)

    def _on_frame_changed(self, frame_num: int):
        self._frame_label.setText(f"帧: {frame_num + 1}/{self._frame_count}")


# =========================================================================
#  文件夹导航控制栏
# =========================================================================
class _FolderArrowOverlay(QWidget):
    """
    文件夹浏览时在预览区两侧显示半透明浮动箭头 + 底部页码指示器。
    本身是一个透明的覆盖层 QWidget，叠加在 viewer_container 上方。
    """

    _ARROW_STYLE = """
        QPushButton {
            background: rgba(0, 0, 0, 120);
            color: rgba(255, 255, 255, 200);
            border: none;
            border-radius: 22px;
            font-size: 24px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: rgba(0, 0, 0, 200);
            color: rgba(255, 255, 255, 255);
        }
        QPushButton:pressed {
            background: rgba(0, 0, 0, 240);
        }
    """

    _PAGE_STYLE = """
        QLabel {
            background: rgba(0, 0, 0, 140);
            color: rgba(255, 255, 255, 220);
            border-radius: 10px;
            padding: 4px 14px;
            font-size: 12px;
        }
    """

    def __init__(self, parent_container: QWidget, callback):
        """
        parent_container: 覆盖层的父容器（_viewer_container）
        callback: 翻页回调 callback(new_index: int)
        """
        super().__init__(parent_container)
        self._callback = callback
        self._current = 0
        self._total = 0

        # 覆盖层自身透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        # 注意：不设置 WA_TransparentForMouseEvents，否则子按钮也收不到点击

        # 左箭头
        self._btn_left = QPushButton("◀", self)
        self._btn_left.setFixedSize(44, 44)
        self._btn_left.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_left.setStyleSheet(self._ARROW_STYLE)
        self._btn_left.clicked.connect(lambda: self._go(-1))
        self._btn_left.hide()

        # 右箭头
        self._btn_right = QPushButton("▶", self)
        self._btn_right.setFixedSize(44, 44)
        self._btn_right.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_right.setStyleSheet(self._ARROW_STYLE)
        self._btn_right.clicked.connect(lambda: self._go(1))
        self._btn_right.hide()

        # 底部页码指示器
        self._page_label = QLabel("", self)
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_label.setStyleSheet(self._PAGE_STYLE)
        self._page_label.hide()

        # 覆盖层默认隐藏
        self.hide()

    def setup(self, total: int, start: int = 0):
        """进入文件夹浏览模式，设置总数和起始索引"""
        self._total = total
        self._current = start
        if total > 1:
            self.show()
        self._update_ui()

    def hide_all(self):
        """隐藏覆盖层（退出文件夹浏览模式时调用）"""
        self._btn_left.hide()
        self._btn_right.hide()
        self._page_label.hide()
        self._total = 0
        self.hide()

    def is_active(self) -> bool:
        """是否处于文件夹浏览模式"""
        return self._total > 1

    def reposition(self):
        """跟随父容器尺寸，调整覆盖层和箭头位置"""
        parent = self.parentWidget()
        if not parent:
            return
        # 覆盖层跟父容器一样大
        self.setGeometry(0, 0, parent.width(), parent.height())

        if self._total <= 1:
            return
        cw = self.width()
        ch = self.height()
        margin = 14
        arrow_h = 44
        # 左箭头：垂直居中，紧贴左侧
        self._btn_left.move(margin, (ch - arrow_h) // 2)
        # 右箭头：垂直居中，紧贴右侧
        self._btn_right.move(cw - 44 - margin, (ch - arrow_h) // 2)
        # 页码：底部居中
        self._page_label.adjustSize()
        pw = self._page_label.width()
        self._page_label.move((cw - pw) // 2, ch - 40)

        # 确保箭头和页码在最上层
        self.raise_()

    def go_prev(self):
        """外部调用：切换到上一张"""
        self._go(-1)

    def go_next(self):
        """外部调用：切换到下一张"""
        self._go(1)

    def _go(self, delta: int):
        new_idx = self._current + delta
        if 0 <= new_idx < self._total:
            self._current = new_idx
            self._update_ui()
            self._callback(self._current)

    def _update_ui(self):
        """更新箭头的可见性和页码文字"""
        if self._total <= 1:
            self.hide_all()
            return
        # 左箭头：第一张时消失
        self._btn_left.setVisible(self._current > 0)
        # 右箭头：最后一张时消失
        self._btn_right.setVisible(self._current < self._total - 1)
        # 页码
        self._page_label.setText(f"  {self._current + 1} / {self._total}  ")
        self._page_label.adjustSize()
        self._page_label.show()
        self.show()
        # 重新定位
        self.reposition()

    # ---- 鼠标事件穿透：非按钮区域的事件传递给底层 viewer ----
    def _is_on_button(self, pos):
        """检查点击位置是否在某个可见按钮上"""
        for btn in (self._btn_left, self._btn_right):
            if btn.isVisible() and btn.geometry().contains(pos):
                return True
        return False

    def mousePressEvent(self, event):
        if not self._is_on_button(event.pos()):
            event.ignore()    # 穿透到底层
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._is_on_button(event.pos()):
            event.ignore()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if not self._is_on_button(event.pos()):
            event.ignore()
        else:
            super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self._is_on_button(event.pos()):
            event.ignore()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        event.ignore()  # 滚轮始终穿透，让底层 viewer 缩放


# =========================================================================
#  主 Tab: ImageViewerTab
# =========================================================================
class ImageViewerTab(QWidget):
    """全能看图 Tab"""
    # 转移到贴图修改 tab 的信号，参数是临时 PNG 文件路径
    transfer_to_texture = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._current_file: Optional[str] = None
        self._pdf_doc = None  # PyMuPDF doc 对象
        self._pil_image: Optional[Image.Image] = None

        # 文件夹浏览状态
        self._folder_files: list = []   # 文件夹中所有支持的文件路径
        self._folder_index: int = -1    # 当前浏览索引

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ---- 左侧：图片预览区 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # 工具栏
        toolbar = QHBoxLayout()
        self._btn_open = QPushButton("🖼 打开文件")
        self._btn_open.setFixedHeight(32)
        self._btn_open_folder = QPushButton("📂 打开文件夹")
        self._btn_open_folder.setFixedHeight(32)
        self._btn_fit = QPushButton("🔲 适应窗口")
        self._btn_fit.setFixedHeight(32)
        self._btn_1to1 = QPushButton("1:1 原始大小")
        self._btn_1to1.setFixedHeight(32)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(60)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_label.setStyleSheet("color: #aaa; font-size: 12px;")

        self._btn_transfer = QPushButton("📤 转移至贴图修改")
        self._btn_transfer.setFixedHeight(32)
        self._btn_transfer.setEnabled(False)
        self._btn_transfer.setToolTip("将当前预览的图片以 PNG 格式转移到「贴图修改」tab 中")

        self._btn_eyedropper = QPushButton("🔍 吸管取色")
        self._btn_eyedropper.setFixedHeight(32)
        self._btn_eyedropper.setCheckable(True)
        self._btn_eyedropper.setToolTip("启用吸管取色\n左键点击画面取色并复制 HEX 色值到剪贴板\n再次点击按钮或按 ESC 退出")

        toolbar.addWidget(self._btn_open)
        toolbar.addWidget(self._btn_open_folder)
        toolbar.addWidget(self._btn_fit)
        toolbar.addWidget(self._btn_1to1)
        toolbar.addWidget(self._btn_transfer)
        toolbar.addWidget(self._btn_eyedropper)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("缩放:"))
        toolbar.addWidget(self._zoom_label)

        left_layout.addLayout(toolbar)

        # 图片显示区（使用容器 QWidget 来叠加 minimap）
        self._viewer_container = QWidget()
        self._viewer_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        container_layout = QVBoxLayout(self._viewer_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self._viewer = _CheckerWidget(self._viewer_container)
        self._viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        container_layout.addWidget(self._viewer)

        # 右下角缩略图导航（叠加在 viewer 上方）
        self._minimap = _MiniMapWidget(self._viewer)
        self._minimap.bind_viewer(self._viewer)
        self._minimap.navigate_to.connect(self._on_minimap_navigate)

        left_layout.addWidget(self._viewer_container, 1)

        # GIF 控制栏
        self._gif_bar = _GifControlBar()
        left_layout.addWidget(self._gif_bar)

        # PDF 控制栏
        self._pdf_bar = _PdfControlBar()
        left_layout.addWidget(self._pdf_bar)

        # 文件夹浏览浮动箭头（叠加在 viewer_container 上，浮于 viewer 之上）
        self._folder_arrows = _FolderArrowOverlay(
            self._viewer_container, self._on_folder_index_changed
        )

        # ---- 右侧：信息面板 ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._info_panel = _InfoPanel()
        right_layout.addWidget(self._info_panel)

        # 格式支持状态
        status_group = QGroupBox("格式支持")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 13px;
                border: 1px solid #444; border-radius: 6px;
                margin-top: 8px; padding-top: 16px; color: #ddd;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
            }
            QLabel { font-size: 11px; color: #ccc; }
        """)
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(6)

        # --- 内置支持的格式 ---
        lbl_builtin_title = QLabel("内置支持")
        lbl_builtin_title.setStyleSheet("color: #8cf; font-weight: bold; font-size: 12px;")
        status_layout.addWidget(lbl_builtin_title)

        builtin_formats = [
            ("PNG", ".png"),
            ("JPEG", ".jpg .jpeg"),
            ("BMP", ".bmp"),
            ("GIF", ".gif (支持动图)"),
            ("WebP", ".webp"),
            ("ICO", ".ico"),
            ("TGA", ".tga"),
            ("TIFF", ".tiff .tif"),
            ("PPM", ".ppm"),
            ("PCX", ".pcx"),
            ("DIB", ".dib"),
            ("SVG", ".svg (矢量动态渲染)"),
            ("HDR", ".hdr (Reinhard色调映射)"),
        ]
        for name, exts in builtin_formats:
            lbl = QLabel(f"  ✅ {name}  <span style='color:#888'>{exts}</span>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            status_layout.addWidget(lbl)

        # --- 分隔线 ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #555;")
        status_layout.addWidget(sep1)

        # --- 需要额外依赖的格式 ---
        lbl_optional_title = QLabel("扩展支持（需额外依赖）")
        lbl_optional_title.setStyleSheet("color: #8cf; font-weight: bold; font-size: 12px;")
        status_layout.addWidget(lbl_optional_title)

        psd_ok = _try_import_psd() is not None
        pdf_ok = _try_import_fitz() is not None

        optional_formats = [
            ("PSD / PSB", ".psd .psb", psd_ok, "psd-tools"),
            ("PDF", ".pdf (支持翻页)", pdf_ok, "PyMuPDF"),
        ]
        for name, exts, installed, pkg in optional_formats:
            icon = "✅" if installed else "❌"
            color = "#8f8" if installed else "#f88"
            tip = f"已安装 {pkg}" if installed else f"未安装 (pip install {pkg})"
            lbl = QLabel(f"  {icon} {name}  <span style='color:#888'>{exts}</span>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet(f"color: {color};")
            status_layout.addWidget(lbl)
            lbl_tip = QLabel(f"      {tip}")
            lbl_tip.setStyleSheet(f"color: {color}; font-size: 10px;")
            status_layout.addWidget(lbl_tip)

        # --- 分隔线 ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #555;")
        status_layout.addWidget(sep2)

        right_layout.addWidget(status_group)

        right_layout.addStretch()
        right_widget.setFixedWidth(280)

        # ---- 组装 ----
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget)

        # ---- 信号连接 ----
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        self._btn_fit.clicked.connect(self._viewer.fit_in_view)
        self._btn_1to1.clicked.connect(self._viewer.zoom_1to1)
        self._btn_transfer.clicked.connect(self._on_transfer_to_texture)
        self._btn_eyedropper.toggled.connect(self._toggle_eyedropper)
        self._pdf_bar.page_changed.connect(self._on_pdf_page_changed)
        # _folder_arrows 的回调已在构造时传入，无需额外连接信号

        # 定时更新缩放比例显示
        self._zoom_timer = QTimer()
        self._zoom_timer.setInterval(100)
        self._zoom_timer.timeout.connect(self._update_zoom_label)
        self._zoom_timer.start()

        # viewer resize 时重新定位 minimap
        self._viewer.view_changed.connect(self._reposition_minimap)

        # 加载进度条叠加控件
        self._loading_overlay = _LoadingOverlay(self._viewer)
        # 后台加载线程引用
        self._load_worker: Optional[_LoadWorker] = None

    # =====================================================================
    #  拖拽支持
    # =====================================================================
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in ALL_SUPPORTED_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            # 拖拽打开单个文件时退出文件夹浏览模式
            self._folder_files = []
            self._folder_index = -1
            self._folder_arrows.hide_all()
            self._load_file(path)

    # =====================================================================
    #  打开文件对话框
    # =====================================================================
    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", FILE_FILTER)
        if path:
            # 打开单个文件时退出文件夹浏览模式
            self._folder_files = []
            self._folder_index = -1
            self._folder_arrows.hide_all()
            self._load_file(path)

    # =====================================================================
    #  核心：加载文件
    # =====================================================================
    def _load_file(self, filepath: str):
        filepath = os.path.normpath(filepath)
        ext = os.path.splitext(filepath)[1].lower()

        if ext not in ALL_SUPPORTED_EXTS:
            QMessageBox.warning(self, "不支持的格式", f"不支持的文件格式: {ext}")
            return

        # 文件大小检查
        try:
            fsize = os.path.getsize(filepath)
        except OSError as e:
            QMessageBox.warning(self, "文件错误", f"无法读取文件:\n{e}")
            return

        if fsize > SIZE_WARN_BYTES:
            ret = QMessageBox.question(
                self, "文件较大",
                f"文件大小为 {self._fmt_size(fsize)}，加载可能较慢。\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        # 清理旧状态
        self._cleanup()
        self._current_file = filepath

        # 显示加载进度条
        self._loading_overlay.reposition(self._viewer.width(), self._viewer.height())
        self._loading_overlay.start()

        # 禁用打开按钮，防止重复加载
        self._btn_open.setEnabled(False)
        self._btn_open_folder.setEnabled(False)

        # 启动后台加载线程
        self._load_worker = _LoadWorker(filepath, ext)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _cleanup(self):
        """清理旧的加载状态"""
        # 如果有正在运行的加载线程，等待结束
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.wait(2000)  # 最多等2秒
        self._load_worker = None

        self._viewer.clear_svg()
        self._viewer.stop_movie()
        self._viewer.set_pixmap(None)
        self._minimap.set_image(None)
        self._gif_bar.unbind()
        self._pdf_bar.hide()
        # 注意：这里不清理 _folder_arrows，因为文件夹浏览模式下
        # 每次切换图片都会调用 _cleanup，箭头需要保留
        # 退出文件夹模式由 _on_open_file / dropEvent 单独处理
        self._info_panel.clear_info()
        self._loading_overlay.hide()
        self._btn_transfer.setEnabled(False)
        if self._pdf_doc:
            self._pdf_doc.close()
            self._pdf_doc = None
        self._pil_image = None

    # =====================================================================
    #  后台加载回调
    # =====================================================================
    def _on_load_progress(self, value: int, stage: str):
        """后台线程进度回调（在主线程中执行）"""
        self._loading_overlay.update_progress(value, stage)

    def _on_load_finished(self, result: dict, extra: dict):
        """后台线程加载完成回调（在主线程中执行）"""
        self._loading_overlay.finish()
        self._btn_open.setEnabled(True)
        self._btn_open_folder.setEnabled(True)

        filepath = self._current_file
        ext = os.path.splitext(filepath)[1].lower() if filepath else ""

        try:
            rtype = result.get("type", "")

            if rtype == "pillow":
                # 在主线程中构造 QPixmap
                data = result["data"]
                w, h = result["width"], result["height"]
                qimg = QImage(data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
                pm = QPixmap.fromImage(qimg.copy())
                self._viewer.set_pixmap(pm)
                self._minimap.set_image(pm)
                self._reposition_minimap()
                self._pil_image = extra.get("pil_image")

            elif rtype == "gif":
                # GIF 必须在主线程处理 QMovie
                gif_path = result["path"]
                self._finalize_gif(gif_path)

            elif rtype == "svg":
                # SVG 在主线程中用 QSvgRenderer 渲染
                svg_path = result["path"]
                self._finalize_svg(svg_path)

            elif rtype == "hdr":
                # HDR 色调映射后的 8bit RGBA 数据
                data = result["data"]
                w, h = result["width"], result["height"]
                qimg = QImage(data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
                pm = QPixmap.fromImage(qimg.copy())
                self._viewer.set_pixmap(pm)
                self._minimap.set_image(pm)
                self._reposition_minimap()
                self._pil_image = extra.get("pil_image")

            elif rtype == "psd":
                data = result["data"]
                w, h = result["width"], result["height"]
                qimg = QImage(data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
                pm = QPixmap.fromImage(qimg.copy())
                self._viewer.set_pixmap(pm)
                self._minimap.set_image(pm)
                self._reposition_minimap()
                self._pil_image = extra.get("pil_image")
                self._psd_layer_count = extra.get("psd_layer_count", 0)

            elif rtype == "pdf":
                doc = result["doc"]
                self._pdf_doc = doc
                total = result["total"]
                self._pdf_bar.setup(total, 0)
                # 用后台已渲染好的第一页数据构造 QPixmap
                samples = result["samples"]
                pw, ph = result["width"], result["height"]
                stride = result["stride"]
                has_alpha = result["has_alpha"]
                fmt = QImage.Format.Format_RGBA8888 if has_alpha else QImage.Format.Format_RGB888
                qimg = QImage(samples, pw, ph, stride, fmt)
                pm = QPixmap.fromImage(qimg.copy())
                self._viewer.set_pixmap(pm)
                self._minimap.set_image(pm)
                self._reposition_minimap()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"处理加载结果时出错:\n{e}")
            return

        # 启用转移按钮
        self._btn_transfer.setEnabled(True)

        # 收集并显示文件信息
        if filepath:
            self._show_file_info(filepath, ext)

    def _on_load_error(self, msg: str):
        """后台线程加载出错回调"""
        self._loading_overlay.hide()
        self._btn_open.setEnabled(True)
        self._btn_open_folder.setEnabled(True)
        QMessageBox.critical(self, "加载失败", f"加载文件失败:\n{msg}")

    def _finalize_gif(self, filepath: str):
        """在主线程中完成 GIF 的 QMovie 创建和绑定"""
        movie = QMovie(filepath)
        if not movie.isValid():
            # 回退到同步 Pillow 加载（GIF无效时文件通常很小）
            self._sync_load_pillow(filepath)
            return

        frame_count = movie.frameCount()
        if frame_count <= 1:
            self._sync_load_pillow(filepath)
            return

        self._viewer.set_movie(movie)
        self._gif_bar.bind_movie(movie, frame_count)

        first_frame = movie.currentPixmap()
        if not first_frame.isNull():
            self._minimap.set_image(first_frame)
            self._reposition_minimap()

        try:
            self._pil_image = Image.open(filepath)
        except Exception:
            pass

    def _sync_load_pillow(self, filepath: str):
        """同步加载（仅用于 GIF 回退等小文件场景）"""
        img = Image.open(filepath)
        img.load()
        self._pil_image = img
        pm = _pil_to_qpixmap(img)
        self._viewer.set_pixmap(pm)
        self._minimap.set_image(pm)
        self._reposition_minimap()

    def _finalize_svg(self, filepath: str):
        """在主线程中完成 SVG 的加载和初始渲染"""
        renderer = QSvgRenderer(filepath)
        if not renderer.isValid():
            QMessageBox.critical(self, "加载失败", "无法解析 SVG 文件")
            return

        default_size = renderer.defaultSize()
        if default_size.isEmpty():
            # 某些 SVG 没有设置默认尺寸，给一个合理默认值
            default_size = QSize(1024, 1024)

        # 初始渲染：按 SVG 默认尺寸生成位图缓存
        img = QImage(default_size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(img)
        # 使用 QRectF 指定渲染区域，确保无默认尺寸的 SVG 也能正确适配画布
        renderer.render(painter, QRectF(0, 0, default_size.width(), default_size.height()))
        painter.end()

        # 为 minimap 生成一个缩略用的静态位图
        thumb_pm = QPixmap.fromImage(img)

        self._viewer.set_svg(renderer, default_size)
        self._minimap.set_image(thumb_pm)
        self._reposition_minimap()

    def _render_pdf_page(self, page_idx: int):
        """渲染 PDF 指定页面"""
        if not self._pdf_doc:
            return
        page = self._pdf_doc.load_page(page_idx)
        # 使用2倍缩放获得清晰图片
        mat = None
        fitz = _try_import_fitz()
        if fitz:
            mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)

        # fitz Pixmap → QPixmap
        if pix.alpha:
            fmt = QImage.Format.Format_RGBA8888
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
        else:
            fmt = QImage.Format.Format_RGB888
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)

        pm = QPixmap.fromImage(img.copy())
        self._viewer.set_pixmap(pm)
        self._minimap.set_image(pm)
        self._reposition_minimap()

    def _on_pdf_page_changed(self, page_idx: int):
        """PDF翻页时同步渲染（翻页通常很快，保持同步即可）"""
        self._render_pdf_page(page_idx)

    # =====================================================================
    #  文件信息收集与显示
    # =====================================================================
    def _show_file_info(self, filepath: str, ext: str):
        info = {}
        info["file_name"] = os.path.basename(filepath)
        info["file_path"] = filepath

        try:
            fsize = os.path.getsize(filepath)
            info["file_size"] = self._fmt_size(fsize)
        except OSError:
            pass

        try:
            mtime = os.path.getmtime(filepath)
            info["modified_time"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            pass

        info["format"] = ext.upper().lstrip(".")

        # Pillow 图片信息
        if self._pil_image:
            img = self._pil_image
            info["dimensions"] = f"{img.width} × {img.height} px"
            info["color_mode"] = img.mode

            # 位深度
            mode_bits = {
                "1": "1-bit", "L": "8-bit", "P": "8-bit",
                "RGB": "24-bit", "RGBA": "32-bit",
                "CMYK": "32-bit", "I": "32-bit", "F": "32-bit",
                "LA": "16-bit", "PA": "16-bit",
                "I;16": "16-bit", "I;16L": "16-bit", "I;16B": "16-bit",
            }
            info["bit_depth"] = mode_bits.get(img.mode, f"{img.mode}")

            # DPI
            dpi = img.info.get("dpi")
            if dpi:
                info["dpi"] = f"{dpi[0]:.0f} × {dpi[1]:.0f}"

            # GIF 帧数
            if ext == ".gif":
                try:
                    n_frames = getattr(img, "n_frames", 1)
                    info["frames"] = str(n_frames)
                except Exception:
                    pass

        # PSD 图层数
        if ext in PSD_EXTS and hasattr(self, "_psd_layer_count"):
            info["layers"] = str(self._psd_layer_count)

        # PDF 页数
        if ext in PDF_EXTS and self._pdf_doc:
            info["pdf_pages"] = str(len(self._pdf_doc))
            # PDF 也获取当前页尺寸
            page = self._pdf_doc.load_page(0)
            rect = page.rect
            info["dimensions"] = f"{rect.width:.0f} × {rect.height:.0f} pt"

        # SVG 矢量信息
        if ext in SVG_EXTS:
            renderer = self._viewer._svg_renderer
            if renderer and renderer.isValid():
                ds = renderer.defaultSize()
                info["dimensions"] = f"{ds.width()} × {ds.height()} (矢量)"
                info["color_mode"] = "矢量图形"
                info["bit_depth"] = "无限缩放"

        # 对于非 Pillow 加载的情况（PDF 等），尝试从 viewer 获取像素尺寸
        if "dimensions" not in info:
            pm = self._viewer.current_pixmap()
            if pm and not pm.isNull():
                info["dimensions"] = f"{pm.width()} × {pm.height()} px"

        self._info_panel.set_info(info)

    # =====================================================================
    #  辅助方法
    # =====================================================================
    @staticmethod
    def _fmt_size(nbytes: int) -> str:
        if nbytes < 1024:
            return f"{nbytes} B"
        elif nbytes < 1024 ** 2:
            return f"{nbytes / 1024:.1f} KB"
        elif nbytes < 1024 ** 3:
            return f"{nbytes / 1024 ** 2:.1f} MB"
        else:
            return f"{nbytes / 1024 ** 3:.2f} GB"

    def _update_zoom_label(self):
        scale = self._viewer.get_scale()
        self._zoom_label.setText(f"{scale * 100:.0f}%")

    def _reposition_minimap(self):
        """将缩略图定位到 viewer 右下角，同时更新进度条和文件夹箭头位置"""
        vw = self._viewer.width()
        vh = self._viewer.height()

        # 进度条始终跟随 viewer 尺寸居中
        if self._loading_overlay.isVisible():
            self._loading_overlay.reposition(vw, vh)

        # 文件夹浮动箭头跟随容器尺寸重定位
        if self._folder_arrows.is_active():
            self._folder_arrows.reposition()

        if not self._minimap.isVisible():
            return
        margin = 8
        mw = self._minimap.width()
        mh = self._minimap.height()
        self._minimap.move(vw - mw - margin, vh - mh - margin)
        self._minimap.raise_()

    def _toggle_eyedropper(self, checked: bool):
        """切换吸管取色模式"""
        self._viewer.set_eyedropper_mode(checked)

    def keyPressEvent(self, event):
        """ESC 键退出吸管取色模式；左右方向键切换文件夹中的上/下一张"""
        if event.key() == Qt.Key.Key_Escape and self._btn_eyedropper.isChecked():
            self._btn_eyedropper.setChecked(False)  # 会触发 toggled → _toggle_eyedropper(False)
            return
        # 文件夹浏览模式下，左右方向键翻页
        if self._folder_files and self._folder_arrows.is_active():
            if event.key() == Qt.Key.Key_Left:
                self._folder_arrows.go_prev()
                return
            elif event.key() == Qt.Key.Key_Right:
                self._folder_arrows.go_next()
                return
        super().keyPressEvent(event)

    def _on_transfer_to_texture(self):
        """将当前预览的图片以 PNG 格式转移到贴图修改 tab"""
        # 优先使用 PIL Image（保留原始精度）
        pil_img = self._pil_image
        if pil_img is None:
            # 没有 PIL Image 时从 QPixmap 转换
            pm = self._viewer.current_pixmap()
            if pm is None or pm.isNull():
                QMessageBox.warning(self, "无图片", "当前没有可转移的图片，请先打开一张图片。")
                return
            # QPixmap → PIL Image
            qimg = pm.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            # PySide6 的 bits() 返回 memoryview
            arr = bytes(ptr)
            pil_img = Image.frombytes("RGBA", (w, h), arr, "raw", "RGBA")

        # 保存为临时 PNG 文件
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="viewer_transfer_", delete=False)
            tmp_path = tmp.name
            tmp.close()
            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")
            pil_img.save(tmp_path, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "转移失败", f"保存临时 PNG 文件时出错:\n{e}")
            return

        # 发出信号，由主窗口负责加载到贴图修改 tab 并切换
        self.transfer_to_texture.emit(tmp_path)

    # =====================================================================
    #  打开文件夹 & 文件夹导航
    # =====================================================================
    def _on_open_folder(self):
        """打开文件夹，扫描所有支持的图片文件并加载第一张"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return

        # 扫描文件夹中所有支持格式的文件（不递归子目录）
        supported = []
        try:
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in ALL_SUPPORTED_EXTS:
                    supported.append(fpath)
        except OSError as e:
            QMessageBox.warning(self, "文件夹错误", f"无法读取文件夹:\n{e}")
            return

        if not supported:
            QMessageBox.information(self, "无支持的文件",
                                   f"文件夹中没有找到支持的图片文件。\n\n"
                                   f"支持的格式：{', '.join(sorted(ALL_SUPPORTED_EXTS))}")
            return

        self._folder_files = supported
        self._folder_index = 0
        # 先加载第一张（_load_file 会调用 _cleanup，但不会清理 _folder_arrows）
        self._load_file(supported[0])
        # 加载启动后再 setup 箭头（确保不被 _cleanup 清掉）
        self._folder_arrows.setup(len(supported), 0)

    def _on_folder_index_changed(self, idx: int):
        """文件夹导航栏切换时加载对应文件"""
        if 0 <= idx < len(self._folder_files):
            self._folder_index = idx
            self._load_file(self._folder_files[idx])

    def _on_minimap_navigate(self, img_x: float, img_y: float):
        """缩略图导航：将主视图中心对准图片坐标 (img_x, img_y)"""
        scale = self._viewer.get_scale()
        vw = self._viewer.width()
        vh = self._viewer.height()
        # 计算新的偏移量，使 (img_x, img_y) 对应视口中心
        new_offset = QPointF(
            vw / 2 - img_x * scale,
            vh / 2 - img_y * scale,
        )
        self._viewer.set_offset(new_offset)
