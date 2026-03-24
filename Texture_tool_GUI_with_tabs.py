import sys
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageEnhance, ImageOps

from PySide6.QtCore import Qt, QRect, QPoint, QRegularExpression, QSize, QThread, Signal
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QPen,
    QColor,
    QRegularExpressionValidator,
    QFontMetrics,
)

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QGroupBox,
    QMessageBox,
    QComboBox,
    QLineEdit,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QSpinBox,
    QTabWidget,
    QTabBar,
    QStyleOptionTab,
    QStyle,
    QProgressDialog,
)



# ========= 兼容：不同 PySide6 版本 QStylePainter 所在模块不同 =========
try:
    from PySide6.QtWidgets import QStylePainter
except Exception:
    from PySide6.QtGui import QStylePainter

from sprite_sheet_tab import SpriteSheetTab
from flowmap_tab import FlowMapTab
from growth_gray_tab import GrowthGrayTab
from image_viewer_tab import ImageViewerTab
from version import __version__

# =========================
# 规则：输入框仅允许 A-Z a-z 0-9 _ （导出名由输入框保证）
# =========================
VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp"]


# =========================
# 图像/UI 工具函数
# =========================
def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


def to_bw_rgba(img_rgba: Image.Image) -> Image.Image:
    base = img_rgba.convert("RGBA")
    gray = ImageOps.grayscale(base)
    alpha = base.split()[-1]
    return Image.merge("RGBA", (gray, gray, gray, alpha))


@dataclass
class PixRect:
    x: int
    y: int
    w: int
    h: int

    def to_qrect(self) -> QRect:
        return QRect(self.x, self.y, self.w, self.h)


class CropCanvas(QLabel):
    """稳定版裁切：拖拽创建；框内拖动移动；四角缩放；带遮罩与镂空。"""

    HANDLE = 10
    MIN_SIZE = 12

    def __init__(self, pil_img: Image.Image):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#2a2a38;border-radius:10px;")
        self.setMouseTracking(True)

        self.pil_img = pil_img.convert("RGBA")
        self.img_w, self.img_h = self.pil_img.size

        self._pixmap_scaled: Optional[QPixmap] = None
        self._pix_rect: Optional[QRect] = None  # scaled pixmap 在 label 内的实际区域

        self.sel_rect: Optional[QRect] = None
        self.drag_mode: Optional[str] = None  # new/move/resize
        self.drag_start = QPoint()
        self.sel_start = QRect()
        self.resize_handle: Optional[str] = None

        self._render()

    def _render(self):
        pix = pil_to_qpixmap(self.pil_img)
        self._pixmap_scaled = pix.scaled(
            max(1, self.width()),
            max(1, self.height()),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(self._pixmap_scaled)

        # 计算 pixmap 在 label 中的居中位置
        lw, lh = self.width(), self.height()
        pw, ph = self._pixmap_scaled.width(), self._pixmap_scaled.height()
        x = int((lw - pw) / 2)
        y = int((lh - ph) / 2)
        self._pix_rect = QRect(x, y, pw, ph)
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _clamp_to_pix(self, r: QRect) -> QRect:
        if not self._pix_rect:
            return r
        rr = r.normalized()

        # 限制在 pix_rect 内
        if rr.left() < self._pix_rect.left():
            rr.moveLeft(self._pix_rect.left())
        if rr.top() < self._pix_rect.top():
            rr.moveTop(self._pix_rect.top())
        if rr.right() > self._pix_rect.right():
            rr.moveRight(self._pix_rect.right())
        if rr.bottom() > self._pix_rect.bottom():
            rr.moveBottom(self._pix_rect.bottom())

        return rr

    def _handle_rects(self) -> dict:
        if self.sel_rect is None:
            return {}
        r = self.sel_rect.normalized()
        s = self.HANDLE
        half = s // 2
        pts = {
            "nw": QPoint(r.left(), r.top()),
            "ne": QPoint(r.right(), r.top()),
            "se": QPoint(r.right(), r.bottom()),
            "sw": QPoint(r.left(), r.bottom()),
        }
        return {k: QRect(p.x() - half, p.y() - half, s, s) for k, p in pts.items()}

    def _hit_handle(self, pos: QPoint) -> Optional[str]:
        for k, hr in self._handle_rects().items():
            if hr.contains(pos):
                return k
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or not self._pix_rect:
            return

        pos = event.pos()
        if not self._pix_rect.contains(pos):
            return

        self.drag_start = pos

        h = self._hit_handle(pos)
        if h and self.sel_rect:
            self.drag_mode = "resize"
            self.resize_handle = h
            self.sel_start = QRect(self.sel_rect)
            return

        if self.sel_rect and self.sel_rect.contains(pos):
            self.drag_mode = "move"
            self.sel_start = QRect(self.sel_rect)
            return

        self.drag_mode = "new"
        self.sel_rect = QRect(pos, pos)
        self.update()

    def mouseMoveEvent(self, event):
        if not self._pix_rect:
            return

        pos = event.pos()

        if self.drag_mode is None:
            # cursor 提示
            if self._hit_handle(pos):
                self.setCursor(Qt.SizeFDiagCursor)
            elif self.sel_rect and self.sel_rect.contains(pos):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self.drag_mode == "new":
            end = QPoint(
                max(self._pix_rect.left(), min(self._pix_rect.right(), pos.x())),
                max(self._pix_rect.top(), min(self._pix_rect.bottom(), pos.y())),
            )
            self.sel_rect = self._clamp_to_pix(QRect(self.drag_start, end))
            self.update()
            return

        if not self.sel_rect:
            return

        if self.drag_mode == "move":
            delta = pos - self.drag_start
            r = QRect(self.sel_start)
            r.translate(delta)
            self.sel_rect = self._clamp_to_pix(r)
            self.update()
            return

        if self.drag_mode == "resize":
            r0 = QRect(self.sel_start).normalized()
            px = max(self._pix_rect.left(), min(self._pix_rect.right(), pos.x()))
            py = max(self._pix_rect.top(), min(self._pix_rect.bottom(), pos.y()))

            left, top, right, bottom = r0.left(), r0.top(), r0.right(), r0.bottom()
            h = self.resize_handle

            if h == "nw":
                left, top = px, py
            elif h == "ne":
                right, top = px, py
            elif h == "se":
                right, bottom = px, py
            elif h == "sw":
                left, bottom = px, py

            r = QRect(QPoint(left, top), QPoint(right, bottom)).normalized()
            r = self._clamp_to_pix(r)

            if r.width() < self.MIN_SIZE:
                r.setRight(r.left() + self.MIN_SIZE)
            if r.height() < self.MIN_SIZE:
                r.setBottom(r.top() + self.MIN_SIZE)

            self.sel_rect = self._clamp_to_pix(r)
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag_mode = None
        self.resize_handle = None
        self.setCursor(Qt.ArrowCursor)
        if self.sel_rect:
            self.sel_rect = self.sel_rect.normalized()
            self.update()

    def paintEvent(self, e):
        # 先绘制框内底色（在图片下方，不会遮挡图片）
        if self._pix_rect:
            p0 = QPainter(self)
            p0.setPen(Qt.NoPen)
            p0.setBrush(QColor(58, 58, 74))  # #3a3a4a — 框内较亮底色
            p0.drawRect(self._pix_rect)
            p0.end()

        # QLabel 绘制图片（在底色之上）
        super().paintEvent(e)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # 图片边缘指引线（比框内底色再亮一档）
        if self._pix_rect:
            border_pen = QPen(QColor(90, 90, 106))  # #5a5a6a
            border_pen.setWidth(1)
            p.setPen(border_pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(self._pix_rect)

        if not self.sel_rect:
            p.end()
            return
        from PySide6.QtGui import QPainterPath

        path = QPainterPath()
        path.addRect(self.rect())
        path.addRect(self.sel_rect.normalized())
        path.setFillRule(Qt.OddEvenFill)

        # 半透明遮罩（PS 风格）
        p.fillPath(path, QColor(0, 0, 0, 80))

        # 选区边框
        pen = QPen(QColor(255, 255, 255, 230))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(self.sel_rect)

        # 四角 handles
        for hr in self._handle_rects().values():
            p.fillRect(hr, QColor(255, 255, 255, 230))

        p.end()

    def rotate_image(self, clockwise: bool = True):
        """旋转图片90度，clockwise=True顺时针，False逆时针。"""
        angle = -90 if clockwise else 90
        self.pil_img = self.pil_img.rotate(angle, expand=True)
        self.img_w, self.img_h = self.pil_img.size
        # 清除裁切选区
        self.sel_rect = None
        self._render()

    def get_cropped_image(self) -> Optional[Image.Image]:
        if not self.sel_rect or not self._pix_rect or not self._pixmap_scaled:
            return None

        r = self.sel_rect.normalized()

        # 转为相对 pix_rect 的坐标
        rx = r.left() - self._pix_rect.left()
        ry = r.top() - self._pix_rect.top()
        rw = r.width()
        rh = r.height()

        pw = self._pix_rect.width()
        ph = self._pix_rect.height()

        # 映射到原图
        l = int(rx / pw * self.img_w)
        t = int(ry / ph * self.img_h)
        rr = int((rx + rw) / pw * self.img_w)
        bb = int((ry + rh) / ph * self.img_h)

        l = max(0, min(self.img_w - 1, l))
        t = max(0, min(self.img_h - 1, t))
        rr = max(l + 1, min(self.img_w, rr))
        bb = max(t + 1, min(self.img_h, bb))

        if rr - l < 2 or bb - t < 2:
            return None

        return self.pil_img.crop((l, t, rr, bb))


class MaskThresholdDialog(QDialog):
    """亮度阈值遮罩对话框：基于当前显示画面，通过亮度阈值识别主体。"""

    def __init__(self, pil_img: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("遮罩生成 — 亮度阈值")
        self.resize(800, 640)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #a6adc8; }
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 7px;
                padding: 6px 18px; font-size: 13px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
            QSlider::groove:horizontal {
                height: 6px; background: #313244; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa; width: 16px; height: 16px;
                margin: -5px 0; border-radius: 8px;
            }
            QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
        """)

        self.source_img = pil_img.convert("RGBA")
        self.result_img: Optional[Image.Image] = None

        # 预览区域
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 440)
        self.preview_label.setStyleSheet("background-color: #11111b; border: 1px solid #45475a;")

        # 阈值滑块行
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("亮度阈值："))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(128)
        self.thresh_label = QLabel("128")
        self.thresh_label.setFixedWidth(36)
        self.thresh_label.setAlignment(Qt.AlignCenter)
        thresh_row.addWidget(self.thresh_slider, 1)
        thresh_row.addWidget(self.thresh_label)

        tips = QLabel("提示：亮度 ≥ 阈值的像素将被识别为主体（白色），其余部分变为透明。\n"
                      "此计算基于当前显示画面（已应用亮度/对比度调整）。")
        tips.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tips)
        layout.addWidget(self.preview_label, 1)
        layout.addLayout(thresh_row)
        layout.addWidget(btns)

        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        # 初始预览
        self._update_preview()

    def _compute_mask(self, thresh: int) -> Image.Image:
        """根据亮度阈值计算遮罩图：主体白色，背景透明。"""
        img = self.source_img
        r, g, b, a = img.split()
        # 逐像素计算亮度：L = 0.299R + 0.587G + 0.114B
        # 使用 PIL 的 merge + point 实现，无需 numpy
        lum = Image.merge("RGB", (r, g, b)).convert("L")
        # 亮度 >= 阈值 => 主体（白色不透明），否则 => 背景（透明）
        mask = lum.point(lambda x: 255 if x >= thresh else 0)
        # 生成结果：主体区域为白色，背景为透明
        result = Image.new("RGBA", img.size, (0, 0, 0, 0))
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        result.paste(white, mask=mask)
        return result

    def _on_thresh_changed(self, v: int):
        self.thresh_label.setText(str(v))
        self._update_preview()

    def _update_preview(self):
        thresh = self.thresh_slider.value()
        preview = self._compute_mask(thresh)
        pix = pil_to_qpixmap(preview)
        pix = pix.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    def _on_ok(self):
        thresh = self.thresh_slider.value()
        self.result_img = self._compute_mask(thresh)
        self.accept()


class CropDialog(QDialog):

    def __init__(self, pil_img: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("裁切/旋转")
        self.resize(980, 720)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #a6adc8; }
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 7px;
                padding: 6px 18px; font-size: 13px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
        """)

        self.canvas = CropCanvas(pil_img)
        self.canvas.setMinimumSize(810, 560)

        tips = QLabel("拖拽创建裁切框；框内拖动移动；拖动角点缩放；确定后应用。")
        tips.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)

        # ---- 左侧工具栏 ----
        toolbar = QWidget()
        toolbar.setFixedWidth(42)
        toolbar.setStyleSheet("background: #252536; border-radius: 8px;")
        tb_layout = QVBoxLayout(toolbar)
        tb_layout.setContentsMargins(4, 8, 4, 8)
        tb_layout.setSpacing(6)

        btn_ccw = QPushButton("↺")
        btn_ccw.setToolTip("逆时针旋转 90°")
        btn_ccw.setFixedSize(34, 34)
        btn_ccw.setStyleSheet("""
            QPushButton { font-size: 18px; padding: 0; border-radius: 6px;
                          background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            QPushButton:hover { background: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
        """)
        btn_ccw.clicked.connect(lambda: self.canvas.rotate_image(clockwise=False))

        btn_cw = QPushButton("↻")
        btn_cw.setToolTip("顺时针旋转 90°")
        btn_cw.setFixedSize(34, 34)
        btn_cw.setStyleSheet("""
            QPushButton { font-size: 18px; padding: 0; border-radius: 6px;
                          background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            QPushButton:hover { background: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
        """)
        btn_cw.clicked.connect(lambda: self.canvas.rotate_image(clockwise=True))

        tb_layout.addWidget(btn_ccw)
        tb_layout.addWidget(btn_cw)
        tb_layout.addStretch()

        # ---- 中间区域（工具栏 + 画布）水平排列 ----
        center_layout = QHBoxLayout()
        center_layout.setSpacing(6)
        center_layout.addWidget(toolbar)
        center_layout.addWidget(self.canvas, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(tips)
        layout.addLayout(center_layout, 1)
        layout.addWidget(btns)

        self.result_img: Optional[Image.Image] = None

    def _on_ok(self):
        cropped = self.canvas.get_cropped_image()
        if cropped is not None:
            # 有裁切框：应用裁切
            self.result_img = cropped.convert("RGBA")
        else:
            # 无裁切框：直接使用当前图片（可能已旋转）
            self.result_img = self.canvas.pil_img.convert("RGBA")
        self.accept()


class DropLabel(QLabel):
    def __init__(self, on_drop_callback):
        super().__init__()
        self.on_drop_callback = on_drop_callback
        self._parent_window = None  # 用于弹出文件对话框
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("拖拽图片到这里\n或点击此区域导入")
        self.setStyleSheet(
            "border:2px dashed #45475a;border-radius:10px;padding:20px;"
            "background:transparent;color:#6c7086;font-size:13px;"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            parent = self._parent_window or self
            path, _ = QFileDialog.getOpenFileName(
                parent, "选择图片", "", "Images (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
            )
            if path:
                self.on_drop_callback(path)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.on_drop_callback(path)


class CheckerLabel(QLabel):
    """带棋盘格透明背景的预览Label，hover时显示图片边界线"""
    def __init__(self, cell=12, color1=None, color2=None, parent=None):
        super().__init__(parent)
        self.cell = cell
        self.color1 = color1 or QColor(42, 42, 58)
        self.color2 = color2 or QColor(30, 30, 46)
        self._hovered = False
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def _pixmap_rect(self):
        """计算当前pixmap在label中居中显示的实际矩形区域"""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        # label的内容区域（去除margin/border）
        cr = self.contentsRect()
        # pixmap实际尺寸
        pw, ph = pm.width(), pm.height()
        # 居中偏移
        x = cr.x() + (cr.width() - pw) // 2
        y = cr.y() + (cr.height() - ph) // 2
        return QRect(x, y, pw, ph)

    def paintEvent(self, event):
        painter = QPainter(self)
        cell = self.cell
        cols = self.width() // cell + 1
        rows = self.height() // cell + 1
        for r in range(rows):
            for c in range(cols):
                color = self.color1 if (r + c) % 2 == 0 else self.color2
                painter.fillRect(c * cell, r * cell, cell, cell, color)
        painter.end()
        super().paintEvent(event)

        # hover时绘制图片边界指引线
        if self._hovered:
            pix_rect = self._pixmap_rect()
            if pix_rect:
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing, False)
                border_pen = QPen(QColor(90, 90, 106))  # #5a5a6a — 与crop一致的边界线颜色
                border_pen.setWidth(1)
                p.setPen(border_pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(pix_rect)
                p.end()


# =========================
# 自定义左侧 TabBar：文字竖排堆叠显示（上下排列）
# =========================
class StackedTextTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDrawBase(False)

    def tabSizeHint(self, index):
        text = self.tabText(index) or ""
        fm = QFontMetrics(self.font())
        lines = text.split("\n")
        max_w = 0
        for ln in lines:
            max_w = max(max_w, fm.horizontalAdvance(ln))
        h = fm.height() * max(1, len(lines)) + 18
        w = max(44, max_w + 18)
        return QSize(w, max(120, h))

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)

            r = opt.rect.adjusted(6, 6, -6, -6)
            painter.save()
            painter.drawText(r, Qt.AlignCenter, self.tabText(i))
            painter.restore()


class MainWindow(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("PPEditor")
        # 设置窗口左上角图标（兼容打包环境，多路径回退查找）
        _ico_name = "TextureToolGUI.ico"
        _candidates = []
        if getattr(sys, 'frozen', False):
            _candidates.append(os.path.join(os.path.dirname(sys.executable), _ico_name))
            _candidates.append(os.path.join(getattr(sys, '_MEIPASS', ''), _ico_name))
        else:
            _candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), _ico_name))
        _ico_path = next((p for p in _candidates if os.path.exists(p)), None)
        if _ico_path:
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(_ico_path))
        self.resize(1180, 740)

        _base_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        _style = """
            /* ===== 全局背景 ===== */
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
                font-size: 13px;
            }

            /* ===== GroupBox ===== */
            QGroupBox {
                background-color: #252535;
                border: 1px solid #383850;
                border-radius: 10px;
                margin-top: 18px;
                padding-top: 14px;
                padding-bottom: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 700;
                color: #89b4fa;
                background-color: #252535;
                border-radius: 4px;
            }

            /* ===== 按钮 ===== */
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 7px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-color: #89b4fa;
            }
            QPushButton:disabled {
                background-color: #2a2a3a;
                color: #585b70;
                border-color: #383850;
            }

            /* ===== 输入框 ===== */
            QLineEdit {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 5px 8px;
                selection-background-color: #89b4fa;
                selection-color: #1e1e2e;
            }
            QLineEdit:focus {
                border-color: #89b4fa;
            }

            /* ===== 下拉框 ===== */
            QComboBox {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px 30px 4px 8px;
                min-height: 26px;
            }
            QComboBox:hover {
                border-color: #89b4fa;
            }
            QComboBox:disabled {
                background-color: #252535;
                color: #585b70;
                border-color: #383850;
            }
            QComboBox::drop-down {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                background-color: transparent;
            }
            QComboBox::drop-down:hover {
                background-color: transparent;
            }
            QComboBox::down-arrow {
                image: url("__COMBO_DN_ARROW__");
                width: 10px;
                height: 6px;
            }
            QComboBox::down-arrow:disabled {
                opacity: 0.3;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                selection-background-color: #89b4fa;
                selection-color: #1e1e2e;
                outline: none;
                padding: 2px;
            }

            /* ===== SpinBox ===== */
            QSpinBox {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px 22px 4px 8px;
                min-height: 26px;
            }
            QSpinBox:focus {
                border-color: #89b4fa;
            }
            QSpinBox:disabled {
                background-color: #252535;
                color: #585b70;
            }
            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                background-color: transparent;
                border: none;
                width: 20px;
                height: 12px;
            }
            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                background-color: transparent;
                border: none;
                width: 20px;
                height: 12px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: transparent;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background-color: transparent;
            }
            QSpinBox::up-arrow {
                image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAMCAYAAABiDJ37AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAA7ElEQVQokeWRz0oCURSHf+fWDZwWhTthnkK5vYC4kXwBoY0vMQrRMBDoddErCC58ANNhFOoFfA33tbrBPdOcdpLQjLStb3s+fucf8O+gU0Ici6oZ3weAj93FIkmoqPJVVXG69M3AcEakGkSqERjOpkvf/PWET5nU85wfIbiSszyKusEeAGzqQvo8t0J4L6DvR7f0VhkYx6Iub3gggrtCUTLq6tefGtpnboPkAcDc7fTs+xkOK49XvhUY3hZC187pTlkYAEQ9/eKc7oCoXjO8Ga9860gQEZqseWhTF5aFlGFTF07WPBSRkw/+o3wBC2hZRH08FNEAAAAASUVORK5CYII=");
                width: 10px;
                height: 6px;
            }
            QSpinBox::down-arrow {
                image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAMCAYAAABiDJ37AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAA5UlEQVQokeWPMUoDURRF7/vM/OJ3WmcTgjuIEAizgSzEOFPIECycb+EWUgpphxkCfwkxuAlLid0E8uK/VmMhicE6p32cw33AeUJSqkZz33aD/7q+7QZVozlJAQADACJCkoExmftGb8sF7alQuaCtap0yJnOSQUT4EwSAIrPr7Sodgdw4p8HXOjy6qtahcxqM8GO7SkdFZtf9TQ4Jz0te6l4fAFzA7Kd3Y/fev4eYPAmx+ZL0vsjk87d7MNjz2O6uTISHSAAAIt5QJC/G9u2Y82cQAMqSxl3vJgDQvdqX2UziKefM+Abk12Ee9AJi7AAAAABJRU5ErkJggg==");
                width: 10px;
                height: 6px;
            }
            QSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled {
                opacity: 0.3;
            }

            /* ===== 滑块 ===== */
            QSlider::groove:horizontal {
                height: 4px;
                background: #45475a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                border: 2px solid #1e1e2e;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #b4befe;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border-radius: 2px;
            }

            /* ===== 复选框 ===== */
            QCheckBox {
                spacing: 8px;
                color: #cdd6f4;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #45475a;
                border-radius: 4px;
                background-color: #181825;
            }
            QCheckBox::indicator:checked {
                background-color: #89b4fa;
                border-color: #89b4fa;
            }
            QCheckBox::indicator:hover {
                border-color: #89b4fa;
            }

            /* ===== Label ===== */
            QLabel {
                color: #cdd6f4;
                background: transparent;
            }

            /* ===== 列表 ===== */
            QListWidget {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #383850;
                border-radius: 8px;
                outline: none;
                padding: 4px;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-radius: 5px;
            }
            QListWidget::item:selected {
                background-color: #313244;
                color: #89b4fa;
            }
            QListWidget::item:hover {
                background-color: #252535;
            }

            /* ===== 滚动条 ===== */
            QScrollBar:vertical {
                background: #181825;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #45475a;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #89b4fa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: #181825;
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #45475a;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #89b4fa;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }

            /* ===== Splitter ===== */
            QSplitter::handle {
                background-color: #383850;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
            QSplitter::handle:vertical {
                height: 2px;
            }

            /* ===== TabWidget / TabBar ===== */
            QTabWidget::pane {
                border: 1px solid #383850;
                border-radius: 8px;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #252535;
                color: #6c7086;
                border: 1px solid #383850;
                border-radius: 6px;
                padding: 10px 8px;
                margin: 3px 2px;
                font-size: 12px;
                font-weight: 600;
                min-width: 44px;
            }
            QTabBar::tab:selected {
                background-color: #313244;
                color: #89b4fa;
                border-color: #89b4fa;
            }
            QTabBar::tab:hover:!selected {
                background-color: #2a2a3a;
                color: #cdd6f4;
            }

            /* ===== MessageBox ===== */
            QMessageBox {
                background-color: #1e1e2e;
            }
            QDialogButtonBox QPushButton {
                min-width: 80px;
            }
        """
        # 将 ComboBox 下拉箭头图片保存为临时文件，供样式表引用
        import tempfile, base64 as _b64
        _dn_arrow_b64 = (
            b"iVBORw0KGgoAAAANSUhEUgAAABQAAAAMCAYAAABiDJ37AAAACXBIWXMAAA7EAAAOxAGVKw4b"
            b"AAAA5UlEQVQokeWPMUoDURRF7/vM/OJ3WmcTgjuIEAizgSzEOFPIECycb+EWUgpphxkCfwkx"
            b"uAlLid0E8uK/VmMhicE6p32cw33AeUJSqkZz33aD/7q+7QZVozlJAQADACJCkoExmftGb8sF"
            b"7alQuaCtap0yJnOSQUT4EwSAIrPr7Sodgdw4p8HXOjy6qtahcxqM8GO7SkdFZtf9TQ4Jz0te"
            b"6l4fAFzA7Kd3Y/fev4eYPAmx+ZL0vsjk87d7MNjz2O6uTISHSAAAIt5QJC/G9u2Y82cQAMqS"
            b"xl3vJgDQvdqX2UziKefM+Abk12Ee9AJi7AAAAABJRU5ErkJggg=="
        )
        _tmp_dn = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        _tmp_dn.write(_b64.b64decode(_dn_arrow_b64))
        _tmp_dn.close()
        import atexit as _atexit
        _atexit.register(lambda p=_tmp_dn.name: os.path.exists(p) and os.remove(p))
        _style = _style.replace("__COMBO_DN_ARROW__", _tmp_dn.name.replace("\\", "/"))
        self.setStyleSheet(_style)

        self.src_path: Optional[str] = None
        self.source_color: Optional[Image.Image] = None  # 最初导入（永远不变）
        self.master_color: Optional[Image.Image] = None  # 当前基础（裁切后会替换为裁切结果）

        self.is_bw: bool = False
        self.has_black_bg: bool = False
        self.target_size: Optional[Tuple[int, int]] = None

        self.working_img: Optional[Image.Image] = None
        self.preview_img: Optional[Image.Image] = None

        self.output_basename: Optional[str] = None

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        # 用自定义 TabBar（竖排文字），注意：必须先 setTabBar 再 setShape
        tabs.setTabBar(StackedTextTabBar())
        tabs.tabBar().setShape(QTabBar.RoundedWest)
        self.setCentralWidget(tabs)
        self._tabs = tabs  # 保存引用，供跨 tab 切换使用

        # Tab 1: 贴图修改（原有功能）
        root = QWidget()
        main_layout = QHBoxLayout(root)

        # Tab 2: 精灵图制作
        sprite_tab = SpriteSheetTab()
        # Tab 3: 法线绘制
        flowmap_tab = FlowMapTab()
        # Tab 4: 生长灰度图生成
        growth_tab = GrowthGrayTab()
        # Tab 5: 全能看图
        image_viewer_tab = ImageViewerTab()
        tabs.addTab(root, "贴\n图\n修\n改")
        tabs.addTab(sprite_tab, "精\n灵\n图\n制\n作")
        tabs.addTab(flowmap_tab, "法\n线\n绘\n制")
        tabs.addTab(growth_tab, "灰\n度\n图\n生\n成")
        tabs.addTab(image_viewer_tab, "全\n能\n看\n图")

        # 连接全能看图的「转移至贴图修改」信号
        image_viewer_tab.transfer_to_texture.connect(self._on_transfer_from_viewer)
        # Left
        left_layout = QVBoxLayout()
        self.drop_area = DropLabel(self.load_image)
        self.drop_area._parent_window = self
        self.preview_label = CheckerLabel(cell=12)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border-radius:10px; border:1px solid #383850;")
        self.preview_label.setMinimumSize(680, 560)

        left_layout.addWidget(self.drop_area, 1)
        left_layout.addWidget(self.preview_label, 6)

        # Right
        right_layout = QVBoxLayout()

        # 遮罩生成 + 重置（80%/20%）
        mask_gen_row = QHBoxLayout()
        btn_mask_gen = QPushButton("遮罩生成")
        btn_mask_gen.setStyleSheet("text-align:center;")
        btn_mask_gen.setMinimumWidth(340)
        btn_mask_gen.clicked.connect(self.generate_mask)
        btn_mask_gen.setMinimumHeight(34)

        btn_reset_mask = QPushButton("重置")
        btn_reset_mask.clicked.connect(self.reset_mask)
        btn_reset_mask.setMinimumHeight(34)

        mask_gen_row.addWidget(btn_mask_gen, 4)
        mask_gen_row.addWidget(btn_reset_mask, 1)

        # 裁切 + 重置（80%/20%）
        crop_row = QHBoxLayout()
        btn_crop = QPushButton("裁切/旋转")
        btn_crop.setStyleSheet("text-align:center;")
        btn_crop.setMinimumWidth(340)
        btn_crop.clicked.connect(self.open_crop_dialog)
        btn_crop.setMinimumHeight(34)

        btn_reset_crop = QPushButton("重置")
        btn_reset_crop.clicked.connect(self.reset_crop)
        btn_reset_crop.setMinimumHeight(34)

        crop_row.addWidget(btn_crop, 4)
        crop_row.addWidget(btn_reset_crop, 1)

        # 一键黑白 + 重置（80%/20%）
        bw_row = QHBoxLayout()
        btn_bw = QPushButton("一键黑白")
        btn_bw.setStyleSheet("text-align:center;")
        btn_bw.setMinimumWidth(340)
        btn_bw.clicked.connect(self.apply_black_white)
        btn_bw.setMinimumHeight(34)

        btn_reset_bw = QPushButton("重置")
        btn_reset_bw.clicked.connect(self.cancel_black_white)
        btn_reset_bw.setMinimumHeight(34)

        bw_row.addWidget(btn_bw, 4)
        bw_row.addWidget(btn_reset_bw, 1)

        # 添加黑底 + 重置（80%/20%）
        black_bg_row = QHBoxLayout()
        btn_black_bg = QPushButton("添加黑底")
        btn_black_bg.setMinimumHeight(34)
        btn_black_bg.setStyleSheet("text-align:center;")
        btn_black_bg.clicked.connect(self.apply_black_bg)

        btn_reset_black_bg = QPushButton("重置")
        btn_reset_black_bg.setMinimumHeight(34)
        btn_reset_black_bg.clicked.connect(self.reset_black_bg)

        black_bg_row.addWidget(btn_black_bg, 4)
        black_bg_row.addWidget(btn_reset_black_bg, 1)

        # Brightness/Contrast
        adj_group = QGroupBox("亮度 / 对比度")
        adj_layout = QVBoxLayout(adj_group)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)

        self.brightness_spin = QLineEdit("100%")
        self.brightness_spin.setFixedWidth(90)
        self.brightness_spin.setAlignment(Qt.AlignCenter)

        btn_reset_brightness = QPushButton("重置")
        btn_reset_brightness.setFixedWidth(70)
        btn_reset_brightness.clicked.connect(self.reset_brightness)

        bright_row = QHBoxLayout()
        lbl_b = QLabel("亮度")
        lbl_b.setFixedWidth(60)
        bright_row.addWidget(lbl_b, 0)
        bright_row.addWidget(self.brightness_slider, 1)
        bright_row.addWidget(self.brightness_spin, 0)
        bright_row.addWidget(btn_reset_brightness, 0)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)

        self.contrast_spin = QLineEdit("100%")
        self.contrast_spin.setFixedWidth(90)
        self.contrast_spin.setAlignment(Qt.AlignCenter)

        btn_reset_contrast = QPushButton("重置")
        btn_reset_contrast.setFixedWidth(70)
        btn_reset_contrast.clicked.connect(self.reset_contrast)

        contrast_row = QHBoxLayout()
        lbl_c = QLabel("对比度")
        lbl_c.setFixedWidth(60)
        contrast_row.addWidget(lbl_c, 0)
        contrast_row.addWidget(self.contrast_slider, 1)
        contrast_row.addWidget(self.contrast_spin, 0)
        contrast_row.addWidget(btn_reset_contrast, 0)

        adj_layout.addLayout(bright_row)
        adj_layout.addLayout(contrast_row)

        self.brightness_slider.valueChanged.connect(self._on_brightness_slider)
        self.brightness_spin.editingFinished.connect(self._on_brightness_spin)
        self.contrast_slider.valueChanged.connect(self._on_contrast_slider)
        self.contrast_spin.editingFinished.connect(self._on_contrast_spin)

        # Resize
        resize_group = QGroupBox("一键尺寸")
        resize_layout = QVBoxLayout(resize_group)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.setContentsMargins(0, 4, 0, 4)
        for s in (64, 128, 256):
            b = QPushButton(f"{s}x{s}")
            b.setMinimumWidth(80)
            b.setMinimumHeight(32)
            b.clicked.connect(lambda checked=False, size=s: self.set_size(size, size))
            row1.addWidget(b)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.setContentsMargins(0, 0, 0, 4)
        for s in (512, 1024, 2048):
            b = QPushButton(f"{s}x{s}")
            b.setMinimumWidth(80)
            b.setMinimumHeight(32)
            b.clicked.connect(lambda checked=False, size=s: self.set_size(size, size))
            row2.addWidget(b)

        btn_reset_size = QPushButton("重置尺寸（回到原尺寸）")
        btn_reset_size.clicked.connect(self.reset_size)

        _pow2_sizes = ["32", "64", "128", "256", "512", "1024", "2048"]

        custom_row = QHBoxLayout()
        self.custom_w = QComboBox()
        self.custom_w.setEditable(True)
        self.custom_w.addItems(_pow2_sizes)
        self.custom_w.setCurrentText("1024")
        self.custom_w.setFixedWidth(110)

        self.custom_h = QComboBox()
        self.custom_h.setEditable(True)
        self.custom_h.addItems(_pow2_sizes)
        self.custom_h.setCurrentText("1024")
        self.custom_h.setFixedWidth(110)

        btn_apply_custom = QPushButton("应用")
        btn_apply_custom.clicked.connect(self.apply_custom_size)
        btn_apply_custom.setFixedWidth(70)

        custom_row.addWidget(QLabel("自定义尺寸："), 0)
        custom_row.addWidget(self.custom_w, 0)
        custom_row.addWidget(QLabel("×"), 0)
        custom_row.addWidget(self.custom_h, 0)
        custom_row.addStretch(1)
        custom_row.addWidget(btn_apply_custom, 0)

        resize_layout.addLayout(row1)
        resize_layout.addLayout(row2)
        resize_layout.addWidget(btn_reset_size)
        resize_layout.addLayout(custom_row)

        # Naming
        name_group = QGroupBox("一键命名")
        name_layout = QVBoxLayout(name_group)

        # history buttons row
        self.history_layout = QHBoxLayout()
        self.history_btn1 = QPushButton("")
        self.history_btn2 = QPushButton("")
        self.history_btn3 = QPushButton("")

        for btn in (self.history_btn1, self.history_btn2, self.history_btn3):
            btn.setVisible(False)
            btn.clicked.connect(
                lambda checked=False, b=btn: self.apply_history_name(b.text())
            )
            self.history_layout.addWidget(btn)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入：例如 apple（导出名将变为 T_apple）")
        self.name_input.setValidator(
            QRegularExpressionValidator(QRegularExpression("^[A-Za-z0-9_]*$"))
        )
        self.name_input.textChanged.connect(self.update_name_preview)

        self.name_preview = QLabel("预览：-")
        self.name_preview.setStyleSheet("font-weight:700; color:#a6e3a1; padding:2px 0;")

        self.btn_apply_name = QPushButton("确定/应用命名")
        self.btn_apply_name.clicked.connect(self.apply_naming)

        self.btn_reset_name = QPushButton("重置命名（回到原名）")
        self.btn_reset_name.clicked.connect(self.reset_naming)

        name_layout.addLayout(self.history_layout)
        name_layout.addWidget(self.name_input)
        name_layout.addWidget(self.name_preview)
        name_layout.addWidget(self.btn_apply_name)
        name_layout.addWidget(self.btn_reset_name)

        # Export
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout(export_group)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG"])

        self.chk_overwrite = QCheckBox("覆盖原图（会先备份 .bak）")
        self.chk_overwrite.setChecked(False)

        self.btn_export = QPushButton("导出")
        self.btn_export.setStyleSheet(
            "background:#89b4fa; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        self.btn_export.clicked.connect(self.export_image)

        export_layout.addWidget(QLabel("格式："))
        export_layout.addWidget(self.format_combo)
        export_layout.addWidget(self.chk_overwrite)
        export_layout.addWidget(self.btn_export)

        self.info_label = QLabel("未导入图片")
        self.info_label.setWordWrap(True)

        right_layout.addLayout(mask_gen_row)
        right_layout.addLayout(crop_row)
        right_layout.addLayout(black_bg_row)
        right_layout.addLayout(bw_row)
        right_layout.addWidget(adj_group)
        right_layout.addWidget(resize_group)
        right_layout.addWidget(name_group)
        right_layout.addWidget(export_group)
        right_layout.addWidget(self.info_label)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, 13)
        main_layout.addLayout(right_layout, 12)

        self.set_enabled(False)

        # 启动时加载历史
        last = self.load_last_name()
        if last:
            self.build_history_buttons(last)

        # 启动参数自动导入
        if initial_path:
            self.load_image(initial_path)

        # ── 后台检查更新 ──
        self._start_update_checker()

    # ---------------- 在线更新 ----------------
    def _start_update_checker(self):
        """在后台线程中检查 GitHub Release 是否有新版本"""
        class _UpdateThread(QThread):
            update_found = Signal(dict)
            def run(self_t):
                try:
                    from updater import check_for_update
                    result = check_for_update()
                    if result:
                        self_t.update_found.emit(result)
                except Exception:
                    pass  # 静默失败

        self._update_thread = _UpdateThread(self)
        self._update_thread.update_found.connect(self._on_update_found)
        self._update_thread.start()

    def _on_update_found(self, info: dict):
        """收到新版本信息后弹窗提示用户"""
        changelog = info.get("changelog", "暂无更新说明")
        if len(changelog) > 600:
            changelog = changelog[:600] + "\n..."

        msg = QMessageBox(self)
        msg.setWindowTitle("发现新版本")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            f"发现新版本 v{info['version']}\n"
            f"当前版本 v{__version__}\n\n"
            f"更新内容：\n{changelog}"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.button(QMessageBox.Yes).setText("立即更新")
        msg.button(QMessageBox.No).setText("稍后提醒")

        if msg.exec() == QMessageBox.Yes:
            self._do_update(info["download_url"])

    def _do_update(self, download_url: str):
        """执行下载和更新流程，带完整进度条（下载 0~50%，应用更新 50~100%）"""
        import threading

        progress = QProgressDialog("准备更新...", "取消", 0, 100, self)
        progress.setWindowTitle("更新中")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # 用 threading.Event 作为取消信号，后台线程会检查它
        self._stop_event = threading.Event()

        def on_cancel():
            self._stop_event.set()
        progress.canceled.connect(on_cancel)

        # ── 后台线程：负责下载 + 应用更新 ──
        class _UpdateWorkerThread(QThread):
            # 下载进度信号 (percent 0-100, 仅下载部分)
            download_progress = Signal(int)
            # 应用更新进度信号 (percent 0-100, stage 描述)
            apply_progress = Signal(int, str)
            # 下载完成信号
            download_done = Signal()
            # 全部完成信号，携带新版本 exe 路径
            finished_ok = Signal(str)
            # 失败信号
            finished_err = Signal(str)
            # 用户取消信号
            cancelled = Signal()

            def __init__(self_t, url, stop_event):
                super().__init__()
                self_t.url = url
                self_t.stop_event = stop_event

            def run(self_t):
                try:
                    # 阶段一：下载（支持取消中断）
                    from updater import download_update, UpdateCancelledError
                    try:
                        zip_path = download_update(
                            self_t.url,
                            progress_callback=self_t.download_progress.emit,
                            stop_event=self_t.stop_event,
                        )
                    except UpdateCancelledError:
                        self_t.cancelled.emit()
                        return

                    self_t.download_done.emit()

                    # 阶段二：应用更新（解压、备份、安装——不可取消）
                    from updater import apply_update
                    result = apply_update(
                        zip_path,
                        progress_callback=self_t.apply_progress.emit
                    )
                    if result and isinstance(result, str):
                        # result 是新版本 exe 的路径
                        self_t.finished_ok.emit(result)
                    else:
                        self_t.finished_err.emit("应用更新失败")
                except Exception as e:
                    self_t.finished_err.emit(str(e))

        def on_download_progress(percent):
            """下载阶段：映射到总进度 0% ~ 50%"""
            if not self._stop_event.is_set():
                total_pct = int(percent * 0.5)
                progress.setValue(total_pct)
                progress.setLabelText(f"正在下载更新... {percent}%")
                QApplication.processEvents()

        def on_download_done():
            """下载完成，切换到应用更新阶段"""
            if not self._stop_event.is_set():
                progress.setValue(50)
                progress.setLabelText("下载完成，准备安装...")
                # 下载完成后禁用取消按钮（应用更新不可中断）
                progress.setCancelButton(None)
                QApplication.processEvents()

        def on_apply_progress(percent, stage):
            """应用更新阶段：映射到总进度 50% ~ 100%"""
            total_pct = 50 + int(percent * 0.5)
            progress.setValue(total_pct)
            progress.setLabelText(stage)
            QApplication.processEvents()

        def on_finished_ok(new_exe_path):
            """更新完成，在主线程中启动新版本并退出"""
            progress.setValue(100)
            progress.setLabelText("更新完成，正在重启...")
            QApplication.processEvents()
            # 在主线程中安全地启动新版本并退出
            import subprocess
            from updater import get_app_dir
            subprocess.Popen([new_exe_path], cwd=get_app_dir())
            QApplication.quit()

        def on_finished_err(err_msg):
            progress.close()
            if not self._stop_event.is_set():
                QMessageBox.warning(
                    self, "更新失败",
                    f"更新时出错：\n{err_msg}\n\n"
                    "程序将继续使用当前版本。\n"
                    "你可以稍后手动从 GitHub Release 页面下载最新版本。"
                )

        def on_cancelled():
            """用户取消了下载，关闭进度条"""
            progress.close()

        self._dl_thread = _UpdateWorkerThread(download_url, self._stop_event)
        self._dl_thread.download_progress.connect(on_download_progress)
        self._dl_thread.download_done.connect(on_download_done)
        self._dl_thread.apply_progress.connect(on_apply_progress)
        self._dl_thread.finished_ok.connect(on_finished_ok)
        self._dl_thread.finished_err.connect(on_finished_err)
        self._dl_thread.cancelled.connect(on_cancelled)
        self._dl_thread.start()

    # ---------------- UI enable ----------------
    def set_enabled(self, enabled: bool):
        self.brightness_slider.setEnabled(enabled)
        self.contrast_slider.setEnabled(enabled)
        self.brightness_spin.setEnabled(enabled)
        self.contrast_spin.setEnabled(enabled)

        self.name_input.setEnabled(enabled)
        self.btn_apply_name.setEnabled(enabled)
        self.btn_reset_name.setEnabled(enabled)

        self.btn_export.setEnabled(enabled)
        self.chk_overwrite.setEnabled(enabled)

    # ---------------- file ----------------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            img = Image.open(path).convert("RGBA")
            self.src_path = path

            self.source_color = img
            self.master_color = img.copy()

            self.is_bw = False
            self.has_black_bg = False
            self.target_size = None
            self.output_basename = None

            # reset sliders/spin to 100
            for w in (
                self.brightness_slider,
                self.contrast_slider,
            ):
                w.blockSignals(True)

            self.brightness_slider.setValue(100)
            self.contrast_slider.setValue(100)
            self.brightness_spin.setText("100%")
            self.contrast_spin.setText("100%")

            for w in (
                self.brightness_slider,
                self.contrast_slider,
            ):
                w.blockSignals(False)

            self.name_input.blockSignals(True)
            self.name_input.setText("")
            self.name_input.blockSignals(False)
            self.update_name_preview()

            self.rebuild_working()
            self.set_enabled(True)
            self.update_info()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：\n{e}")

    def _on_transfer_from_viewer(self, tmp_png_path: str):
        """从全能看图 tab 接收转移过来的图片（临时 PNG 文件）"""
        try:
            self.load_image(tmp_png_path)
            # 切换到贴图修改 tab（索引 0）
            self._tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "转移失败", f"加载转移的图片时出错：\n{e}")
        finally:
            # 清理临时文件
            try:
                if os.path.exists(tmp_png_path):
                    os.remove(tmp_png_path)
            except OSError:
                pass

    # ---------------- crop ----------------
    def open_crop_dialog(self):
        if self.source_color is None:
            return
        dlg = CropDialog(self.source_color, self)  # 永远从最初图裁
        if dlg.exec() == QDialog.Accepted and dlg.result_img is not None:
            self.master_color = dlg.result_img.convert("RGBA")
            self.rebuild_working()

    def reset_crop(self):
        # 取消裁切：回到最初导入图（但保留你当前黑白/尺寸/亮度对比度设置）
        if self.source_color is None:
            return
        self.master_color = self.source_color.copy()
        self.rebuild_working()

    # ---------------- BW ----------------
    def apply_black_white(self):
        if self.master_color is None:
            return
        self.is_bw = True
        self.rebuild_working()

    def cancel_black_white(self):
        if self.master_color is None:
            return
        self.is_bw = False
        self.rebuild_working()

    # ---------------- mask generation ----------------
    def generate_mask(self):
        """遮罩生成：如果图片带透明通道，则将非透明部分转为白色。"""
        if self.master_color is None:
            return
        img = self.master_color.convert("RGBA")
        alpha = img.split()[3]
        # 判断是否有实际的透明区域（不是所有像素都完全不透明）
        alpha_extrema = alpha.getextrema()
        if alpha_extrema[0] < 255:
            # 带透明通道：非透明部分转为白色，透明部分保持透明
            white = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # 用 alpha 作为遮罩，非透明区域填白，透明区域保持
            result = Image.new("RGBA", img.size, (0, 0, 0, 0))
            result.paste(white, mask=alpha)
            self.master_color = result
            self.rebuild_working()
        else:
            # 非透明通道图片：基于当前显示画面，通过亮度阈值识别主体
            # 获取当前显示画面（已应用亮度/对比度调整）
            current_display = self.preview_img if self.preview_img is not None else self.working_img
            if current_display is None:
                return
            dlg = MaskThresholdDialog(current_display, self)
            if dlg.exec() == QDialog.Accepted and dlg.result_img is not None:
                self.master_color = dlg.result_img
                self.rebuild_working()

    def reset_mask(self):
        """重置遮罩生成：恢复到原始导入图片。"""
        if self.source_color is None:
            return
        self.master_color = self.source_color.copy()
        self.rebuild_working()

    # ---------------- black bg ----------------
    def apply_black_bg(self):
        if self.master_color is None:
            return
        self.has_black_bg = True
        self.rebuild_working()

    def reset_black_bg(self):
        if self.master_color is None:
            return
        self.has_black_bg = False
        self.rebuild_working()

    # ---------------- brightness/contrast sync ----------------
    def _on_brightness_slider(self, v: int):
        self.brightness_spin.blockSignals(True)
        self.brightness_spin.setText(f"{v}%")
        self.brightness_spin.blockSignals(False)
        self.update_preview()

    def _on_brightness_spin(self):
        try:
            v = int(self.brightness_spin.text().replace("%", "").strip())
            v = max(0, min(200, v))
        except ValueError:
            v = 100
        self.brightness_spin.setText(f"{v}%")
        self.brightness_slider.blockSignals(True)
        self.brightness_slider.setValue(v)
        self.brightness_slider.blockSignals(False)
        self.update_preview()

    def _on_contrast_slider(self, v: int):
        self.contrast_spin.blockSignals(True)
        self.contrast_spin.setText(f"{v}%")
        self.contrast_spin.blockSignals(False)
        self.update_preview()

    def _on_contrast_spin(self):
        try:
            v = int(self.contrast_spin.text().replace("%", "").strip())
            v = max(0, min(200, v))
        except ValueError:
            v = 100
        self.contrast_spin.setText(f"{v}%")
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(v)
        self.contrast_slider.blockSignals(False)
        self.update_preview()

    def reset_brightness(self):
        self.brightness_slider.setValue(100)

    def reset_contrast(self):
        self.contrast_slider.setValue(100)

    # ---------------- size ----------------
    def set_size(self, w: int, h: int):
        self.target_size = (int(w), int(h))
        self.custom_w.blockSignals(True)
        self.custom_h.blockSignals(True)
        self.custom_w.setCurrentText(str(int(w)))
        self.custom_h.setCurrentText(str(int(h)))
        self.custom_w.blockSignals(False)
        self.custom_h.blockSignals(False)
        self.rebuild_working()

    def apply_custom_size(self):
        try:
            w = int(self.custom_w.currentText())
        except ValueError:
            w = 1024
        try:
            h = int(self.custom_h.currentText())
        except ValueError:
            h = 1024
        self.set_size(w, h)

    def reset_size(self):
        self.target_size = None
        self.rebuild_working()

    # ---------------- naming ----------------
    def original_base(self) -> str:
        if not self.src_path:
            return "Unnamed"
        return os.path.splitext(os.path.basename(self.src_path))[0]

    def compute_preview_basename(self) -> str:
        tag = (self.name_input.text() or "").strip()
        if tag:
            return f"T_{tag}"
        return self.original_base()

    # ===== 导出路径记忆 =====
    def _get_export_dir_cache_path(self) -> str:
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "last_export_dir.txt")

    def _load_last_export_dir(self) -> str:
        try:
            with open(self._get_export_dir_cache_path(), "r", encoding="utf-8") as f:
                d = f.read().strip()
                if d and os.path.isdir(d):
                    return d
        except Exception:
            pass
        return os.path.dirname(self.src_path) if self.src_path else ""

    def _save_last_export_dir(self, path: str):
        try:
            with open(self._get_export_dir_cache_path(), "w", encoding="utf-8") as f:
                f.write(os.path.dirname(path))
        except Exception:
            pass

    # ===== 历史命名系统 =====
    def get_history_path(self) -> str:
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "name_history.txt")

    def save_last_name(self, name: str):
        try:
            with open(self.get_history_path(), "w", encoding="utf8") as f:
                f.write(name)
        except Exception:
            pass

    def load_last_name(self) -> Optional[str]:
        try:
            with open(self.get_history_path(), "r", encoding="utf8") as f:
                s = f.read().strip()
                return s or None
        except Exception:
            return None

    def apply_history_name(self, text: str):
        self.name_input.setText(text)

    def build_history_buttons(self, last: str):
        for b in (self.history_btn1, self.history_btn2, self.history_btn3):
            b.setVisible(False)

        m = re.match(r"(.+?)_(\d+)$", last)
        buttons = []

        if m:
            base = m.group(1)
            num = int(m.group(2))
            buttons = [base, f"{base}_{num + 1}"]
        else:
            if "_" in last:
                buttons = [last.split("_")[0], last, f"{last}_1"]
            else:
                buttons = [last, f"{last}_1"]

        btns = [self.history_btn1, self.history_btn2, self.history_btn3]
        for i, text in enumerate(buttons[:3]):
            btns[i].setText(text)
            btns[i].setVisible(True)

    def update_name_preview(self):
        if not self.src_path:
            self.name_preview.setText("预览：-")
            return
        preview = self.compute_preview_basename()
        locked = "（已应用）" if self.output_basename else "（未应用）"
        self.name_preview.setText(f"预览：{preview} {locked}")

    def apply_naming(self):
        if not self.src_path:
            return

        self.output_basename = self.compute_preview_basename()

        tag = (self.name_input.text() or "").strip()
        if tag:
            self.save_last_name(tag)
            self.build_history_buttons(tag)

        self.update_name_preview()
        self.update_info()
        QMessageBox.information(self, "命名已应用", f"导出时将使用：\n{self.output_basename}")

    def reset_naming(self):
        self.output_basename = None
        self.name_input.setText("")
        self.update_name_preview()
        self.update_info()

    def get_export_basename(self) -> str:
        return self.output_basename if self.output_basename else self.original_base()

    def validate_export_name(self, name: str) -> bool:
        return bool(VALID_NAME_RE.fullmatch(name))

    # ---------------- pipeline ----------------
    def rebuild_working(self):
        if self.master_color is None:
            return

        base = self.master_color

        if self.is_bw:
            base = to_bw_rgba(base)

        if self.has_black_bg:
            black = Image.new("RGBA", base.size, (0, 0, 0, 255))
            black.paste(base, mask=base.split()[3])
            black.putalpha(255)  # 强制完全不透明，确保黑底区域纯黑
            base = black

        if self.target_size is not None:
            w, h = self.target_size
            base = base.resize((int(w), int(h)), resample=Image.LANCZOS)

        self.working_img = base
        self.update_preview()
        self.update_info()

    def update_info(self):
        if self.working_img is None:
            self.info_label.setText("未导入图片")
            return
        w, h = self.working_img.size
        name = os.path.basename(self.src_path) if self.src_path else "(未命名)"
        export_base = self.get_export_basename()
        bw_state = "黑白" if self.is_bw else "彩色"
        size_state = (
            f"{self.target_size[0]}x{self.target_size[1]}"
            if self.target_size
            else "原尺寸"
        )
        self.info_label.setText(
            f"当前：{name}\n当前尺寸：{w} x {h}（{size_state}）\n模式：{bw_state}\n导出名：{export_base}"
        )

    def update_preview(self):
        if self.working_img is None:
            self.preview_label.clear()
            return

        b = self.brightness_slider.value() / 100.0
        c = self.contrast_slider.value() / 100.0

        img = self.working_img.copy()
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        self.preview_img = img

        pix = pil_to_qpixmap(img)
        pix = pix.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    # ---------------- export ----------------
    def export_image(self):
        if self.preview_img is None or self.src_path is None:
            return

        export_base = self.get_export_basename()

        fmt = self.format_combo.currentText()
        ext = "png" if fmt == "PNG" else "jpg"

        # 覆盖原图
        if self.chk_overwrite.isChecked():
            orig_ext = os.path.splitext(self.src_path)[1].lower().lstrip(".")
            if orig_ext not in ("png", "jpg", "jpeg", "tga", "bmp", "webp"):
                QMessageBox.warning(self, "导出失败", "不支持覆盖此文件类型。请用另存为。")
                return

            folder = os.path.dirname(self.src_path)
            target_path = os.path.join(folder, f"{export_base}.{orig_ext}")

            try:
                bak_path = self.src_path + ".bak"
                if not os.path.exists(bak_path):
                    shutil.copy2(self.src_path, bak_path)
            except Exception:
                QMessageBox.information(self, "提示", "备份 .bak 失败，但仍将继续导出覆盖。")

            try:
                out = self.preview_img
                if orig_ext in ("jpg", "jpeg"):
                    bg = Image.new("RGB", out.size, (0, 0, 0))
                    bg.paste(out, mask=out.split()[-1])
                    bg.save(target_path, quality=95)
                else:
                    out.save(target_path)

                QMessageBox.information(self, "完成", f"已导出：\n{target_path}")
                return
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败：\n{e}")
                return

        # 另存为
        default_dir = self._load_last_export_dir()
        suggested = f"{export_base}.{ext}"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出图片",
            os.path.join(default_dir, suggested),
            "PNG (*.png);;JPG (*.jpg *.jpeg)",
        )
        if not path:
            return

        try:
            out = self.preview_img
            if fmt == "JPG":
                bg = Image.new("RGB", out.size, (0, 0, 0))
                bg.paste(out, mask=out.split()[-1])
                bg.save(path, quality=95)
            else:
                out.save(path)
            self._save_last_export_dir(path)
            QMessageBox.information(self, "完成", f"已导出：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.preview_img is not None:
            self.update_preview()


def pick_initial_path(argv) -> Optional[str]:
    if len(argv) >= 2:
        p = argv[1].strip().strip('"')
        if os.path.isfile(p):
            return p
    return None


def _show_crash_dialog(exc_text: str):
    """程序崩溃时弹出错误对话框，并将错误写入日志文件"""
    import traceback
    from datetime import datetime

    # 写入日志文件，方便用户反馈
    # 打包成 exe 后用 sys.executable 获取 exe 所在目录，否则用脚本目录
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "error_log.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(exc_text)
            f.write("\n")
    except Exception:
        pass

    # 弹出错误对话框
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setWindowTitle("程序启动失败")
        msg.setIcon(QMessageBox.Critical)
        msg.setText("程序遇到错误无法启动，请将以下信息截图或复制后反馈给开发者：")
        msg.setDetailedText(exc_text)
        msg.setInformativeText(f"错误日志已保存至：\n{log_path}")
        msg.exec()
    except Exception:
        pass


def main():
    # 切换工作目录到 exe/脚本所在目录，确保资源文件的相对路径正确
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        app = QApplication(sys.argv)
        # 给整个应用设置图标（任务栏图标，多路径回退查找）
        _ico_name = "TextureToolGUI.ico"
        _candidates = []
        if getattr(sys, 'frozen', False):
            _candidates.append(os.path.join(os.path.dirname(sys.executable), _ico_name))
            _candidates.append(os.path.join(getattr(sys, '_MEIPASS', ''), _ico_name))
        else:
            _candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), _ico_name))
        _ico_path = next((p for p in _candidates if os.path.exists(p)), None)
        if _ico_path:
            from PySide6.QtGui import QIcon
            app.setWindowIcon(QIcon(_ico_path))
        initial = pick_initial_path(sys.argv)
        w = MainWindow(initial_path=initial)
        w.show()
        sys.exit(app.exec())
    except Exception:
        import traceback
        _show_crash_dialog(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
