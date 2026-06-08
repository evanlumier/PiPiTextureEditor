# -*- coding: utf-8 -*-
"""
dialogs.py - 对话框组件

从 Texture_tool_GUI_with_tabs.py 拆分出的对话框相关类：
- PixRect: 像素矩形数据类
- CropCanvas: 裁切画布控件
- MaskThresholdDialog: 亮度阈值遮罩对话框
- CropDialog: 裁切/旋转对话框
"""

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from PySide6.QtCore import Qt, QRect, QPoint, QTimer
from PySide6.QtGui import (
    QPixmap,
    QPainter,
    QPainterPath,
    QPen,
    QColor,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QDialog,
    QDialogButtonBox,
)

from utils import pil_to_qpixmap


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

        self._original_img = pil_img.convert("RGBA")  # 保留原图用于重新旋转
        self.pil_img = self._original_img.copy()
        self.img_w, self.img_h = self.pil_img.size
        self._rotation_angle: float = 0.0  # 当前旋转角度

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
        self._rotation_angle = (self._rotation_angle + (-angle)) % 360
        if self._rotation_angle > 180:
            self._rotation_angle -= 360
        self.pil_img = self._original_img.rotate(-self._rotation_angle, expand=True, resample=Image.BICUBIC)
        self.img_w, self.img_h = self.pil_img.size
        # 清除裁切选区
        self.sel_rect = None
        self._render()

    def rotate_to_angle(self, angle: float):
        """旋转图片到指定角度（基于原图重新旋转）。"""
        self._rotation_angle = angle
        if angle == 0:
            self.pil_img = self._original_img.copy()
        else:
            # PIL rotate 正值为逆时针，所以取负
            self.pil_img = self._original_img.rotate(-angle, expand=True, resample=Image.BICUBIC)
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
            QSlider::groove:horizontal {
                height: 6px; background: #313244; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa; width: 14px; height: 14px;
                margin: -4px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
        """)

        self.canvas = CropCanvas(pil_img)
        self.canvas.setMinimumSize(860, 560)

        tips = QLabel("拖拽创建裁切框；框内拖动移动；拖动角点缩放；确定后应用。")
        tips.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)

        # ---- 顶部旋转滑块 ----
        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(8)

        rotate_label = QLabel("旋转：")
        rotate_label.setFixedWidth(40)

        self._rotate_slider = QSlider(Qt.Horizontal)
        self._rotate_slider.setRange(-1800, 1800)  # 0.1° 精度
        self._rotate_slider.setValue(0)
        self._rotate_slider.setSingleStep(1)
        self._rotate_slider.setPageStep(150)  # 15°

        self._rotate_value_label = QLabel("0°")
        self._rotate_value_label.setFixedWidth(48)
        self._rotate_value_label.setAlignment(Qt.AlignCenter)
        self._rotate_value_label.setStyleSheet("color: #89b4fa; font-size: 13px; font-weight: bold;")

        btn_reset_rotate = QPushButton("归零")
        btn_reset_rotate.setFixedSize(48, 28)
        btn_reset_rotate.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 2px 6px; border-radius: 5px;
                          background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            QPushButton:hover { background: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
        """)
        btn_reset_rotate.clicked.connect(self._reset_rotation)

        btn_ccw90 = QPushButton("↺")
        btn_ccw90.setToolTip("逆时针 90°")
        btn_ccw90.setFixedSize(28, 28)
        btn_ccw90.setStyleSheet("""
            QPushButton { font-size: 16px; padding: 0; border-radius: 5px;
                          background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            QPushButton:hover { background: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
        """)
        btn_ccw90.clicked.connect(lambda: self._rotate_step(-90))

        btn_cw90 = QPushButton("↻")
        btn_cw90.setToolTip("顺时针 90°")
        btn_cw90.setFixedSize(28, 28)
        btn_cw90.setStyleSheet("""
            QPushButton { font-size: 16px; padding: 0; border-radius: 5px;
                          background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            QPushButton:hover { background: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
        """)
        btn_cw90.clicked.connect(lambda: self._rotate_step(90))

        rotate_row.addWidget(rotate_label)
        rotate_row.addWidget(btn_ccw90)
        rotate_row.addWidget(self._rotate_slider, 1)
        rotate_row.addWidget(btn_cw90)
        rotate_row.addWidget(self._rotate_value_label)
        rotate_row.addWidget(btn_reset_rotate)

        self._rotate_slider.valueChanged.connect(self._on_rotate_slider_changed)

        # 防抖定时器（避免拖动滑块时频繁旋转大图导致卡顿）
        self._rotate_debounce = QTimer(self)
        self._rotate_debounce.setSingleShot(True)
        self._rotate_debounce.setInterval(50)  # 50ms 防抖
        self._rotate_debounce.timeout.connect(self._apply_rotation)
        self._pending_angle: float = 0.0

        # ---- 布局 ----
        layout = QVBoxLayout(self)
        layout.addWidget(tips)
        layout.addLayout(rotate_row)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(btns)

        self.result_img: Optional[Image.Image] = None

    def _on_rotate_slider_changed(self, value: int):
        """滑块值变化时旋转画布（带防抖）"""
        angle = value / 10.0
        self._rotate_value_label.setText(f"{angle:.1f}°")
        self._pending_angle = angle
        self._rotate_debounce.start()

    def _apply_rotation(self):
        """防抖结束后实际执行旋转"""
        self.canvas.rotate_to_angle(self._pending_angle)

    def _reset_rotation(self):
        """归零旋转"""
        self._rotate_slider.setValue(0)

    def _rotate_step(self, degrees: int):
        """步进旋转（±90°）"""
        current = self._rotate_slider.value() / 10.0
        new_angle = current + degrees
        # 限制在 -180 ~ 180 范围内
        if new_angle > 180:
            new_angle -= 360
        elif new_angle < -180:
            new_angle += 360
        self._rotate_slider.setValue(int(new_angle * 10))

    def _on_ok(self):
        cropped = self.canvas.get_cropped_image()
        if cropped is not None:
            # 有裁切框：应用裁切
            self.result_img = cropped.convert("RGBA")
        else:
            # 无裁切框：直接使用当前图片（可能已旋转）
            self.result_img = self.canvas.pil_img.convert("RGBA")
        self.accept()
