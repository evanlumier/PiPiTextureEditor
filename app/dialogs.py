# -*- coding: utf-8 -*-
"""
dialogs.py - 对话框组件

从 Texture_tool_GUI_with_tabs.py 拆分出的对话框相关类：
- PixRect: 像素矩形数据类
- CropCanvas: 裁切画布控件（支持裁切/旋转/留白预览）
- MaskThresholdDialog: 亮度阈值遮罩对话框
- CropDialog: 裁切/旋转/边缘留白对话框
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QRect, QPoint, QPointF, Signal
from PySide6.QtGui import (
    QPixmap,
    QPainter,
    QPainterPath,
    QPen,
    QColor,
    QFont,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QAbstractSpinBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QSizePolicy,
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
    """画布控件：支持裁切模式、旋转模式和留白预览可视化。"""

    HANDLE = 10
    MIN_SIZE = 12

    # 交互模式
    MODE_NONE = "none"
    MODE_CROP = "crop"
    MODE_ROTATE = "rotate"
    MODE_ERASER = "eraser"

    # 信号
    rotation_changed = Signal(float)
    crop_applied = Signal()    # 裁切框被用户确认应用
    crop_cancelled = Signal()  # 裁切框被用户取消

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

        # ── 视图缩放/平移 ──
        self._view_scale: float = 1.0  # 视图缩放倍率
        self._view_offset: QPointF = QPointF(0, 0)  # 视图平移偏移
        self._view_fit_done: bool = False  # 是否已执行过适配
        self._panning: bool = False  # 右键拖拽中
        self._pan_last_pos: Optional[QPointF] = None
        self._right_press_pos_view: Optional[QPointF] = None  # 右键按下位置

        # 当前交互模式
        self._mode: str = self.MODE_NONE

        # ── 裁切相关 ──
        self.sel_rect: Optional[QRect] = None
        self._crop_confirmed: bool = False  # 裁切框是否已被用户确认
        self.drag_mode: Optional[str] = None  # new/move/resize
        self.drag_start = QPoint()
        self.sel_start = QRect()
        self.resize_handle: Optional[str] = None

        # 裁切确认/取消按钮的点击区域
        self._btn_confirm_rect: Optional[QRect] = None
        self._btn_cancel_rect: Optional[QRect] = None

        # ── 旋转相关（Adobe式） ──
        self._rotate_dragging: bool = False
        self._rotate_prev_pos: Optional[QPointF] = None

        # ── 橡皮擦相关 ──
        self._eraser_size: int = 20  # 笔刷半径（px，图片像素）
        self._eraser_feather: int = 5  # 羽化半径（px）
        self._eraser_drawing: bool = False  # 是否正在绘制
        self._eraser_undo_stack: list = []  # undo栈，存储Alpha通道快照
        self._eraser_max_undo: int = 15  # 最大undo步数
        self._eraser_brush_mask: Optional[np.ndarray] = None  # 预计算的笔刷mask
        self._eraser_mouse_pos: Optional[QPoint] = None  # 当前鼠标位置（用于绘制光标）

        # ── 留白预览相关 ──
        self._margin_top: int = 0
        self._margin_bottom: int = 0
        self._margin_left: int = 0
        self._margin_right: int = 0
        self._feather_enabled: bool = False
        self._feather_px: int = 10
        self._margin_preview_pixmap: Optional[QPixmap] = None  # 方案B：实时计算的预览图
        self._margin_preview_rect: Optional[QRect] = None  # 预览图的显示区域

        self._render()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        old_mode = self._mode
        self._mode = value

        # 切换模式时：如果之前的模式有未烘焙的修改，先烘焙保存
        # 旋转模式：角度非零说明有旋转修改需要烘焙
        # 橡皮擦模式：有undo栈说明有擦除修改需要烘焙
        if old_mode != value and old_mode != self.MODE_NONE:
            need_bake = False
            if old_mode == self.MODE_ROTATE and abs(self._rotation_angle) > 0.01:
                need_bake = True
            elif old_mode == self.MODE_ERASER and len(self._eraser_undo_stack) > 0:
                need_bake = True
            if need_bake:
                self._bake_current_state()

        # 只退出交互状态
        self._rotate_dragging = False
        self._rotate_prev_pos = None
        self.drag_mode = None
        # 根据新模式设置光标
        if value == self.MODE_CROP:
            self.setCursor(Qt.CrossCursor)
        elif value == self.MODE_ROTATE:
            self.setCursor(Qt.OpenHandCursor)
        elif value == self.MODE_ERASER:
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_margin_preview(self, top: int, bottom: int, left: int, right: int):
        """设置留白预览参数（用于画布上可视化显示）"""
        self._margin_top = top
        self._margin_bottom = bottom
        self._margin_left = left
        self._margin_right = right
        self._rebuild_margin_preview()
        self.update()

    def set_feather_preview(self, enabled: bool, px: int = 10):
        """设置羽化预览参数"""
        self._feather_enabled = enabled
        self._feather_px = px
        self._rebuild_margin_preview()
        self.update()

    def _rebuild_margin_preview(self):
        """方案B：实时计算带留白+羽化的预览图（所见即所得）"""
        mt, mb, ml, mr = self._margin_top, self._margin_bottom, self._margin_left, self._margin_right
        if mt == 0 and mb == 0 and ml == 0 and mr == 0:
            self._margin_preview_pixmap = None
            self._margin_preview_rect = None
            return

        # 基于当前画布图片生成带留白的预览图
        img = self.pil_img.convert("RGBA")
        new_w = img.width + ml + mr
        new_h = img.height + mt + mb
        canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        canvas.paste(img, (ml, mt))

        # 如果开启了羽化，应用羽化效果
        if self._feather_enabled and self._feather_px > 0:
            canvas = self._apply_feather_preview(canvas, self._feather_px, mt, mb, ml, mr)

        # 转为 QPixmap
        self._margin_preview_pixmap = pil_to_qpixmap(canvas)

    @staticmethod
    def _apply_feather_preview(img: Image.Image, feather_px: int,
                               top: int, bottom: int, left: int, right: int) -> Image.Image:
        """对预览图应用羽化效果（与最终输出完全一致的算法）"""
        if feather_px <= 0:
            return img
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # 原图内容区域边界
        ct = top
        cb = h - bottom if bottom > 0 else h
        cl = left
        cr = w - right if right > 0 else w

        # 顶部边缘
        if top > 0:
            f = min(feather_px, cb - ct)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[ct + i, cl:cr, 3] *= g[i]

        # 底部边缘
        if bottom > 0:
            f = min(feather_px, cb - ct)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[cb - 1 - i, cl:cr, 3] *= g[i]

        # 左侧边缘
        if left > 0:
            f = min(feather_px, cr - cl)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[ct:cb, cl + i, 3] *= g[i]

        # 右侧边缘
        if right > 0:
            f = min(feather_px, cr - cl)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[ct:cb, cr - 1 - i, 3] *= g[i]

        return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")

    def _render(self):
        """将 PIL 图片转为 QPixmap 并更新视图"""
        self._pixmap_scaled = pil_to_qpixmap(self.pil_img)
        # 不再使用 setPixmap()，改为 paintEvent 中手动绘制
        self.setPixmap(QPixmap())  # 清空 QLabel 的 pixmap
        if not self._view_fit_done:
            self._fit_to_view()
        self._update_pix_rect()
        # 如果有留白参数，重新生成预览图（因为底图可能变了）
        self._rebuild_margin_preview()
        self.update()

    def _fit_to_view(self):
        """让图片完整显示在控件中（适配窗口）"""
        pm = self._pixmap_scaled
        if pm is None or pm.isNull():
            return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        pw, ph = pm.width(), pm.height()
        if pw <= 0 or ph <= 0:
            return
        self._view_scale = min(w / pw, h / ph)
        # 居中
        disp_w = pw * self._view_scale
        disp_h = ph * self._view_scale
        self._view_offset = QPointF((w - disp_w) / 2, (h - disp_h) / 2)
        self._view_fit_done = True

    def _update_pix_rect(self):
        """根据当前视图缩放和偏移计算 pixmap 在控件中的显示矩形"""
        pm = self._pixmap_scaled
        if pm is None or pm.isNull():
            self._pix_rect = None
            return
        pw = int(pm.width() * self._view_scale)
        ph = int(pm.height() * self._view_scale)
        x = int(self._view_offset.x())
        y = int(self._view_offset.y())
        self._pix_rect = QRect(x, y, pw, ph)

    def _clamp_view_offset(self):
        """限制图片不能完全拖出视图（至少保留 20% 可见）"""
        pm = self._pixmap_scaled
        if pm is None or pm.isNull():
            return
        pw = pm.width() * self._view_scale
        ph = pm.height() * self._view_scale
        w, h = self.width(), self.height()
        margin_x = pw * 0.2
        margin_y = ph * 0.2
        min_x = -(pw - margin_x)
        max_x = w - margin_x
        min_y = -(ph - margin_y)
        max_y = h - margin_y
        ox = max(min_x, min(max_x, self._view_offset.x()))
        oy = max(min_y, min(max_y, self._view_offset.y()))
        self._view_offset = QPointF(ox, oy)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # 窗口大小变化时重新适配
        self._view_fit_done = False
        self._render()

    # ── 裁切辅助方法 ──

    def _clamp_to_pix(self, r: QRect) -> QRect:
        if not self._pix_rect:
            return r
        rr = r.normalized()
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

    # ── 鼠标事件 ──

    def mousePressEvent(self, event):
        pos = event.pos()

        # 右键：开始拖拽平移
        if event.button() == Qt.RightButton:
            self._right_press_pos_view = QPointF(pos)
            self._panning = False
            self._pan_last_pos = QPointF(pos)
            return

        if event.button() != Qt.LeftButton or not self._pix_rect:
            return

        # 检查是否点击了裁切确认/取消按钮
        if self.sel_rect and not self._crop_confirmed:
            if self._btn_confirm_rect and self._btn_confirm_rect.contains(pos):
                self._confirm_crop()
                return
            if self._btn_cancel_rect and self._btn_cancel_rect.contains(pos):
                self._cancel_crop()
                return

        if self._mode == self.MODE_CROP:
            if not self._pix_rect.contains(pos):
                return
            # 如果已有确认的裁切框，新拖拽会重新创建（覆盖旧的）
            if self._crop_confirmed:
                self._crop_confirmed = False
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

        elif self._mode == self.MODE_ROTATE:
            self._rotate_dragging = True
            self._rotate_prev_pos = QPointF(pos)
            self.setCursor(Qt.ClosedHandCursor)

        elif self._mode == self.MODE_ERASER:
            if self._pix_rect and self._pix_rect.contains(pos):
                # 开始一笔擦除：保存当前Alpha通道快照
                self._eraser_start_stroke()
                self._eraser_drawing = True
                self._eraser_apply_at(pos)
                self.update()

    def mouseMoveEvent(self, event):
        if not self._pix_rect:
            return

        pos = event.pos()

        # ── 右键拖拽平移 ──
        if self._right_press_pos_view is not None and (event.buttons() & Qt.RightButton):
            if not self._panning:
                delta = QPointF(pos) - self._right_press_pos_view
                dist = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
                if dist >= 5:  # 5px 阈值
                    self._panning = True
                    self.setCursor(Qt.ClosedHandCursor)
                    self._pan_last_pos = QPointF(pos)
                return
            # 已进入拖拽模式
            if self._pan_last_pos is not None:
                delta = QPointF(pos) - self._pan_last_pos
                self._view_offset = QPointF(
                    self._view_offset.x() + delta.x(),
                    self._view_offset.y() + delta.y()
                )
                self._clamp_view_offset()
                self._update_pix_rect()
                self._pan_last_pos = QPointF(pos)
                self.update()
            return

        if self._mode == self.MODE_CROP:
            if self.drag_mode is None:
                if self._hit_handle(pos):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif self.sel_rect and self.sel_rect.contains(pos):
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.CrossCursor)
                return

            shift_held = event.modifiers() & Qt.ShiftModifier

            if self.drag_mode == "new":
                end = QPoint(
                    max(self._pix_rect.left(), min(self._pix_rect.right(), pos.x())),
                    max(self._pix_rect.top(), min(self._pix_rect.bottom(), pos.y())),
                )
                if shift_held:
                    # Shift 约束正方形
                    dx = end.x() - self.drag_start.x()
                    dy = end.y() - self.drag_start.y()
                    side = max(abs(dx), abs(dy))
                    new_dx = side if dx >= 0 else -side
                    new_dy = side if dy >= 0 else -side
                    end = QPoint(self.drag_start.x() + new_dx, self.drag_start.y() + new_dy)
                    # 再次 clamp
                    end = QPoint(
                        max(self._pix_rect.left(), min(self._pix_rect.right(), end.x())),
                        max(self._pix_rect.top(), min(self._pix_rect.bottom(), end.y())),
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

                if shift_held:
                    # Shift 约束正方形（resize 时）
                    w_new = abs(right - left)
                    h_new = abs(bottom - top)
                    side = max(w_new, h_new)
                    if h in ("nw", "sw"):
                        left = right - side
                    else:
                        right = left + side
                    if h in ("nw", "ne"):
                        top = bottom - side
                    else:
                        bottom = top + side

                r = QRect(QPoint(left, top), QPoint(right, bottom)).normalized()
                r = self._clamp_to_pix(r)

                if r.width() < self.MIN_SIZE:
                    r.setRight(r.left() + self.MIN_SIZE)
                if r.height() < self.MIN_SIZE:
                    r.setBottom(r.top() + self.MIN_SIZE)

                self.sel_rect = self._clamp_to_pix(r)
                self.update()

        elif self._mode == self.MODE_ROTATE and self._rotate_dragging:
            # Adobe 式旋转：以画布中心为圆心
            center = QPointF(self._pix_rect.center())
            curr = QPointF(pos)
            prev = self._rotate_prev_pos

            if prev is not None:
                prev_angle = math.atan2(prev.y() - center.y(), prev.x() - center.x())
                curr_angle = math.atan2(curr.y() - center.y(), curr.x() - center.x())
                delta = math.degrees(curr_angle - prev_angle)
                # 处理 ±180° 跳变
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360

                new_angle = self._rotation_angle + delta
                # 限制在 -180 ~ 180
                if new_angle > 180:
                    new_angle -= 360
                elif new_angle < -180:
                    new_angle += 360

                self._rotation_angle = new_angle
                self._apply_rotation_internal()
                self.rotation_changed.emit(self._rotation_angle)

            self._rotate_prev_pos = curr

        elif self._mode == self.MODE_ERASER:
            self._eraser_mouse_pos = pos
            if self._eraser_drawing and (event.buttons() & Qt.LeftButton):
                self._eraser_apply_at(pos)
            self.update()

        else:
            # 无模式时默认光标
            if self._mode == self.MODE_ROTATE:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        # 右键释放
        if event.button() == Qt.RightButton:
            self._panning = False
            self._right_press_pos_view = None
            self._pan_last_pos = None
            # 恢复光标
            if self._mode == self.MODE_CROP:
                self.setCursor(Qt.CrossCursor)
            elif self._mode == self.MODE_ROTATE:
                self.setCursor(Qt.OpenHandCursor)
            elif self._mode == self.MODE_ERASER:
                self.setCursor(Qt.BlankCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._mode == self.MODE_CROP:
            self.drag_mode = None
            self.resize_handle = None
            self.setCursor(Qt.CrossCursor)
            if self.sel_rect:
                self.sel_rect = self.sel_rect.normalized()
                # 拖拽结束后，裁切框处于待确认状态
                self._crop_confirmed = False
                self.update()
        elif self._mode == self.MODE_ROTATE:
            self._rotate_dragging = False
            self._rotate_prev_pos = None
            self.setCursor(Qt.OpenHandCursor)

        elif self._mode == self.MODE_ERASER:
            if self._eraser_drawing:
                self._eraser_drawing = False
                # 笔画结束，刷新画布和留白预览
                self._render()
                self._rebuild_margin_preview()
                self.update()

    # ── 橡皮擦方法 ──

    def set_eraser_params(self, size: int, feather: int):
        """设置橡皮擦参数"""
        self._eraser_size = size
        self._eraser_feather = feather
        self._eraser_brush_mask = self._build_eraser_mask(size, feather)
        self.update()

    @staticmethod
    def _build_eraser_mask(size: int, feather: int) -> np.ndarray:
        """预计算圆形笔刷mask（带径向羽化渐变）
        返回 (2*size+1, 2*size+1) 的 float32 数组，值域 [0, 1]
        1 = 完全擦除，0 = 不擦除
        """
        diameter = size * 2 + 1
        y, x = np.ogrid[:diameter, :diameter]
        center = size
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2).astype(np.float32)

        mask = np.zeros((diameter, diameter), dtype=np.float32)
        inner_radius = size - feather

        # 核心区域：完全擦除
        mask[dist <= inner_radius] = 1.0

        # 羽化区域：线性渐变
        if feather > 0:
            feather_zone = (dist > inner_radius) & (dist <= size)
            mask[feather_zone] = 1.0 - (dist[feather_zone] - inner_radius) / feather
        else:
            mask[dist <= size] = 1.0

        return mask

    def _eraser_start_stroke(self):
        """开始一笔：保存当前Alpha通道快照到undo栈"""
        arr = np.array(self.pil_img, dtype=np.uint8)
        alpha_snapshot = arr[:, :, 3].copy()
        self._eraser_undo_stack.append(alpha_snapshot)
        # 限制undo栈深度
        if len(self._eraser_undo_stack) > self._eraser_max_undo:
            self._eraser_undo_stack.pop(0)

    def eraser_undo(self):
        """撤销一笔橡皮擦"""
        if not self._eraser_undo_stack:
            return False
        alpha_snapshot = self._eraser_undo_stack.pop()
        arr = np.array(self.pil_img, dtype=np.uint8)
        arr[:, :, 3] = alpha_snapshot
        self.pil_img = Image.fromarray(arr, "RGBA")
        self._render()
        self._rebuild_margin_preview()
        self.update()
        return True

    def _eraser_apply_at(self, screen_pos: QPoint):
        """在屏幕坐标处应用橡皮擦"""
        if not self._pix_rect or self._eraser_brush_mask is None:
            return

        # 屏幕坐标 → 图片像素坐标
        img_x, img_y = self._screen_to_img_coords(screen_pos)
        if img_x is None:
            return

        # 获取图片数组
        arr = np.array(self.pil_img, dtype=np.uint8)
        h, w = arr.shape[:2]
        alpha = arr[:, :, 3].astype(np.float32)

        # 计算笔刷覆盖区域（图片坐标系）
        size = self._eraser_size
        mask = self._eraser_brush_mask

        # 笔刷在图片上的边界
        bx_start = img_x - size
        by_start = img_y - size
        bx_end = img_x + size + 1
        by_end = img_y + size + 1

        # 裁剪到图片范围
        mx_start = max(0, -bx_start)
        my_start = max(0, -by_start)
        mx_end = mask.shape[1] - max(0, bx_end - w)
        my_end = mask.shape[0] - max(0, by_end - h)

        ix_start = max(0, bx_start)
        iy_start = max(0, by_start)
        ix_end = min(w, bx_end)
        iy_end = min(h, by_end)

        if ix_start >= ix_end or iy_start >= iy_end:
            return

        # 应用mask：alpha *= (1 - mask_strength)
        region = alpha[iy_start:iy_end, ix_start:ix_end]
        mask_region = mask[my_start:my_end, mx_start:mx_end]
        region *= (1.0 - mask_region)
        alpha[iy_start:iy_end, ix_start:ix_end] = region

        arr[:, :, 3] = alpha.clip(0, 255).astype(np.uint8)
        self.pil_img = Image.fromarray(arr, "RGBA")

        # 实时刷新画布显示
        self._render()
        self.update()

    def _screen_to_img_coords(self, pos: QPoint):
        """屏幕坐标转换为图片像素坐标"""
        if not self._pix_rect:
            return None, None

        # 相对于pix_rect的坐标
        rx = pos.x() - self._pix_rect.left()
        ry = pos.y() - self._pix_rect.top()

        pw = self._pix_rect.width()
        ph = self._pix_rect.height()

        if pw <= 0 or ph <= 0:
            return None, None

        # 转为图片像素坐标
        img_x = int(rx / pw * self.img_w)
        img_y = int(ry / ph * self.img_h)

        # 边界检查
        if img_x < 0 or img_x >= self.img_w or img_y < 0 or img_y >= self.img_h:
            return None, None

        return img_x, img_y

    def _bake_current_state(self):
        """烘焙当前状态：将pil_img设为新的_original_img，角度归零"""
        self._original_img = self.pil_img.copy()
        self._rotation_angle = 0.0
        self.img_w, self.img_h = self.pil_img.size
        # 清除裁切框
        self.sel_rect = None
        self._crop_confirmed = False
        self._btn_confirm_rect = None
        self._btn_cancel_rect = None
        # 清空undo栈（烘焙后之前的undo无效）
        self._eraser_undo_stack.clear()
        self.rotation_changed.emit(0.0)

    # ── 滚轮缩放 ──

    def wheelEvent(self, event):
        """滚轮缩放，以鼠标位置为缩放中心"""
        pm = self._pixmap_scaled
        if pm is None or pm.isNull():
            return
        pos = event.position()
        old_scale = self._view_scale
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_scale = max(0.05, min(old_scale * factor, 20.0))
        # 以鼠标位置为缩放中心
        self._view_offset = QPointF(
            pos.x() - (pos.x() - self._view_offset.x()) * new_scale / old_scale,
            pos.y() - (pos.y() - self._view_offset.y()) * new_scale / old_scale,
        )
        self._view_scale = new_scale
        self._clamp_view_offset()
        self._update_pix_rect()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """双击左键适配窗口"""
        if event.button() == Qt.LeftButton:
            self._fit_to_view()
            self._update_pix_rect()
            self.update()

    def leaveEvent(self, event):
        """鼠标离开画布时清除橡皮擦光标"""
        if self._mode == self.MODE_ERASER:
            self._eraser_mouse_pos = None
            self.update()
        super().leaveEvent(event)

    # ── 裁切确认/取消 ──

    def _confirm_crop(self):
        """用户确认裁切框：应用裁切到画布图片"""
        cropped = self.get_cropped_image()
        if cropped is not None:
            # 应用裁切：更新原图和当前图
            self._original_img = cropped.convert("RGBA")
            self.pil_img = self._original_img.copy()
            self.img_w, self.img_h = self.pil_img.size
            self._rotation_angle = 0.0
            self.sel_rect = None
            self._crop_confirmed = False
            self._btn_confirm_rect = None
            self._btn_cancel_rect = None
            self._view_fit_done = False  # 裁切后重新适配视图
            self._eraser_undo_stack.clear()  # 裁切后undo栈失效
            self._render()
            self.rotation_changed.emit(0.0)  # 同步角度标签
            self.crop_applied.emit()

    def _cancel_crop(self):
        """用户取消裁切框"""
        self.sel_rect = None
        self._crop_confirmed = False
        self._btn_confirm_rect = None
        self._btn_cancel_rect = None
        self.update()
        self.crop_cancelled.emit()

    # ── 旋转方法 ──

    def _apply_rotation_internal(self):
        """内部旋转应用（基于原图重新旋转到当前角度）"""
        if self._rotation_angle == 0:
            self.pil_img = self._original_img.copy()
        else:
            self.pil_img = self._original_img.rotate(
                -self._rotation_angle, expand=True, resample=Image.BICUBIC
            )
        self.img_w, self.img_h = self.pil_img.size
        # 旋转时清除未确认的裁切框（因为图片尺寸变了坐标不再有效）
        if not self._crop_confirmed:
            self.sel_rect = None
        # 旋转后重新适配视图，确保图片始终居中显示
        self._view_fit_done = False
        self._render()

    def rotate_to_angle(self, angle: float):
        """外部设置旋转角度"""
        self._rotation_angle = angle
        self._apply_rotation_internal()

    @property
    def rotation_angle(self) -> float:
        return self._rotation_angle

    # ── 裁切结果 ──

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

    # ── 绘制 ──

    def paintEvent(self, e):
        # 背景
        p0 = QPainter(self)
        p0.setPen(Qt.NoPen)
        p0.setBrush(QColor(42, 42, 56))  # 整体背景
        p0.drawRect(self.rect())

        # 绘制图片区域底色
        if self._pix_rect:
            p0.setBrush(QColor(58, 58, 74))  # #3a3a4a
            p0.drawRect(self._pix_rect)

        # 绘制图片（手动绘制，支持缩放和平移）
        if self._margin_preview_pixmap and not self._margin_preview_pixmap.isNull() and self._pix_rect:
            # 方案B：显示带留白+羽化的完整预览图
            self._draw_margin_preview_image(p0)
        elif self._pixmap_scaled and not self._pixmap_scaled.isNull() and self._pix_rect:
            p0.drawPixmap(self._pix_rect, self._pixmap_scaled)

        p0.end()

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # 图片边缘指引线
        if self._pix_rect:
            border_pen = QPen(QColor(90, 90, 106))
            border_pen.setWidth(1)
            p.setPen(border_pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(self._pix_rect)

        # ── 留白预览可视化 ──
        if self._pix_rect and (self._margin_top or self._margin_bottom or self._margin_left or self._margin_right):
            self._draw_margin_preview(p)

        # ── 裁切选区 ──
        if self.sel_rect:
            sel_r = self.sel_rect.normalized()

            # 半透明遮罩
            path = QPainterPath()
            path.addRect(self.rect())
            path.addRect(sel_r)
            path.setFillRule(Qt.OddEvenFill)
            p.fillPath(path, QColor(0, 0, 0, 80))

            # 选区边框
            pen = QPen(QColor(255, 255, 255, 230))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRect(sel_r)

            # 四角 handles（仅裁切模式下显示）
            if self._mode == self.MODE_CROP:
                for hr in self._handle_rects().values():
                    p.fillRect(hr, QColor(255, 255, 255, 230))

            # ── 裁切确认/取消按钮（右上角） ──
            if not self._crop_confirmed:
                self._draw_crop_action_buttons(p, sel_r)

        # ── 橡皮擦光标（圆形笔刷预览） ──
        if self._mode == self.MODE_ERASER and self._eraser_mouse_pos and self._pix_rect:
            self._draw_eraser_cursor(p)

        p.end()

    def _draw_crop_action_buttons(self, p: QPainter, sel_r: QRect):
        """在裁切框右上角绘制 ✓ 和 ✗ 按钮"""
        btn_size = 24
        btn_gap = 4
        # 按钮位于裁切框右上角外侧
        base_x = sel_r.right() - btn_size * 2 - btn_gap
        base_y = sel_r.top() - btn_size - 4

        # 如果上方空间不够，放到框内右上角
        if base_y < 0:
            base_y = sel_r.top() + 4

        # ✓ 确认按钮（绿色）
        confirm_rect = QRect(base_x, base_y, btn_size, btn_size)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(166, 227, 161, 220))  # #a6e3a1 绿色
        p.drawRoundedRect(confirm_rect, 4, 4)
        p.setPen(QColor(30, 30, 46))
        p.setFont(QFont("Arial", 13, QFont.Bold))
        p.drawText(confirm_rect, Qt.AlignCenter, "✓")

        # ✗ 取消按钮（红色）
        cancel_rect = QRect(base_x + btn_size + btn_gap, base_y, btn_size, btn_size)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(243, 139, 168, 220))  # #f38ba8 红色
        p.drawRoundedRect(cancel_rect, 4, 4)
        p.setPen(QColor(30, 30, 46))
        p.drawText(cancel_rect, Qt.AlignCenter, "✗")

        # 保存按钮区域用于点击检测
        self._btn_confirm_rect = confirm_rect
        self._btn_cancel_rect = cancel_rect

    def _draw_eraser_cursor(self, p: QPainter):
        """绘制橡皮擦圆形光标（显示笔刷范围和羽化范围）"""
        pos = self._eraser_mouse_pos
        if not pos or not self._pix_rect:
            return

        # 计算笔刷在屏幕上的半径（需要考虑视图缩放）
        pw = self._pix_rect.width()
        img_scale = pw / self.img_w if self.img_w > 0 else 1.0

        outer_radius = int(self._eraser_size * img_scale)
        inner_radius = int((self._eraser_size - self._eraser_feather) * img_scale)

        # 外圈（笔刷总范围）- 白色虚线
        pen_outer = QPen(QColor(255, 255, 255, 180))
        pen_outer.setWidth(1)
        pen_outer.setStyle(Qt.DashLine)
        p.setPen(pen_outer)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(pos, outer_radius, outer_radius)

        # 内圈（完全擦除区域）- 白色实线
        if inner_radius > 0 and self._eraser_feather > 0:
            pen_inner = QPen(QColor(255, 255, 255, 120))
            pen_inner.setWidth(1)
            p.setPen(pen_inner)
            p.drawEllipse(pos, inner_radius, inner_radius)

        # 中心十字
        cross_size = 3
        pen_cross = QPen(QColor(255, 255, 255, 200))
        pen_cross.setWidth(1)
        p.setPen(pen_cross)
        p.drawLine(pos.x() - cross_size, pos.y(), pos.x() + cross_size, pos.y())
        p.drawLine(pos.x(), pos.y() - cross_size, pos.x(), pos.y() + cross_size)

    def _draw_margin_preview_image(self, p: QPainter):
        """方案B：绘制带留白+羽化的完整预览图（所见即所得）"""
        if not self._pix_rect or not self._margin_preview_pixmap:
            return

        pr = self._pix_rect
        pw, ph = pr.width(), pr.height()

        # 计算留白在画布上的像素比例
        scale_x = pw / self.img_w if self.img_w > 0 else 1
        scale_y = ph / self.img_h if self.img_h > 0 else 1

        mt = int(self._margin_top * scale_y)
        mb = int(self._margin_bottom * scale_y)
        ml = int(self._margin_left * scale_x)
        mr = int(self._margin_right * scale_x)

        # 预览图的显示区域（包含留白的完整区域）
        preview_rect = QRect(
            pr.left() - ml, pr.top() - mt,
            pw + ml + mr, ph + mt + mb
        )
        p.drawPixmap(preview_rect, self._margin_preview_pixmap)

    def _draw_margin_preview(self, p: QPainter):
        """绘制留白区域的边界指示（虚线外框）"""
        if not self._pix_rect:
            return

        pr = self._pix_rect
        pw, ph = pr.width(), pr.height()

        # 计算留白在画布上的像素比例
        scale_x = pw / self.img_w if self.img_w > 0 else 1
        scale_y = ph / self.img_h if self.img_h > 0 else 1

        mt = int(self._margin_top * scale_y)
        mb = int(self._margin_bottom * scale_y)
        ml = int(self._margin_left * scale_x)
        mr = int(self._margin_right * scale_x)

        if mt == 0 and mb == 0 and ml == 0 and mr == 0:
            return

        # 绘制外框虚线（表示最终输出边界）
        dash_pen = QPen(QColor(137, 180, 250, 180))
        dash_pen.setWidth(1)
        dash_pen.setStyle(Qt.DashLine)
        p.setPen(dash_pen)
        p.setBrush(Qt.NoBrush)
        outer_rect = QRect(
            pr.left() - ml, pr.top() - mt,
            pw + ml + mr, ph + mt + mb
        )
        p.drawRect(outer_rect)

        # 绘制原图边界指示线（区分原图和留白区域）
        inner_pen = QPen(QColor(166, 227, 161, 120))  # 绿色半透明
        inner_pen.setWidth(1)
        inner_pen.setStyle(Qt.DotLine)
        p.setPen(inner_pen)
        p.drawRect(pr)


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
    """裁切/旋转/边缘留白对话框 — 左侧画布 + 右侧工具栏布局"""

    # Toggle 按钮样式
    _TOGGLE_OFF_STYLE = """
        QPushButton { font-size: 13px; padding: 6px 16px; border-radius: 7px;
                      background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
        QPushButton:hover { background: #45475a; border-color: #89b4fa; }
    """
    _TOGGLE_ON_STYLE = """
        QPushButton { font-size: 13px; padding: 6px 16px; border-radius: 7px;
                      background: #89b4fa; color: #1e1e2e; border: 1px solid #89b4fa;
                      font-weight: bold; }
        QPushButton:hover { background: #74c7ec; border-color: #74c7ec; }
    """

    def __init__(self, pil_img: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像调整")
        self.resize(1100, 720)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #a6adc8; font-size: 12px; }
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 7px;
                padding: 6px 18px; font-size: 13px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
            QSpinBox {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 5px; padding: 3px 8px; font-size: 12px;
            }
            QSpinBox:focus { border-color: #89b4fa; }
            QCheckBox { color: #cdd6f4; font-size: 12px; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border-radius: 3px;
                border: 1px solid #45475a; background: #313244;
            }
            QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
        """)

        self.result_img: Optional[Image.Image] = None

        # ── 画布 ──
        self.canvas = CropCanvas(pil_img)
        self.canvas.setMinimumSize(700, 560)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── 右侧工具栏 ──
        right_panel = QWidget()
        right_panel.setFixedWidth(240)
        right_panel.setStyleSheet("QWidget { background: transparent; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 8, 8, 8)
        right_layout.setSpacing(12)

        # ─── Toggle 按钮行（裁切 / 旋转 互斥） ───
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(8)

        self._btn_crop = QPushButton("✂ 裁切")
        self._btn_crop.setCheckable(True)
        self._btn_crop.setFixedHeight(32)
        self._btn_crop.setAutoDefault(False)
        self._btn_crop.setDefault(False)
        self._btn_crop.setStyleSheet(self._TOGGLE_OFF_STYLE)
        self._btn_crop.clicked.connect(self._on_crop_toggle)

        self._btn_rotate = QPushButton("↻ 旋转")
        self._btn_rotate.setCheckable(True)
        self._btn_rotate.setFixedHeight(32)
        self._btn_rotate.setAutoDefault(False)
        self._btn_rotate.setDefault(False)
        self._btn_rotate.setStyleSheet(self._TOGGLE_OFF_STYLE)
        self._btn_rotate.clicked.connect(self._on_rotate_toggle)

        toggle_row.addWidget(self._btn_crop, 1)
        toggle_row.addWidget(self._btn_rotate, 1)
        right_layout.addLayout(toggle_row)

        # 旋转角度输入行
        self._rotate_row = QWidget()
        rotate_row_layout = QHBoxLayout(self._rotate_row)
        rotate_row_layout.setContentsMargins(0, 0, 0, 0)
        rotate_row_layout.setSpacing(6)

        rotate_label = QLabel("旋转角度：")
        rotate_label.setStyleSheet("color: #a6adc8; font-size: 12px;")

        self._rotate_spin = QDoubleSpinBox()
        self._rotate_spin.setRange(-180.0, 180.0)
        self._rotate_spin.setDecimals(1)
        self._rotate_spin.setSingleStep(1.0)
        self._rotate_spin.setValue(0.0)
        self._rotate_spin.setSuffix("°")
        self._rotate_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)  # 去掉上下箭头
        self._rotate_spin.setKeyboardTracking(False)  # 仅在回车/失焦时触发valueChanged
        self._rotate_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #313244; color: #89b4fa; border: 1px solid #45475a;
                border-radius: 5px; padding: 3px 8px; font-size: 12px;
                font-weight: bold;
            }
            QDoubleSpinBox:focus { border-color: #89b4fa; }
        """)
        self._rotate_spin.valueChanged.connect(self._on_rotate_spin_changed)
        self._rotate_spin_updating = False  # 防止循环触发

        btn_reset_angle = QPushButton("归零")
        btn_reset_angle.setFixedSize(40, 24)
        btn_reset_angle.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 2px 4px; border-radius: 4px;
                          background: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            QPushButton:hover { background: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
        """)
        btn_reset_angle.clicked.connect(self._reset_rotation)

        rotate_row_layout.addWidget(rotate_label)
        rotate_row_layout.addWidget(self._rotate_spin, 1)
        rotate_row_layout.addWidget(btn_reset_angle)

        self._rotate_row.setVisible(False)
        right_layout.addWidget(self._rotate_row)

        # ─── 分隔线 ───
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("QFrame { color: #45475a; }")
        right_layout.addWidget(sep1)

        # ─── 边缘留白区域 ───
        margin_title = QLabel("边缘留白")
        margin_title.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")
        right_layout.addWidget(margin_title)

        margin_grid = QGridLayout()
        margin_grid.setSpacing(6)

        margin_grid.addWidget(QLabel("上："), 0, 0)
        self._spin_top = QSpinBox()
        self._spin_top.setRange(0, 500)
        margin_grid.addWidget(self._spin_top, 0, 1)
        margin_grid.addWidget(QLabel("px"), 0, 2)

        margin_grid.addWidget(QLabel("下："), 1, 0)
        self._spin_bottom = QSpinBox()
        self._spin_bottom.setRange(0, 500)
        margin_grid.addWidget(self._spin_bottom, 1, 1)
        margin_grid.addWidget(QLabel("px"), 1, 2)

        margin_grid.addWidget(QLabel("左："), 2, 0)
        self._spin_left = QSpinBox()
        self._spin_left.setRange(0, 500)
        margin_grid.addWidget(self._spin_left, 2, 1)
        margin_grid.addWidget(QLabel("px"), 2, 2)

        margin_grid.addWidget(QLabel("右："), 3, 0)
        self._spin_right = QSpinBox()
        self._spin_right.setRange(0, 500)
        margin_grid.addWidget(self._spin_right, 3, 1)
        margin_grid.addWidget(QLabel("px"), 3, 2)

        right_layout.addLayout(margin_grid)

        # 留白值变化时实时更新预览
        self._spin_top.valueChanged.connect(self._on_margin_changed)
        self._spin_bottom.valueChanged.connect(self._on_margin_changed)
        self._spin_left.valueChanged.connect(self._on_margin_changed)
        self._spin_right.valueChanged.connect(self._on_margin_changed)

        # ─── 羽化 ───
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("QFrame { color: #45475a; }")
        right_layout.addWidget(sep2)

        feather_row = QHBoxLayout()
        feather_row.setSpacing(8)
        self._chk_feather = QCheckBox("羽化")
        self._chk_feather.setChecked(False)
        feather_row.addWidget(self._chk_feather)

        feather_row.addWidget(QLabel("强度："))
        self._spin_feather = QSpinBox()
        self._spin_feather.setRange(1, 100)
        self._spin_feather.setValue(10)
        self._spin_feather.setEnabled(False)
        feather_row.addWidget(self._spin_feather)
        feather_row.addWidget(QLabel("px"))

        right_layout.addLayout(feather_row)
        self._chk_feather.toggled.connect(self._spin_feather.setEnabled)
        self._chk_feather.toggled.connect(self._on_feather_changed)
        self._spin_feather.valueChanged.connect(self._on_feather_changed)

        # ─── 橡皮擦 ───
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("QFrame { color: #45475a; }")
        right_layout.addWidget(sep3)

        self._btn_eraser = QPushButton("🧽 橡皮擦")
        self._btn_eraser.setCheckable(True)
        self._btn_eraser.setFixedHeight(32)
        self._btn_eraser.setAutoDefault(False)
        self._btn_eraser.setDefault(False)
        self._btn_eraser.setStyleSheet(self._TOGGLE_OFF_STYLE)
        self._btn_eraser.clicked.connect(self._on_eraser_toggle)
        right_layout.addWidget(self._btn_eraser)

        # 橡皮擦参数行
        self._eraser_params_row = QWidget()
        eraser_params_layout = QGridLayout(self._eraser_params_row)
        eraser_params_layout.setContentsMargins(0, 0, 0, 0)
        eraser_params_layout.setSpacing(6)

        eraser_params_layout.addWidget(QLabel("笔刷大小："), 0, 0)
        self._spin_eraser_size = QSpinBox()
        self._spin_eraser_size.setRange(2, 200)
        self._spin_eraser_size.setValue(20)
        eraser_params_layout.addWidget(self._spin_eraser_size, 0, 1)
        eraser_params_layout.addWidget(QLabel("px"), 0, 2)

        eraser_params_layout.addWidget(QLabel("羽化半径："), 1, 0)
        self._spin_eraser_feather = QSpinBox()
        self._spin_eraser_feather.setRange(0, 100)
        self._spin_eraser_feather.setValue(5)
        eraser_params_layout.addWidget(self._spin_eraser_feather, 1, 1)
        eraser_params_layout.addWidget(QLabel("px"), 1, 2)

        self._eraser_params_row.setVisible(False)
        right_layout.addWidget(self._eraser_params_row)

        self._spin_eraser_size.valueChanged.connect(self._on_eraser_params_changed)
        self._spin_eraser_feather.valueChanged.connect(self._on_eraser_params_changed)

        # ─── 弹性空间 ───
        right_layout.addStretch(1)

        # ─── 提示文字 ───
        tips = QLabel("裁切：拖拽创建框，Shift正方形\n旋转：拖拽画布旋转\n留白/羽化：参数变化实时预览\n橡皮擦：涂抹擦除Alpha，Ctrl+Z撤销")
        tips.setWordWrap(True)
        tips.setStyleSheet("color: #6c7086; font-size: 11px;")
        right_layout.addWidget(tips)

        # ─── 确定/取消 ───
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_ok = QPushButton("确定")
        btn_ok.setAutoDefault(False)
        btn_ok.setDefault(False)
        btn_ok.setStyleSheet("""
            QPushButton { background: #89b4fa; color: #1e1e2e; font-weight: bold;
                          border: none; border-radius: 7px; padding: 8px 20px; }
            QPushButton:hover { background: #74c7ec; }
        """)
        btn_ok.clicked.connect(self._on_ok)

        btn_cancel = QPushButton("取消")
        btn_cancel.setAutoDefault(False)
        btn_cancel.setDefault(False)
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_ok, 1)
        btn_row.addWidget(btn_cancel, 1)
        right_layout.addLayout(btn_row)

        # ── 主布局（左画布 + 右工具栏） ──
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(right_panel)

        # ── 连接画布旋转信号 ──
        self.canvas.rotation_changed.connect(self._on_canvas_rotation_changed)

    # ── Toggle 互斥逻辑 ──

    def _on_crop_toggle(self):
        if self._btn_crop.isChecked():
            # 开启裁切，关闭旋转和橡皮擦模式
            self._btn_rotate.setChecked(False)
            self._btn_rotate.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._btn_eraser.setChecked(False)
            self._btn_eraser.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._eraser_params_row.setVisible(False)
            self._btn_crop.setStyleSheet(self._TOGGLE_ON_STYLE)
            self.canvas.mode = CropCanvas.MODE_CROP
        else:
            # 关闭裁切模式（保留裁切框，确定时仍会应用）
            self._btn_crop.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self.canvas.mode = CropCanvas.MODE_NONE

    def _on_rotate_toggle(self):
        if self._btn_rotate.isChecked():
            # 开启旋转，关闭裁切和橡皮擦模式
            self._btn_crop.setChecked(False)
            self._btn_crop.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._btn_eraser.setChecked(False)
            self._btn_eraser.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._eraser_params_row.setVisible(False)
            self._btn_rotate.setStyleSheet(self._TOGGLE_ON_STYLE)
            self.canvas.mode = CropCanvas.MODE_ROTATE
            self._rotate_row.setVisible(True)
            self._sync_spin_from_canvas()
        else:
            # 关闭旋转模式（保留旋转结果）
            self._btn_rotate.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self.canvas.mode = CropCanvas.MODE_NONE
            # 如果有旋转角度，保持角度行可见
            if abs(self.canvas.rotation_angle) > 0.01:
                self._rotate_row.setVisible(True)
            else:
                self._rotate_row.setVisible(False)

    def _on_eraser_toggle(self):
        if self._btn_eraser.isChecked():
            # 开启橡皮擦，关闭裁切和旋转模式
            self._btn_crop.setChecked(False)
            self._btn_crop.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._btn_rotate.setChecked(False)
            self._btn_rotate.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._btn_eraser.setStyleSheet(self._TOGGLE_ON_STYLE)
            self._eraser_params_row.setVisible(True)

            # 烘焙机制：如果当前有旋转角度，先烘焙
            if abs(self.canvas.rotation_angle) > 0.01:
                self.canvas._bake_current_state()
                self._sync_spin_from_canvas()
                self._rotate_row.setVisible(False)

            # 设置橡皮擦模式和参数
            self.canvas.mode = CropCanvas.MODE_ERASER
            self._on_eraser_params_changed()
            self.canvas.setCursor(Qt.BlankCursor)
        else:
            # 关闭橡皮擦模式
            self._btn_eraser.setStyleSheet(self._TOGGLE_OFF_STYLE)
            self._eraser_params_row.setVisible(False)
            self.canvas.mode = CropCanvas.MODE_NONE
            self.canvas._eraser_mouse_pos = None
            self.canvas.setCursor(Qt.ArrowCursor)
            self.canvas.update()

    def _on_eraser_params_changed(self):
        """橡皮擦参数变化时更新画布"""
        self.canvas.set_eraser_params(
            self._spin_eraser_size.value(),
            self._spin_eraser_feather.value(),
        )

    def _on_rotate_spin_changed(self, value: float):
        """用户手动输入角度时，应用到画布"""
        if self._rotate_spin_updating:
            return
        self.canvas.rotate_to_angle(value)

    def _reset_rotation(self):
        """归零旋转角度"""
        self._rotate_spin.setValue(0.0)

    def _on_canvas_rotation_changed(self, angle: float):
        """画布旋转角度变化时同步更新SpinBox显示"""
        self._sync_spin_from_canvas()
        # 角度归零且不在旋转模式时隐藏角度行
        if abs(angle) < 0.01 and not self._btn_rotate.isChecked():
            self._rotate_row.setVisible(False)
        else:
            self._rotate_row.setVisible(True)

    def _sync_spin_from_canvas(self):
        """从画布同步角度到SpinBox（防止循环触发）"""
        self._rotate_spin_updating = True
        self._rotate_spin.setValue(self.canvas.rotation_angle)
        self._rotate_spin_updating = False

    # ── 留白预览 ──

    def _on_margin_changed(self):
        """留白值变化时实时更新画布预览"""
        self.canvas.set_margin_preview(
            self._spin_top.value(),
            self._spin_bottom.value(),
            self._spin_left.value(),
            self._spin_right.value(),
        )

    def _on_feather_changed(self):
        """羽化参数变化时实时更新画布预览"""
        self.canvas.set_feather_preview(
            self._chk_feather.isChecked(),
            self._spin_feather.value(),
        )

    # ── 键盘事件 ──

    def keyPressEvent(self, event):
        """处理键盘快捷键"""
        # Ctrl+Z：橡皮擦撤销
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            if self._btn_eraser.isChecked():
                self.canvas.eraser_undo()
                event.accept()
                return
        super().keyPressEvent(event)

    # ── 确定逻辑 ──

    def _on_ok(self):
        """处理流程：直接使用当前画布图片 → 留白 → 羽化
        （裁切已通过 ✓ 按钮实时应用到画布，旋转也已实时应用）"""
        # Step 1: 获取当前画布图片（已包含所有已应用的裁切和旋转）
        # 如果还有未确认的裁切框，也应用它
        cropped = self.canvas.get_cropped_image()
        if cropped is not None:
            img = cropped.convert("RGBA")
        else:
            img = self.canvas.pil_img.convert("RGBA")

        # Step 2: 边缘留白（外扩）
        mt = self._spin_top.value()
        mb = self._spin_bottom.value()
        ml = self._spin_left.value()
        mr = self._spin_right.value()

        if mt > 0 or mb > 0 or ml > 0 or mr > 0:
            new_w = img.width + ml + mr
            new_h = img.height + mt + mb
            canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            canvas.paste(img, (ml, mt))
            img = canvas

        # Step 3: 羽化（如果开启且有留白）
        if self._chk_feather.isChecked() and (mt > 0 or mb > 0 or ml > 0 or mr > 0):
            feather_px = self._spin_feather.value()
            img = self._apply_feather(img, feather_px, mt, mb, ml, mr)

        self.result_img = img
        self.accept()

    @staticmethod
    def _apply_feather(img: Image.Image, feather_px: int,
                       top: int, bottom: int, left: int, right: int) -> Image.Image:
        """对原图内容区域的边缘做 Alpha 线性渐变羽化"""
        if feather_px <= 0:
            return img
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # 原图内容区域边界
        ct = top
        cb = h - bottom if bottom > 0 else h
        cl = left
        cr = w - right if right > 0 else w

        # 顶部边缘
        if top > 0:
            f = min(feather_px, cb - ct)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[ct + i, cl:cr, 3] *= g[i]

        # 底部边缘
        if bottom > 0:
            f = min(feather_px, cb - ct)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[cb - 1 - i, cl:cr, 3] *= g[i]

        # 左侧边缘
        if left > 0:
            f = min(feather_px, cr - cl)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[ct:cb, cl + i, 3] *= g[i]

        # 右侧边缘
        if right > 0:
            f = min(feather_px, cr - cl)
            if f > 0:
                g = np.linspace(0, 1, f, dtype=np.float32)
                for i in range(f):
                    arr[ct:cb, cr - 1 - i, 3] *= g[i]

        return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")
