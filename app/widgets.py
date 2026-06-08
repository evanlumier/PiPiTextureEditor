# -*- coding: utf-8 -*-
"""
widgets.py - 通用 UI 控件

从 Texture_tool_GUI_with_tabs.py 拆分出的独立 UI 控件：
- DropLabel: 支持拖拽/点击导入图片的标签
- CheckerLabel: 带棋盘格透明背景的预览控件（支持缩放、平移、拖拽导入）
- StackedTextTabBar: 自定义左侧 TabBar（文字竖排堆叠显示）
"""

import os
from typing import Optional, Callable

from PySide6.QtCore import Qt, QRectF, QPointF, QSize
from PySide6.QtGui import (
    QPixmap,
    QPainter,
    QPainterPath,
    QPen,
    QColor,
    QFontMetrics,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QFileDialog,
    QTabBar,
    QStyleOptionTab,
    QStyle,
)

from tab_transfer import RIGHT_CLICK_THRESHOLD

# ========= 兼容：不同 PySide6 版本 QStylePainter 所在模块不同 =========
try:
    from PySide6.QtWidgets import QStylePainter
except Exception:
    from PySide6.QtGui import QStylePainter


class DropLabel(QLabel):
    """支持拖拽/点击导入图片的标签控件"""

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
            urls = event.mimeData().urls()
            if urls:
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.webp'):
                    event.acceptProposedAction()
                    self.setStyleSheet(
                        "border:2px dashed #89b4fa;border-radius:10px;padding:20px;"
                        "background:rgba(137,180,250,0.08);color:#89b4fa;font-size:13px;"
                    )
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            "border:2px dashed #45475a;border-radius:10px;padding:20px;"
            "background:transparent;color:#6c7086;font-size:13px;"
        )

    def dropEvent(self, event):
        self.setStyleSheet(
            "border:2px dashed #45475a;border-radius:10px;padding:20px;"
            "background:transparent;color:#6c7086;font-size:13px;"
        )
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.webp'):
                self.on_drop_callback(path)


class CheckerLabel(QWidget):
    """带棋盘格透明背景的预览控件，支持滚轮缩放、右键拖拽平移、拖拽/点击导入"""

    def __init__(self, cell=12, color1=None, color2=None, parent=None):
        super().__init__(parent)
        self.cell = cell
        self.color1 = color1 or QColor(42, 42, 58)
        self.color2 = color2 or QColor(30, 30, 46)
        self._hovered = False
        self._drag_hovering = False  # 拖拽悬停状态
        self._on_drop_callback = None  # 拖拽/点击导入回调
        self._parent_window = None  # 用于弹出文件对话框
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        # ── 缩放 & 平移状态 ──
        self._source_pix: Optional[QPixmap] = None  # 原始 pixmap
        self._scale: float = 1.0       # 当前缩放倍率
        self._offset = QPointF(0, 0)   # 图片左上角在控件中的偏移
        self._fit_done: bool = False    # 是否已执行过自适应

        # ── 右键拖拽 & 单击区分 ──
        self._dragging: bool = False
        self._last_mouse = QPointF()
        self._right_press_pos: Optional[QPointF] = None  # 右键按下位置
        self._right_click_callback: Optional[Callable] = None  # 右键单击回调

    # ── 兼容 QLabel 接口 ──────────────────────────────────────────────
    def setPixmap(self, pix: QPixmap):
        """设置要显示的 pixmap，仅首次或图片尺寸变化时自动适配窗口"""
        old = self._source_pix
        self._source_pix = pix
        # 仅在首次设置或图片尺寸发生变化时才重置缩放
        if old is None or old.isNull() or old.size() != pix.size():
            self._fit_to_view()
        self.update()

    def pixmap(self) -> Optional[QPixmap]:
        return self._source_pix

    def clear(self):
        self._source_pix = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._fit_done = False
        self.update()

    def setAlignment(self, *args):
        pass  # 兼容旧调用，不需要实际处理

    # ── 自适应窗口 ──────────────────────────────────────────────────
    def _fit_to_view(self):
        """让图片完整显示在控件中（适配窗口）"""
        pm = self._source_pix
        if pm is None or pm.isNull():
            return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        pw, ph = pm.width(), pm.height()
        if pw <= 0 or ph <= 0:
            return
        self._scale = min(w / pw, h / ph)
        # 居中
        disp_w = pw * self._scale
        disp_h = ph * self._scale
        self._offset = QPointF((w - disp_w) / 2, (h - disp_h) / 2)
        self._fit_done = True

    def _clamp_offset(self):
        """限制图片不能完全拖出预览框（至少保留 20% 可见）"""
        pm = self._source_pix
        if pm is None or pm.isNull():
            return
        pw = pm.width() * self._scale
        ph = pm.height() * self._scale
        w, h = self.width(), self.height()
        # 至少保留 20% 的图片在可视区域内
        margin_x = pw * 0.2
        margin_y = ph * 0.2
        min_x = -(pw - margin_x)
        max_x = w - margin_x
        min_y = -(ph - margin_y)
        max_y = h - margin_y
        ox = max(min_x, min(max_x, self._offset.x()))
        oy = max(min_y, min(max_y, self._offset.y()))
        self._offset = QPointF(ox, oy)

    def _pixmap_rect(self):
        """计算当前 pixmap 在控件中的显示矩形"""
        pm = self._source_pix
        if pm is None or pm.isNull():
            return None
        pw = pm.width() * self._scale
        ph = pm.height() * self._scale
        return QRectF(self._offset.x(), self._offset.y(), pw, ph)

    # ── 鼠标事件 ──────────────────────────────────────────────────────
    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        if not self._dragging:
            self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # 右键：记录按下位置，延迟判断是单击还是拖拽
        if event.button() == Qt.RightButton and self._source_pix is not None:
            self._right_press_pos = event.position()
            self._dragging = False
            self._last_mouse = event.position()
            return
        # 左键点击导入（仅在没有图片时）
        if event.button() == Qt.LeftButton and self._on_drop_callback is not None:
            if self._source_pix is None or self._source_pix.isNull():
                parent = self._parent_window or self
                path, _ = QFileDialog.getOpenFileName(
                    parent, "选择图片", "", "Images (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
                )
                if path:
                    self._on_drop_callback(path)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 右键移动：检查是否超过阈值进入拖拽
        if self._right_press_pos is not None and (event.buttons() & Qt.RightButton):
            if not self._dragging:
                delta = event.position() - self._right_press_pos
                dist = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
                if dist >= RIGHT_CLICK_THRESHOLD:
                    self._dragging = True
                    self.setCursor(Qt.ClosedHandCursor)
                    self._last_mouse = event.position()
                return  # 阈值内不做任何事
            # 已进入拖拽模式，执行平移
            delta = event.position() - self._last_mouse
            self._offset = QPointF(self._offset.x() + delta.x(),
                                   self._offset.y() + delta.y())
            self._clamp_offset()
            self._last_mouse = event.position()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            was_dragging = self._dragging
            self._dragging = False
            self._right_press_pos = None
            self.setCursor(Qt.ArrowCursor)
            if not was_dragging:
                # 右键单击（未拖拽）→ 弹出发送菜单
                if self._right_click_callback is not None:
                    self._right_click_callback(event.globalPosition().toPoint())
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """滚轮缩放，以鼠标位置为缩放中心"""
        pm = self._source_pix
        if pm is None or pm.isNull():
            return
        pos = event.position()
        old_scale = self._scale
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_scale = max(0.02, min(old_scale * factor, 30.0))
        # 以鼠标位置为缩放中心
        self._offset = QPointF(
            pos.x() - (pos.x() - self._offset.x()) * new_scale / old_scale,
            pos.y() - (pos.y() - self._offset.y()) * new_scale / old_scale,
        )
        self._scale = new_scale
        self._clamp_offset()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """双击左键适配窗口"""
        if event.button() == Qt.LeftButton and self._source_pix is not None:
            self._fit_to_view()
            self.update()

    # ── 拖拽导入 ──────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if self._on_drop_callback is not None and event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.webp'):
                    event.acceptProposedAction()
                    self._drag_hovering = True
                    self.update()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_hovering = False
        self.update()

    def dropEvent(self, event):
        self._drag_hovering = False
        self.update()
        if self._on_drop_callback is None:
            return
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.webp'):
                self._on_drop_callback(path)

    def resizeEvent(self, event):
        old_size = event.oldSize()
        new_size = event.size()
        super().resizeEvent(event)
        # 窗口大小变化时，按比例调整偏移量以保持视觉中心不变
        if self._source_pix is not None and not self._source_pix.isNull():
            if old_size.width() > 0 and old_size.height() > 0:
                # 计算旧视口中心对应的图片坐标，在新视口中保持该点居中
                old_cx = old_size.width() / 2.0
                old_cy = old_size.height() / 2.0
                new_cx = new_size.width() / 2.0
                new_cy = new_size.height() / 2.0
                self._offset = QPointF(
                    self._offset.x() + (new_cx - old_cx),
                    self._offset.y() + (new_cy - old_cy),
                )
                self._clamp_offset()
            else:
                self._fit_to_view()

    # ── 绘制 ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # 圆角裁剪（配合 stylesheet 的 border-radius）
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 10, 10)
        p.setClipPath(path)

        # 棋盘格背景
        cell = self.cell
        cols = self.width() // cell + 1
        rows = self.height() // cell + 1
        for r in range(rows):
            for c in range(cols):
                color = self.color1 if (r + c) % 2 == 0 else self.color2
                p.fillRect(c * cell, r * cell, cell, cell, color)

        pm = self._source_pix
        has_image = pm is not None and not pm.isNull()

        # 绘制图片
        if has_image:
            pix_rect = self._pixmap_rect()
            if pix_rect:
                p.drawPixmap(pix_rect, pm, QRectF(0, 0, pm.width(), pm.height()))

        # 没有图片时显示拖拽提示
        if not has_image and self._on_drop_callback is not None:
            p.setRenderHint(QPainter.Antialiasing)
            # 拖拽悬停时高亮边框
            if self._drag_hovering:
                border_pen = QPen(QColor(137, 180, 250))  # #89b4fa
                border_pen.setWidth(2)
                border_pen.setStyle(Qt.DashLine)
                p.setPen(border_pen)
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 10, 10)
                p.setPen(QColor(137, 180, 250))
            else:
                # 虚线边框
                border_pen = QPen(QColor(69, 71, 90))  # #45475a
                border_pen.setWidth(2)
                border_pen.setStyle(Qt.DashLine)
                p.setPen(border_pen)
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 10, 10)
                p.setPen(QColor(108, 112, 134))  # #6c7086
            font = p.font()
            font.setPixelSize(14)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignCenter, "拖拽图片到此处导入\n或点击此区域选择文件")
            # 有提示时显示手型光标
            self.setCursor(Qt.PointingHandCursor)
        elif not has_image:
            self.setCursor(Qt.ArrowCursor)
        else:
            # 有图片时恢复默认光标
            if not self._dragging:
                self.setCursor(Qt.ArrowCursor)

        # hover 时绘制图片边界指引线
        if self._hovered and has_image:
            pix_rect = self._pixmap_rect()
            if pix_rect:
                p.setRenderHint(QPainter.Antialiasing, False)
                border_pen = QPen(QColor(90, 90, 106))  # #5a5a6a
                border_pen.setWidth(1)
                p.setPen(border_pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(pix_rect.toRect())

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
