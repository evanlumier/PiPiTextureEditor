# -*- coding: utf-8 -*-
import os
import re
import math
from typing import List, Optional, Callable, Tuple, Set

from PIL import Image

from PySide6.QtCore import Qt, QTimer, QPoint, QRect, QByteArray, QMimeData, QRegularExpression
from PySide6.QtGui import (
    QRegularExpressionValidator,
    QPixmap, QImage, QPainter, QPen, QColor, QDrag, QFontMetrics
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QMessageBox, QGroupBox, QComboBox, QSplitter,
    QCheckBox, QSpinBox, QAbstractItemView, QLineEdit
)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp"}

MIME_LIST_MULTI = "application/x-gui-texture-editor-list-multi"
MIME_CELL_MULTI = "application/x-gui-texture-editor-cell-multi"


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


def _make_drag_ghost(names: List[str], *, max_lines: int = 10) -> QPixmap:
    """
    生成“虚影”拖拽提示：白底半透明卡片 + 细边框 + 黑字（多行显示选中项名字）。
    为了让它显示在鼠标右侧，pixmap 左侧留透明 padding。
    """
    names = [n for n in names if n]
    if not names:
        names = ["(空)"]

    extra = 0
    if len(names) > max_lines:
        extra = len(names) - max_lines
        names = names[:max_lines]
        names.append(f"... +{extra}")

    # 画布参数
    left_pad = 18  # 透明区：鼠标热点落在这里 => 内容在鼠标右侧
    pad_x = 12
    pad_y = 10
    line_gap = 3
    radius = 10

    fm = QFontMetrics(QLabel().font())
    text_w = max(fm.horizontalAdvance(s) for s in names)
    line_h = fm.height()

    card_w = pad_x * 2 + text_w
    card_h = pad_y * 2 + len(names) * line_h + (len(names) - 1) * line_gap

    w = left_pad + card_w
    h = card_h

    pm = QPixmap(max(1, w), max(1, h))
    pm.fill(Qt.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    card_rect = QRect(left_pad, 0, card_w, card_h)

    # 深色卡片底
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(30, 30, 46, 220))
    p.drawRoundedRect(card_rect, radius, radius)

    # 细描边
    pen = QPen(QColor(69, 71, 90, 220))
    pen.setWidth(1)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(card_rect, radius, radius)

    # 文字
    p.setPen(QColor(205, 214, 244, 230))
    x = left_pad + pad_x
    y = pad_y + fm.ascent()
    for i, s in enumerate(names):
        p.drawText(x, y + i * (line_h + line_gap), s)

    p.end()
    return pm


class MultiDropLabel(QLabel):
    def __init__(self, on_drop_paths: Callable[[List[str]], None]):
        super().__init__()
        self.on_drop_paths = on_drop_paths
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("拖拽多张图片到这里\n或点击右侧「导入多张图片」")
        self.setStyleSheet(
            "border:2px dashed #45475a; border-radius:10px; padding:18px;"
            "background: transparent; color:#6c7086; font-weight:700;"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths: List[str] = []
        for u in urls:
            p = u.toLocalFile()
            if p and os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in SUPPORTED_EXTS:
                    paths.append(p)
        if paths:
            self.on_drop_paths(paths)


class ReorderListWidget(QListWidget):
    """
    支持：
    - Shift/CTRL 多选
    - 多选拖拽时：鼠标右侧显示选中项名字“虚影”
    - drop indicator 插入线（Qt 自带）
    - 多选拖拽排序：整体搬移
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropMode(QAbstractItemView.DragDrop)

        # 用自绘插入线（我们的自定义 MIME 会绕开 Qt 默认 dropIndicator 绘制）
        self._drop_line_y: Optional[int] = None

        # 延迟单击清空多选：按下时先记录，松开时（非拖拽）再清空
        self._pending_deselect_item = None   # 等待在 mouseRelease 时清空多选
        self._drag_started = False           # 是否已经触发了拖拽
        self._press_pos = QPoint()           # 记录按下位置，用于拖拽阈值判断

    def mousePressEvent(self, event):
        self._drag_started = False
        self._pending_deselect_item = None

        if event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier:
            it = self.itemAt(event.pos())
            # 点击的是已选中的 item，且当前是多选状态 => 先不清空，等 mouseRelease 再决定
            if it is not None and it.isSelected() and len(self.selectedItems()) > 1:
                self._pending_deselect_item = it
                self._press_pos = event.pos()
                # 不调用 super()，阻止 Qt 清空多选
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 如果有待处理的延迟清空，且鼠标移动超过拖拽阈值，说明用户在拖拽，不清空多选
        if self._pending_deselect_item is not None:
            if (event.buttons() & Qt.LeftButton):
                dist = (event.pos() - self._press_pos).manhattanLength()
                if dist >= 8:
                    # 超过拖拽阈值，直接触发拖拽（多选整体）
                    self._drag_started = True
                    self._pending_deselect_item = None
                    self.startDrag(Qt.MoveAction)
                    return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # 如果是单击（没有拖拽），此时才清空多选，只保留点击的那个 item
        if event.button() == Qt.LeftButton and self._pending_deselect_item is not None and not self._drag_started:
            it = self._pending_deselect_item
            self._pending_deselect_item = None
            self.clearSelection()
            it.setSelected(True)
            self.setCurrentItem(it)
            event.accept()
            return
        self._pending_deselect_item = None
        super().mouseReleaseEvent(event)


    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return

        # 生成拖拽虚影（显示文件名）
        names = [it.text() for it in items]
        ghost = _make_drag_ghost(names, max_lines=12)

        drag = QDrag(self)
        mime = QMimeData()

        # 记录被拖拽的 row 列表
        rows = sorted({self.row(it) for it in items})
        mime.setData(MIME_LIST_MULTI, QByteArray(",".join(map(str, rows)).encode("utf-8")))
        drag.setMimeData(mime)

        drag.setPixmap(ghost)
        # hotspot 放在左侧透明 padding 区中间 => 内容在鼠标右侧
        drag.setHotSpot(QPoint(8, min(ghost.height() // 2, 16)))
        drag.exec(Qt.MoveAction)

    def dragEnterEvent(self, event):
        # 接受我们自定义的多选拖拽 mime
        if event.source() is self and event.mimeData().hasFormat(MIME_LIST_MULTI):
            event.setDropAction(Qt.MoveAction)
            event.accept()
            return
        # 拒绝其他来源（外部文件等），避免 DragDrop 模式下误插入
        event.ignore()

    def dragMoveEvent(self, event):
        if event.source() is self and event.mimeData().hasFormat(MIME_LIST_MULTI):
            event.setDropAction(Qt.MoveAction)
            event.accept()

            # 计算插入线位置
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            it = self.itemAt(pos)
            if it is None:
                self._drop_line_y = self.viewport().rect().bottom()
            else:
                r = self.visualItemRect(it)
                mid = r.top() + r.height() // 2
                self._drop_line_y = r.top() if pos.y() < mid else r.bottom()
            self.viewport().update()
            return

        self._drop_line_y = None
        self.viewport().update()
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self._drop_line_y = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_line_y is None:
            return
        p = QPainter(self.viewport())
        pen = QPen(QColor(137, 180, 250, 220))
        pen.setWidth(2)
        p.setPen(pen)
        y = int(self._drop_line_y)
        p.drawLine(4, y, self.viewport().width() - 4, y)
        p.end()

    def dropEvent(self, event):
        self._drop_line_y = None
        self.viewport().update()
        # 多选整体搬移
        if event.mimeData().hasFormat(MIME_LIST_MULTI):
            data = bytes(event.mimeData().data(MIME_LIST_MULTI)).decode("utf-8", "ignore").strip()
            try:
                src_rows = [int(x) for x in data.split(",") if x.strip() != ""]
            except Exception:
                src_rows = []
            src_rows = sorted(set([r for r in src_rows if 0 <= r < self.count()]))
            if not src_rows:
                return super().dropEvent(event)

            # 目标插入位置：区分上下半区，精确计算插入到 item 之前还是之后
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            dst_item = self.itemAt(pos)
            if dst_item is None:
                # 鼠标在列表空白区域，插到末尾
                dst_row = self.count()
            else:
                item_row = self.row(dst_item)
                r = self.visualItemRect(dst_item)
                mid = r.top() + r.height() // 2
                # 鼠标在 item 上半部分 => 插入到该 item 之前；下半部分 => 插入到该 item 之后
                if pos.y() < mid:
                    dst_row = item_row
                else:
                    dst_row = item_row + 1

            # 如果落点在被拖拽区间内部（无变化），退出
            if min(src_rows) < dst_row <= max(src_rows) + 1:
                event.acceptProposedAction()
                return

            # 取出 items（从后往前取，避免 row 偏移）
            taken: List[QListWidgetItem] = []
            for r in reversed(src_rows):
                taken.append(self.takeItem(r))
            taken.reverse()

            # 计算删除后的 dst_row
            removed_before = sum(1 for r in src_rows if r < dst_row)
            dst_row = max(0, min(self.count(), dst_row - removed_before))

            # 插入
            for i, it in enumerate(taken):
                self.insertItem(dst_row + i, it)

            # 恢复选中
            self.clearSelection()
            for i in range(len(taken)):
                self.item(dst_row + i).setSelected(True)
            self.setCurrentRow(dst_row)

            event.acceptProposedAction()
            # 通知外部同步 paths 数据并重建精灵图
            QTimer.singleShot(0, self._notify_reordered)
            return
        return super().dropEvent(event)

    def _notify_reordered(self):
        """拖拽完成后延迟触发，通知外部数据已重排。"""
        self.model().layoutChanged.emit()


class SheetPreviewLabel(QLabel):
    """
    显示 SpriteSheet：
    - 永久淡灰色 cell 边框（一直显示）
    - 点击/Shift/CTRL 多选（与左侧列表联动）
    - 多选拖拽：鼠标右侧显示选中项名字“虚影”
    - 支持在 sheet 里拖拽排序（多选整体搬移）
    - 选中 cell 加粗高亮框
    - 拖拽时显示“插入落点蓝线”（两图之间），兼容单选/多选
    """
    def __init__(
        self,
        on_reorder_multi: Callable[[List[int], int], None],
        get_name: Callable[[int], str],
        on_selection_changed: Optional[Callable[[Set[int], Optional[int]], None]] = None,
    ):
        super().__init__()
        self.on_reorder_multi = on_reorder_multi
        self.get_name = get_name
        self.on_selection_changed = on_selection_changed

        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: transparent; border: none;")

        self._pix: Optional[QPixmap] = None
        self._cols = 1
        self._rows = 1
        self._count = 0

        self._selected: Set[int] = set()
        self._anchor: Optional[int] = None  # shift anchor
        self._drag_from: Optional[int] = None
        self._press_pos = QPoint()

        # 插入落点（0..count），None 表示不显示
        self._drop_insert_index: Optional[int] = None

        # 选中描边颜色（现代蓝，随你改）
        self.sel_color = QColor(137, 180, 250, 230)

    def set_sheet(self, pix: Optional[QPixmap], cols: int, rows: int, count: int):
        self._pix = pix
        self._cols = max(1, int(cols))
        self._rows = max(1, int(rows))
        self._count = max(0, int(count))
        # 清理越界选中
        self._selected = {i for i in self._selected if 0 <= i < self._count}
        if self._anchor is not None and not (0 <= self._anchor < self._count):
            self._anchor = None
        self._drop_insert_index = None
        self.update()

    def set_selected_set(self, s: Set[int], anchor: Optional[int] = None):
        self._selected = {i for i in s if 0 <= i < self._count}
        self._anchor = anchor if (anchor is None or 0 <= anchor < self._count) else None
        self.update()

    def selected_set(self) -> Set[int]:
        return set(self._selected)

    def _pix_rect(self) -> Tuple[QRect, Optional[QPixmap]]:
        if self._pix is None or self._pix.isNull():
            return QRect(), None
        scaled = self._pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height()), scaled

    def _pos_to_index(self, pos: QPoint) -> Optional[int]:
        rect, _ = self._pix_rect()
        if rect.isNull() or not rect.contains(pos):
            return None
        rel_x = pos.x() - rect.x()
        rel_y = pos.y() - rect.y()
        col = int(rel_x * self._cols / max(1, rect.width()))
        row = int(rel_y * self._rows / max(1, rect.height()))
        col = max(0, min(self._cols - 1, col))
        row = max(0, min(self._rows - 1, row))
        idx = row * self._cols + col
        if idx >= self._count:
            return None
        return idx

    def _pos_to_insert_index(self, pos: QPoint) -> Optional[int]:
        """
        返回插入位置（0..count），表示将拖拽内容插入到这个索引之前。
        规则：落在某个 cell 左半边 => idx；右半边 => idx+1
        """
        rect, _ = self._pix_rect()
        if rect.isNull() or not rect.contains(pos) or self._count <= 0:
            return None

        cell_w = rect.width() / self._cols
        cell_h = rect.height() / self._rows

        rel_x = pos.x() - rect.x()
        rel_y = pos.y() - rect.y()
        col = int(rel_x / max(1e-6, cell_w))
        row = int(rel_y / max(1e-6, cell_h))
        col = max(0, min(self._cols - 1, col))
        row = max(0, min(self._rows - 1, row))

        idx = row * self._cols + col
        if idx >= self._count:
            idx = self._count - 1

        # 判断左右半边
        within_x = rel_x - col * cell_w
        if within_x >= cell_w * 0.5:
            ins = idx + 1
        else:
            ins = idx
        ins = max(0, min(self._count, ins))
        return ins

    def _insertion_line(self, insert_index: int) -> Optional[Tuple[QPoint, QPoint]]:
        """
        根据插入位置计算一条“蓝色提示线”段（两图之间）。
        简化为：竖线，画在目标 cell 的左边界 / 或最后一个 cell 的右边界。
        """
        rect, _ = self._pix_rect()
        if rect.isNull() or self._count <= 0:
            return None

        cell_w = rect.width() / self._cols
        cell_h = rect.height() / self._rows

        # clamp
        insert_index = max(0, min(self._count, insert_index))

        if insert_index == 0:
            row, col = 0, 0
            x = rect.x()
            y0 = rect.y()
            y1 = rect.y() + int(cell_h)
            return QPoint(x, y0), QPoint(x, y1)

        if insert_index >= self._count:
            last = self._count - 1
            row = last // self._cols
            col = last % self._cols
            x = rect.x() + int((col + 1) * cell_w)
            y0 = rect.y() + int(row * cell_h)
            y1 = y0 + int(cell_h)
            return QPoint(x, y0), QPoint(x, y1)

        # insert before this index
        row = insert_index // self._cols
        col = insert_index % self._cols
        x = rect.x() + int(col * cell_w)
        y0 = rect.y() + int(row * cell_h)
        y1 = y0 + int(cell_h)
        return QPoint(x, y0), QPoint(x, y1)

    def paintEvent(self, event):
        super().paintEvent(event)
        rect, scaled = self._pix_rect()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        if scaled is not None:
            p.drawPixmap(rect.x(), rect.y(), scaled)

            cell_w = rect.width() / self._cols
            cell_h = rect.height() / self._rows

            # 永久淡灰细框
            pen = QPen(QColor(80, 80, 110, 120))
            pen.setWidth(1)
            p.setPen(pen)
            for idx in range(self._count):
                row = idx // self._cols
                col = idx % self._cols
                x = rect.x() + int(col * cell_w)
                y = rect.y() + int(row * cell_h)
                p.drawRect(QRect(x, y, int(cell_w), int(cell_h)))

            # 多选高亮框（加粗）
            if self._selected:
                sel_pen = QPen(self.sel_color)
                sel_pen.setWidth(3)
                p.setPen(sel_pen)
                for idx in sorted(self._selected):
                    if 0 <= idx < self._count:
                        row = idx // self._cols
                        col = idx % self._cols
                        x = rect.x() + int(col * cell_w)
                        y = rect.y() + int(row * cell_h)
                        p.drawRect(QRect(x, y, int(cell_w), int(cell_h)))

            # 拖拽落点提示线（两图之间）
            if self._drop_insert_index is not None:
                seg = self._insertion_line(self._drop_insert_index)
                if seg is not None:
                    a, b = seg
                    drop_pen = QPen(QColor(137, 180, 250, 230))
                    drop_pen.setWidth(2)
                    p.setPen(drop_pen)
                    p.drawLine(a, b)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
            idx = self._pos_to_index(event.pos())

            mods = event.modifiers()
            if idx is None:
                if not (mods & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self._selected.clear()
                    self._anchor = None
                self.update()
                self._drag_from = None
                super().mousePressEvent(event)
                return

            if mods & Qt.ShiftModifier:
                if self._anchor is None:
                    self._anchor = idx
                a = self._anchor
                lo, hi = sorted((a, idx))
                self._selected = set(range(lo, hi + 1))
            elif mods & Qt.ControlModifier:
                if idx in self._selected:
                    self._selected.remove(idx)
                else:
                    self._selected.add(idx)
                self._anchor = idx
            else:
                # 如果点击的是已选中的其中一个，并且当前是多选状态，
                # 则保持多选集合不被“单击清空”，以便直接拖拽多选排序
                if idx in self._selected and len(self._selected) > 1:
                    self._anchor = idx
                else:
                    self._selected = {idx}
                    self._anchor = idx

            self._drag_from = idx
            self.update()
            if self.on_selection_changed is not None:
                self.on_selection_changed(set(self._selected), self._anchor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 启动拖拽：用“名字虚影”提示
        if (event.buttons() & Qt.LeftButton) and self._drag_from is not None:
            if (event.pos() - self._press_pos).manhattanLength() >= 8:
                # 如果有多选，就优先拖多选集合
                if self._selected and (len(self._selected) > 1 or self._drag_from in self._selected):
                    drag_indices = sorted(self._selected)
                    names = [self.get_name(i) for i in drag_indices]
                else:
                    drag_indices = [self._drag_from]
                    names = [self.get_name(self._drag_from)]

                ghost = _make_drag_ghost(names, max_lines=12)

                drag = QDrag(self)
                mime = QMimeData()
                mime.setData(MIME_CELL_MULTI, QByteArray(",".join(map(str, drag_indices)).encode("utf-8")))
                drag.setMimeData(mime)

                drag.setPixmap(ghost)
                drag.setHotSpot(QPoint(8, min(ghost.height() // 2, 16)))
                drag.exec(Qt.MoveAction)

                self._drag_from = None

        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_CELL_MULTI):
            self._drop_insert_index = None
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME_CELL_MULTI):
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            ins = self._pos_to_insert_index(pos)
            self._drop_insert_index = ins
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_insert_index = None
        self.update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(MIME_CELL_MULTI):
            return
        data = bytes(event.mimeData().data(MIME_CELL_MULTI)).decode("utf-8", "ignore").strip()
        try:
            src = [int(x) for x in data.split(",") if x.strip() != ""]
        except Exception:
            src = []
        src = sorted(set([i for i in src if 0 <= i < self._count]))
        if not src:
            self._drop_insert_index = None
            self.update()
            return

        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        ins = self._pos_to_insert_index(pos)
        if ins is None:
            self._drop_insert_index = None
            self.update()
            return

        self._drop_insert_index = None
        self.update()

        # 回调：用“插入位置”而不是 cell index
        self.on_reorder_multi(src, ins)
        event.acceptProposedAction()
class SpriteSheetTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.paths: List[str] = []
        self.frames_rgba: List[Image.Image] = []
        self.sheet_rgba: Optional[Image.Image] = None

        self.sheet_cols = 1
        self.sheet_rows = 1

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick_preview)
        self.preview_index = 0

        self._syncing = False  # 防止 list<->sheet 互相触发
        self._undo_stack: List[List[str]] = []  # 拖拽排序撤销栈（最多保留20步）

        # -------- 一键命名（精灵图页签，独立记忆） --------
        self.output_basename: Optional[str] = None
        self._last_name: Optional[str] = None

        self._build_ui()

        # 启动时加载历史（精灵图页签独立）
        last = self._load_last_name()
        if last:
            self._last_name = last
            self._build_history_buttons(last)
        self._update_name_preview()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # 精灵图页签局部样式（覆盖继承）
        _sprite_style = """
            QSpinBox {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px 22px 4px 8px;
                min-height: 26px;
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
            QSpinBox::up-button:disabled, QSpinBox::down-button:disabled {
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
            /* 外部SpinBox箭头按钮 */
            QPushButton#spinArrowBtn {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QPushButton#spinArrowBtn:hover {
                background-color: #313244;
                border-radius: 3px;
            }
            QPushButton#spinArrowBtn:pressed {
                background-color: #45475a;
            }
            QComboBox {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 2px 30px 2px 8px;
                min-height: 18px;
            }
            QComboBox:hover { border-color: #89b4fa; }
            QComboBox::drop-down {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                background-color: transparent;
            }
            QComboBox::drop-down:hover { background-color: transparent; }
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
                image: none;
            }
            QCheckBox::indicator:hover {
                border-color: #89b4fa;
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
        _sprite_style = _sprite_style.replace("__COMBO_DN_ARROW__", _tmp_dn.name.replace("\\", "/"))
        self.setStyleSheet(_sprite_style)

        # Left
        left_col = QVBoxLayout()
        self.drop_area = MultiDropLabel(self.add_paths)

        self.imported_title = QLabel("已导入文件：")
        self.imported_title.setStyleSheet("font-weight:700; color:#89b4fa; padding:4px 2px;")

        self.list_widget = ReorderListWidget()
        self.list_widget.setMinimumWidth(230)
        self.list_widget.setFocusPolicy(Qt.StrongFocus)
        # 插入线蓝色（Qt 不同版本的 dropIndicator 选择器不同，这里双写）
        self.list_widget.setStyleSheet(
            "background: #181825; border: 1px solid #383850; border-radius:8px;"
            "color: #cdd6f4; outline: none; padding: 4px;"
        )
        self.list_widget.itemSelectionChanged.connect(self._on_list_selection_changed)
        self.list_widget.installEventFilter(self)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._on_list_context_menu)
        # rowsMoved 对 Qt 内部 move 并不总触发，这里仍保留，但我们也在 selection/rebuild 时保持一致
        self.list_widget.model().rowsMoved.connect(lambda *args: self._on_list_reordered())
        self.list_widget.model().layoutChanged.connect(lambda *args: self._on_list_reordered())

        left_col.addWidget(self.drop_area, 2)
        left_col.addWidget(self.imported_title, 0)
        left_col.addWidget(self.list_widget, 5)

        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: #3b2a2a; color: #f38ba8; border: 1px solid #5a3a3a; border-radius: 6px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #5a3a3a; border-color: #f38ba8; }"
            "QPushButton:pressed { background-color: #f38ba8; color: #1e1e2e; }"
            "QPushButton:disabled { background-color: #2a2a3a; color: #585b70; border-color: #383850; }"
        )
        left_col.addWidget(self.btn_delete, 0)

        left_wrap = QWidget()
        left_wrap.setLayout(left_col)

        # Center
        center_col = QVBoxLayout()

        self.sheet_title = QLabel("精灵图预览")
        self.sheet_title.setStyleSheet("font-weight:700; color:#89b4fa; padding:4px 2px;")

        self.sheet_label = SheetPreviewLabel(
            self._reorder_by_cell_drag_multi,
            self._cell_name,
            self._on_sheet_selection_changed,
        )
        self.sheet_label.setMinimumSize(520, 320)
        self.sheet_label.setFocusPolicy(Qt.StrongFocus)
        self.sheet_label.installEventFilter(self)
        self.sheet_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sheet_label.customContextMenuRequested.connect(self._on_sheet_context_menu)

        self.gif_title = QLabel("GIF预览")
        self.gif_title.setStyleSheet("font-weight:700; color:#89b4fa; padding:4px 2px;")

        self.gif_label = QLabel("")
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setMinimumSize(520, 240)
        self.gif_label.setStyleSheet("background: #181825; border: 1px solid #383850; border-radius:8px;")

        self.btn_refresh_gif = QPushButton("更新GIF预览")
        self.btn_refresh_gif.clicked.connect(self._update_gif_preview)
        self.btn_refresh_gif.setFixedHeight(28)

        center_col.addWidget(self.sheet_title, 0)
        center_col.addWidget(self.sheet_label, 6)
        center_col.addWidget(self.gif_title, 0)
        center_col.addWidget(self.gif_label, 4)
        center_col.addWidget(self.btn_refresh_gif, 0)

        center_wrap = QWidget()
        center_wrap.setLayout(center_col)

        # Right
        right_col = QVBoxLayout()

        grp = QGroupBox("参数 / 导出")
        grp_lay = QVBoxLayout(grp)

        self.btn_add = QPushButton("导入多张图片")
        self.btn_add.clicked.connect(self.open_files)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_all)

        _pow2 = ["32", "64", "128", "256", "512", "1024", "2048"]
        self.size_combo_w = QComboBox()
        self.size_combo_w.setEditable(True)
        self.size_combo_w.addItems(_pow2)
        self.size_combo_w.setCurrentText("1024")
        self.size_combo_w.setFixedWidth(120)

        self.size_combo_h = QComboBox()
        self.size_combo_h.setEditable(True)
        self.size_combo_h.addItems(_pow2)
        self.size_combo_h.setCurrentText("1024")
        self.size_combo_h.setFixedWidth(120)

        self.chk_auto_grid = QCheckBox("自动排列（推荐）")
        self.chk_auto_grid.setChecked(True)
        self.chk_auto_grid.stateChanged.connect(self._on_grid_mode_change)

        rc_row = QHBoxLayout()
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 256)
        self.spin_cols.setValue(3)
        self.spin_cols.setFixedWidth(55)
        self.spin_cols.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_cols.valueChanged.connect(lambda _=0: self._on_cols_changed())

        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 256)
        self.spin_rows.setValue(3)
        self.spin_rows.setFixedWidth(55)
        self.spin_rows.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_rows.valueChanged.connect(lambda _=0: self._on_rows_changed())

        # 列的外部上下按钮
        _up_b64 = b"iVBORw0KGgoAAAANSUhEUgAAABQAAAAMCAYAAABiDJ37AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAA7ElEQVQokeWRz0oCURSHf+fWDZwWhTthnkK5vYC4kXwBoY0vMQrRMBDoddErCC58ANNhFOoFfA33tbrBPdOcdpLQjLStb3s+fucf8O+gU0Ici6oZ3weAj93FIkmoqPJVVXG69M3AcEakGkSqERjOpkvf/PWET5nU85wfIbiSszyKusEeAGzqQvo8t0J4L6DvR7f0VhkYx6Iub3gggrtCUTLq6tefGtpnboPkAcDc7fTs+xkOK49XvhUY3hZC187pTlkYAEQ9/eKc7oCoXjO8Ga9860gQEZqseWhTF5aFlGFTF07WPBSRkw/+o3wBC2hZRH08FNEAAAAASUVORK5CYII="
        _dn_b64 = b"iVBORw0KGgoAAAANSUhEUgAAABQAAAAMCAYAAABiDJ37AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAA5UlEQVQokeWPMUoDURRF7/vM/OJ3WmcTgjuIEAizgSzEOFPIECycb+EWUgpphxkCfwkxuAlLid0E8uK/VmMhicE6p32cw33AeUJSqkZz33aD/7q+7QZVozlJAQADACJCkoExmftGb8sF7alQuaCtap0yJnOSQUT4EwSAIrPr7Sodgdw4p8HXOjy6qtahcxqM8GO7SkdFZtf9TQ4Jz0te6l4fAFzA7Kd3Y/fev4eYPAmx+ZL0vsjk87d7MNjz2O6uTISHSAAAIt5QJC/G9u2Y82cQAMqSxl3vJgDQvdqX2UziKefM+Abk12Ee9AJi7AAAAABJRU5ErkJggg=="
        from PySide6.QtGui import QIcon
        _up_pm = QPixmap(); _up_pm.loadFromData(QByteArray.fromBase64(_up_b64))
        _dn_pm = QPixmap(); _dn_pm.loadFromData(QByteArray.fromBase64(_dn_b64))
        _up_icon = QIcon(_up_pm)
        _dn_icon = QIcon(_dn_pm)

        btn_cols_up = QPushButton()
        btn_cols_up.setFixedSize(20, 14)
        btn_cols_up.setObjectName("spinArrowBtn")
        btn_cols_up.setIcon(_up_icon)
        btn_cols_up.setIconSize(_up_pm.size())
        btn_cols_up.clicked.connect(lambda: self.spin_cols.setValue(self.spin_cols.value() + 1))
        btn_cols_dn = QPushButton()
        btn_cols_dn.setFixedSize(20, 14)
        btn_cols_dn.setObjectName("spinArrowBtn")
        btn_cols_dn.setIcon(_dn_icon)
        btn_cols_dn.setIconSize(_dn_pm.size())
        btn_cols_dn.clicked.connect(lambda: self.spin_cols.setValue(self.spin_cols.value() - 1))
        cols_btn_lay = QVBoxLayout()
        cols_btn_lay.setSpacing(1)
        cols_btn_lay.setContentsMargins(0, 0, 0, 0)
        cols_btn_lay.addWidget(btn_cols_up)
        cols_btn_lay.addWidget(btn_cols_dn)

        # 行的外部上下按钮
        btn_rows_up = QPushButton()
        btn_rows_up.setFixedSize(20, 14)
        btn_rows_up.setObjectName("spinArrowBtn")
        btn_rows_up.setIcon(_up_icon)
        btn_rows_up.setIconSize(_up_pm.size())
        btn_rows_up.clicked.connect(lambda: self.spin_rows.setValue(self.spin_rows.value() + 1))
        btn_rows_dn = QPushButton()
        btn_rows_dn.setFixedSize(20, 14)
        btn_rows_dn.setObjectName("spinArrowBtn")
        btn_rows_dn.setIcon(_dn_icon)
        btn_rows_dn.setIconSize(_dn_pm.size())
        btn_rows_dn.clicked.connect(lambda: self.spin_rows.setValue(self.spin_rows.value() - 1))
        rows_btn_lay = QVBoxLayout()
        rows_btn_lay.setSpacing(1)
        rows_btn_lay.setContentsMargins(0, 0, 0, 0)
        rows_btn_lay.addWidget(btn_rows_up)
        rows_btn_lay.addWidget(btn_rows_dn)

        rc_row.addWidget(QLabel("列："))
        rc_row.addWidget(self.spin_cols)
        rc_row.addLayout(cols_btn_lay)
        rc_row.addSpacing(12)
        rc_row.addWidget(QLabel("行："))
        rc_row.addWidget(self.spin_rows)
        rc_row.addLayout(rows_btn_lay)
        rc_row.addStretch(1)

        self.btn_reset_grid = QPushButton("重置为自动")
        self.btn_reset_grid.clicked.connect(self._reset_grid_to_auto)

        self.btn_export = QPushButton("导出精灵图（PNG）")
        self.btn_export.setStyleSheet(
            "background:#89b4fa; color:#1e1e2e; font-weight:700;"
            "padding:8px; border-radius:7px;"
        )
        self.btn_export.clicked.connect(self.export_sheet)

        grp_lay.addWidget(self.btn_add)
        grp_lay.addWidget(self.btn_clear)
        grp_lay.addSpacing(6)
        grp_lay.addWidget(QLabel("导出尺寸（最终整张图）："))
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("宽："))
        size_row.addWidget(self.size_combo_w)
        size_row.addSpacing(6)
        size_row.addWidget(QLabel("高："))
        size_row.addWidget(self.size_combo_h)
        size_row.addStretch(1)
        grp_lay.addLayout(size_row)
        grp_lay.addSpacing(10)
        grp_lay.addWidget(self.chk_auto_grid)
        grp_lay.addLayout(rc_row)
        grp_lay.addWidget(self.btn_reset_grid)

        right_col.addWidget(grp)
        # -------- 一键命名（精灵图，带记忆） --------
        name_box = QGroupBox("一键命名（精灵图）")
        name_lay = QVBoxLayout(name_box)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入命名 tag（仅 A-Z a-z 0-9 _），导出会自动加前缀 T_")
        self.name_input.setValidator(QRegularExpressionValidator(QRegularExpression("^[A-Za-z0-9_]*$")))
        self.name_input.textChanged.connect(lambda _=None: self._update_name_preview())

        hist_row = QHBoxLayout()
        self.history_btn1 = QPushButton("")
        self.history_btn2 = QPushButton("")
        self.history_btn3 = QPushButton("")
        for btn in (self.history_btn1, self.history_btn2, self.history_btn3):
            btn.setVisible(False)
            btn.clicked.connect(lambda checked=False, b=btn: self._apply_history_name(b.text()))
            hist_row.addWidget(btn)

        self.name_preview = QLabel("预览：-")
        self.name_preview.setStyleSheet("color:#a6e3a1; font-weight:700; padding:2px 0;")

        self.btn_apply_name = QPushButton("应用命名")
        self.btn_apply_name.clicked.connect(self._apply_naming)
        self.btn_reset_name = QPushButton("重置命名")
        self.btn_reset_name.clicked.connect(self._reset_naming)

        name_lay.addWidget(self.name_input)
        name_lay.addLayout(hist_row)
        name_lay.addWidget(self.name_preview)
        name_lay.addWidget(self.btn_apply_name)
        name_lay.addWidget(self.btn_reset_name)

        right_col.addWidget(name_box)

        right_col.addWidget(self.btn_export)

        right_col.addStretch(1)

        right_wrap = QWidget()
        right_wrap.setLayout(right_col)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_wrap)
        splitter.addWidget(center_wrap)
        splitter.addWidget(right_wrap)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)
        splitter.setChildrenCollapsible(False)

        root.addWidget(splitter)

        self._refresh_enabled()
        self._on_grid_mode_change()

    # ---------------- enabled ----------------
    def _refresh_enabled(self):
        has = len(self.frames_rgba) > 0
        self.btn_clear.setEnabled(has)
        self.btn_export.setEnabled(has and self.sheet_rgba is not None)
        if hasattr(self, 'btn_refresh_gif'):
            self.btn_refresh_gif.setEnabled(has)
        self._refresh_delete_btn()

    # ---------------- import ----------------
    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图片", "", "Images (*.png *.jpg *.jpeg *.tga *.bmp *.webp)"
        )
        if paths:
            self.add_paths(paths)

    @staticmethod
    def _natural_sort_key(path: str):
        """自然排序 key：将文件名中的数字部分作为整数比较，支持 1,2,3...10,11 顺序。"""
        name = os.path.basename(path).lower()
        parts = re.split(r'(\d+)', name)
        return [int(p) if p.isdigit() else p for p in parts]

    def add_paths(self, new_paths: List[str]):
        uniq: List[str] = []
        existed = set(self.paths)
        for p in new_paths:
            p = os.path.abspath(p)
            if p not in existed:
                uniq.append(p)
                existed.add(p)

        if not uniq:
            return

        # 按文件名自然排序（支持 1,2,3...10,11 数字顺序）
        uniq.sort(key=self._natural_sort_key)

        self.paths.extend(uniq)
        # 新增图片后重新计算最优行列并同步到 spin（不改变手动/自动模式）
        self._apply_auto_grid_to_spins()
        self._reload_frames_and_rebuild()

    def clear_all(self):
        self.timer.stop()
        self.paths = []
        self.frames_rgba = []
        self.sheet_rgba = None
        self.preview_index = 0

        self.list_widget.clear()
        self.sheet_label.set_sheet(None, 1, 1, 0)
        self.gif_label.clear()
        # 清空后重置行列 spin 的最大值和值
        self.spin_cols.blockSignals(True)
        self.spin_rows.blockSignals(True)
        self.spin_cols.setMaximum(1)
        self.spin_rows.setMaximum(1)
        self.spin_cols.setValue(1)
        self.spin_rows.setValue(1)
        self.spin_cols.blockSignals(False)
        self.spin_rows.blockSignals(False)
        self._refresh_enabled()

    # ---------------- selection sync ----------------
    def _on_sheet_selection_changed(self, sel: Set[int], anchor: Optional[int] = None):
        # 右侧 sheet 选中 -> 同步左侧列表选中
        if self._syncing:
            return
        self._syncing = True
        try:
            self.list_widget.clearSelection()
            for i in sorted(sel):
                it = self.list_widget.item(i)
                if it is not None:
                    it.setSelected(True)
            if anchor is not None:
                self.list_widget.setCurrentRow(anchor)
        finally:
            self._syncing = False
        self._refresh_delete_btn()
        # 选中 cell 后让 sheet_label 获得焦点，使 Delete/Backspace 可用
        if sel:
            self.sheet_label.setFocus()

    def _on_list_selection_changed(self):
        # 左侧列表选中 -> 同步右侧 sheet 多选框
        if self._syncing:
            return
        self._syncing = True
        try:
            rows = {self.list_widget.row(it) for it in self.list_widget.selectedItems()}
            # anchor：用当前 currentRow（更符合用户习惯）
            anchor = self.list_widget.currentRow()
            if anchor < 0:
                anchor = None
            self.sheet_label.set_selected_set(rows, anchor=anchor)
        finally:
            self._syncing = False
        self._refresh_delete_btn()
        # 选中列表项后让 list_widget 获得焦点，使 Delete/Backspace 可用
        if self.list_widget.selectedItems():
            self.list_widget.setFocus()

    def _refresh_delete_btn(self):
        """根据当前是否有选中项来启用/禁用删除按钮"""
        if not hasattr(self, 'btn_delete'):
            return
        has_sel = bool(self.list_widget.selectedItems()) or bool(self.sheet_label.selected_set())
        self.btn_delete.setEnabled(has_sel and bool(self.paths))

    def _on_list_context_menu(self, pos):
        """左侧列表右键菜单"""
        from PySide6.QtWidgets import QMenu
        sel = self.list_widget.selectedItems()
        if not sel:
            return
        menu = QMenu(self)
        count = len(sel)
        if count == 1:
            action_del = menu.addAction(f"删除 {sel[0].text()}")
        else:
            action_del = menu.addAction(f"删除选中的 {count} 张图片")
        action_del.triggered.connect(self.delete_selected)
        menu.exec(self.list_widget.viewport().mapToGlobal(pos))

    def _on_sheet_context_menu(self, pos):
        """精灵图预览区右键菜单"""
        from PySide6.QtWidgets import QMenu
        sel = self.sheet_label.selected_set()
        if not sel:
            return
        menu = QMenu(self)
        count = len(sel)
        if count == 1:
            idx = next(iter(sel))
            name = self._cell_name(idx)
            action_del = menu.addAction(f"删除 {name}")
        else:
            action_del = menu.addAction(f"删除选中的 {count} 张图片")
        action_del.triggered.connect(self.delete_selected)
        menu.exec(self.sheet_label.mapToGlobal(pos))

    # ---------------- key handling (Delete/Backspace) ----------------
    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delete_selected()
                return True
        return super().eventFilter(obj, event)

    def delete_selected(self):
        """删除当前选中的文件/cell（支持单选/多选），按 Delete/Backspace 或按钮触发。"""
        if not self.paths:
            return

        # 优先取左侧列表选中；若没有则取右侧 sheet 选中
        sel_rows = {self.list_widget.row(it) for it in self.list_widget.selectedItems()}
        if not sel_rows:
            sel_rows = self.sheet_label.selected_set()

        sel_rows = {i for i in sel_rows if 0 <= i < len(self.paths)}
        if not sel_rows:
            return

        # 确认弹窗
        count = len(sel_rows)
        if count == 1:
            name = os.path.basename(self.paths[next(iter(sel_rows))])
            msg = f"确认从列表中移除：\n{name}？"
        else:
            msg = f"确认从列表中移除选中的 {count} 张图片？"
        reply = QMessageBox.question(
            self, "确认删除", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 删除
        self.timer.stop()
        self.paths = [p for i, p in enumerate(self.paths) if i not in sel_rows]

        # 清空选中状态
        self.sheet_label.set_selected_set(set())

        if not self.paths:
            self.clear_all()
            return

        # 删除后重新计算最优行列并同步到 spin（不改变手动/自动模式）
        self._apply_auto_grid_to_spins()
        self._reload_frames_and_rebuild(keep_selection=False)
# ---------------- list reorder ----------------
    def _on_list_reordered(self):
        ordered: List[str] = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            p = it.data(Qt.UserRole)
            if p:
                ordered.append(p)
        if ordered and ordered != self.paths:
            self._push_undo()  # 列表拖拽前压栈
            self.paths = ordered
            self._reload_frames_and_rebuild(keep_selection=True)
        self._update_gif_preview()

    def _rebuild_list_items(self, keep_selection: bool):
        selected_paths: Set[str] = set()
        if keep_selection:
            selected_paths = {it.data(Qt.UserRole) for it in self.list_widget.selectedItems()}

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for p in self.paths:
            it = QListWidgetItem(os.path.basename(p))
            it.setData(Qt.UserRole, p)
            self.list_widget.addItem(it)
            if keep_selection and p in selected_paths:
                it.setSelected(True)
        self.list_widget.blockSignals(False)

    # ---------------- sheet reorder (multi) ----------------
    def _reorder_by_cell_drag_multi(self, src_indices: List[int], insert_index: int):
        """在精灵图预览里拖拽排序：insert_index 是插入位置（0..n）"""
        n = len(self.paths)
        src = sorted(set([i for i in src_indices if 0 <= i < n]))
        if not src:
            return

        # clamp insert_index to [0, n]
        insert_index = max(0, min(n, int(insert_index)))

        # 如果落点在被拖拽区间内部（min..max+1），视为不变
        if (min(src) < insert_index <= (max(src) + 1)):
            return
        self._push_undo()  # 精灵图拖拽前压栈
        moving = [self.paths[i] for i in src]
        remaining = [p for i, p in enumerate(self.paths) if i not in src]

        removed_before = sum(1 for i in src if i < insert_index)
        insert_at = insert_index - removed_before
        insert_at = max(0, min(len(remaining), insert_at))

        self.paths = remaining[:insert_at] + moving + remaining[insert_at:]
        self._reload_frames_and_rebuild(keep_selection=True)
        self._update_gif_preview()

        # 同步选中：移动后的 indices
        new_selected = set(range(insert_at, insert_at + len(moving)))
        self._syncing = True
        try:
            self.list_widget.clearSelection()
            for i in sorted(new_selected):
                it = self.list_widget.item(i)
                if it:
                    it.setSelected(True)
            self.sheet_label.set_selected_set(new_selected, anchor=max(new_selected) if new_selected else None)
        finally:
            self._syncing = False

    # ---------------- undo (Ctrl+Z) ----------------
    def _push_undo(self):
        """将当前 paths 快照压入撤销栈（最多保留20步）"""
        self._undo_stack.append(list(self.paths))
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)

    def _undo_reorder(self):
        """撤销上一次拖拽排序"""
        if not self._undo_stack:
            return
        self.paths = self._undo_stack.pop()
        self._reload_frames_and_rebuild(keep_selection=False)
        self._update_gif_preview()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self._undo_reorder()
            event.accept()
            return
        super().keyPressEvent(event)

    def _cell_name(self, idx: int) -> str:
        if 0 <= idx < len(self.paths):
            return os.path.basename(self.paths[idx])
        return ""

    # ---------------- grid mode ----------------
    def _on_grid_mode_change(self):
        manual = not self.chk_auto_grid.isChecked()
        self.spin_cols.setEnabled(manual)
        self.spin_rows.setEnabled(manual)
        self.btn_reset_grid.setEnabled(manual)

        if self.frames_rgba:
            self._rebuild_sheet()
            self._start_preview()
            self._refresh_enabled()

    def _on_cols_changed(self):
        """列数变化：自动计算所需行数（ceil(n/cols)）"""
        if self.chk_auto_grid.isChecked():
            return
        n = len(self.frames_rgba)
        if n > 0:
            cols = max(1, self.spin_cols.value())
            auto_rows = int(math.ceil(n / cols))
            if self.spin_rows.value() != auto_rows:
                self.spin_rows.blockSignals(True)
                self.spin_rows.setValue(auto_rows)
                self.spin_rows.blockSignals(False)
            self._rebuild_sheet()
            self._start_preview()

    def _on_rows_changed(self):
        """行数变化：自动计算所需列数（ceil(n/rows)）"""
        if self.chk_auto_grid.isChecked():
            return
        n = len(self.frames_rgba)
        if n > 0:
            rows = max(1, self.spin_rows.value())
            auto_cols = int(math.ceil(n / rows))
            if self.spin_cols.value() != auto_cols:
                self.spin_cols.blockSignals(True)
                self.spin_cols.setValue(auto_cols)
                self.spin_cols.blockSignals(False)
            self._rebuild_sheet()
            self._start_preview()

    def _apply_auto_grid_to_spins(self):
        """根据当前 paths 数量计算最优行列，直接写入 spin（不改变手动/自动模式）。"""
        n = len(self.paths)
        if n <= 0:
            return
        cols, rows = self._compute_auto_grid(n)
        self.spin_cols.blockSignals(True)
        self.spin_rows.blockSignals(True)
        # 限制最大值：列/行最多不超过图片总数
        self.spin_cols.setMaximum(n)
        self.spin_rows.setMaximum(n)
        self.spin_cols.setValue(cols)
        self.spin_rows.setValue(rows)
        self.spin_cols.blockSignals(False)
        self.spin_rows.blockSignals(False)

    def _reset_grid_to_auto(self):
        self.chk_auto_grid.setChecked(True)

    # ---------------- build sheet ----------------
    def _reload_frames_and_rebuild(self, keep_selection: bool = False):
        self._rebuild_list_items(keep_selection=keep_selection)

        frames: List[Image.Image] = []
        bad: List[str] = []
        for p in self.paths:
            try:
                frames.append(Image.open(p).convert("RGBA"))
            except Exception:
                bad.append(p)

        if bad:
            QMessageBox.warning(self, "部分文件读取失败", "以下文件无法读取，将被跳过：\n" + "\n".join(bad))

        self.frames_rgba = frames

        if not self.frames_rgba:
            self.sheet_rgba = None
            self.timer.stop()
            self._refresh_enabled()
            return

        self._rebuild_sheet()
        self._start_preview()
        self._refresh_enabled()

        if keep_selection:
            self._on_list_selection_changed()

    def _compute_auto_grid(self, n: int) -> Tuple[int, int]:
        """
        找到最优行列组合（始终保证 cols >= rows，即横向排列）：
        - 优先接近正方形（行列差最小）
        - 其次空格少
        - 始终 cols >= rows，避免出现 1 列 n 行
        """
        if n <= 0:
            return 1, 1
        if n == 1:
            return 1, 1

        sqrt_n = math.sqrt(n)
        best_cols, best_rows = n, 1
        best_score = float("inf")

        # 只遍历 cols >= ceil(sqrt(n)) 的情况，保证 cols >= rows
        start = max(1, int(math.ceil(sqrt_n)))
        for cols in range(start, n + 1):
            rows = int(math.ceil(n / cols))
            empty = cols * rows - n
            diff = cols - rows  # cols >= rows，diff >= 0
            # 评分：行列差优先，空格次之
            score = diff * 10 + empty
            if score < best_score:
                best_score = score
                best_cols = cols
                best_rows = rows
            # 当 cols 远大于 sqrt_n 且已有好方案时提前退出
            if cols > sqrt_n * 3 and best_score < 5:
                break

        return max(1, best_cols), max(1, best_rows)

    def _rebuild_sheet(self):
        n = len(self.frames_rgba)
        if n <= 0:
            self.sheet_rgba = None
            return

        if self.chk_auto_grid.isChecked():
            cols, rows = self._compute_auto_grid(n)
            # 同步回 spin，让 UI 显示实际行列数
            self.spin_cols.blockSignals(True)
            self.spin_rows.blockSignals(True)
            self.spin_cols.setValue(cols)
            self.spin_rows.setValue(rows)
            self.spin_cols.blockSignals(False)
            self.spin_rows.blockSignals(False)
        else:
            # 手动模式：spin 里已经是用户设置的值，直接读取
            cols = int(self.spin_cols.value())
            rows = int(self.spin_rows.value())
            # 若行列乘积不够放所有图片，补足行数
            if cols * rows < n:
                rows = int(math.ceil(n / max(1, cols)))
                self.spin_rows.blockSignals(True)
                self.spin_rows.setValue(rows)
                self.spin_rows.blockSignals(False)

        self.sheet_cols, self.sheet_rows = cols, rows

        max_w = max(im.size[0] for im in self.frames_rgba)
        max_h = max(im.size[1] for im in self.frames_rgba)
        max_w = max(1, int(max_w))
        max_h = max(1, int(max_h))

        sheet = Image.new("RGBA", (cols * max_w, rows * max_h), (0, 0, 0, 0))

        for idx, im in enumerate(self.frames_rgba):
            r = idx // cols
            c = idx % cols
            x = c * max_w
            y = r * max_h
            sheet.paste(im.resize((max_w, max_h), resample=Image.LANCZOS), (x, y))

        self.sheet_rgba = sheet
        self._update_sheet_preview()

    # ---------------- previews ----------------
    def _update_sheet_preview(self):
        if self.sheet_rgba is None:
            self.sheet_label.set_sheet(None, 1, 1, 0)
            return
        pix = pil_to_qpixmap(self.sheet_rgba)
        self.sheet_label.set_sheet(pix, self.sheet_cols, self.sheet_rows, len(self.frames_rgba))

    def _update_gif_preview(self):
        if not self.frames_rgba:
            self.gif_label.clear()
            return
        i = self.preview_index % len(self.frames_rgba)
        pix = pil_to_qpixmap(self.frames_rgba[i]).scaled(
            self.gif_label.width(),
            self.gif_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.gif_label.setPixmap(pix)

    def _start_preview(self):
        self.preview_index = 0
        self._update_gif_preview()
        if not self.timer.isActive():
            self.timer.start()

    def _tick_preview(self):
        if not self.frames_rgba:
            self.timer.stop()
            return
        self.preview_index = (self.preview_index + 1) % len(self.frames_rgba)
        self._update_gif_preview()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_sheet_preview()
        self._update_gif_preview()
    # ---------------- naming ----------------
    def original_base(self) -> str:
        return "SpriteSheet"

    def compute_preview_basename(self) -> str:
        tag = (self.name_input.text() or "").strip()
        if tag:
            if tag.startswith("T_"):
                return tag
            return f"T_{tag}"
        return self.original_base()

    def _history_path(self) -> str:
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "name_history_sprite.txt")

    def _save_last_name(self, name: str):
        try:
            with open(self._history_path(), "w", encoding="utf-8") as f:
                f.write(name)
        except Exception:
            pass

    def _load_last_name(self) -> Optional[str]:
        try:
            with open(self._history_path(), "r", encoding="utf-8") as f:
                s = f.read().strip()
                return s or None
        except Exception:
            return None

    def _build_history_buttons(self, last: str):
        for b in (self.history_btn1, self.history_btn2, self.history_btn3):
            b.setVisible(False)

        m = re.match(r"(.+?)_(\d+)$", last)
        buttons: List[str] = []
        if m:
            base = m.group(1)
            num = int(m.group(2))
            buttons = [base, f"{base}_{num + 1}", last]
        else:
            if "_" in last:
                buttons = [last.split("_")[0], last, f"{last}_1"]
            else:
                buttons = [last, f"{last}_1", f"{last}_2"]

        btns = [self.history_btn1, self.history_btn2, self.history_btn3]
        for i, t in enumerate(buttons[:3]):
            btns[i].setText(t)
            btns[i].setVisible(True)

    def _apply_history_name(self, txt: str):
        self.name_input.setText(txt)

    def _update_name_preview(self):
        preview = self.output_basename or self.compute_preview_basename()
        locked = "（已应用）" if self.output_basename else "（未应用）"
        self.name_preview.setText(f"预览：{preview}.png {locked}")

    def _apply_naming(self):
        tag = (self.name_input.text() or "").strip()
        if not tag:
            QMessageBox.warning(self, "提示", "请输入命名（仅允许字母/数字/下划线）。")
            return

        self.output_basename = self.compute_preview_basename()

        mem = tag[2:] if tag.startswith("T_") else tag
        self._last_name = mem
        self._save_last_name(mem)
        self._build_history_buttons(mem)

        self._update_name_preview()
        QMessageBox.information(self, "命名已应用", f"导出时将使用：\n{self.output_basename}.png")

    def _reset_naming(self):
        self.output_basename = None
        self.name_input.setText("")
        self._update_name_preview()

    def get_export_basename(self) -> str:
        return self.output_basename or self.original_base()



    # ---------------- export ----------------
    def _get_export_dir_cache_path(self) -> str:
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "sprite_last_export_dir.txt")

    def _load_last_export_dir(self) -> str:
        try:
            with open(self._get_export_dir_cache_path(), "r", encoding="utf-8") as f:
                d = f.read().strip()
                if d and os.path.isdir(d):
                    return d
        except Exception:
            pass
        return ""

    def _save_last_export_dir(self, path: str):
        try:
            with open(self._get_export_dir_cache_path(), "w", encoding="utf-8") as f:
                f.write(os.path.dirname(path))
        except Exception:
            pass

    def export_sheet(self):
        if self.sheet_rgba is None:
            return
        try:
            out_w = int(self.size_combo_w.currentText())
        except Exception:
            out_w = 1024
        try:
            out_h = int(self.size_combo_h.currentText())
        except Exception:
            out_h = 1024

        out_img = self.sheet_rgba.resize((out_w, out_h), resample=Image.LANCZOS)

        default_dir = self._load_last_export_dir()
        default_name = os.path.join(default_dir, f"{self.get_export_basename()}.png") if default_dir else f"{self.get_export_basename()}.png"
        path, _ = QFileDialog.getSaveFileName(self, "导出精灵图", default_name, "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            out_img.save(path)
            self._save_last_export_dir(path)
            QMessageBox.information(self, "完成", f"已导出：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")
