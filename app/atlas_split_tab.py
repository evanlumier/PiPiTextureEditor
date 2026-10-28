# -*- coding: utf-8 -*-
"""
atlas_split_tab.py - 图集拆分板块

复刻 Sprite2imgV1.1.exe（T3D UV解析工具）：
导入原始图集 + 一批 UE 导出的 Sprite .T3D 文件 → 点“导出”一次性按每个 T3D 的区域裁出单张 PNG 到输出目录；
或用主窗口底部“导入UE4”直接导入（两者独立）。
裁切逻辑见 atlas_split_core.py（与原工具逐行一致）；本文件只负责界面。

界面：
  - 左侧大图预览（只读）：整张图集变暗，T3D 对应的区域保持原样并加细边框，列表选中项边框加粗
  - 右侧：所有资源都从这里添加，各区域只收自己那一类文件
      原始图片区 ← 图片；T3D 文件区 ← .t3d；输出目录区 ← 文件夹
  - 解析失败或与当前图集对不上的 T3D 在列表中标红（鼠标悬停看原因；选中后仍保持红色）
  - 日志只记录导出结果：每个文件成功/失败 + 最终汇总
  - 有对不上的 T3D 时，导出/导入UE4 前弹窗确认，可选择仍然继续
  - 导出名默认 = T3D 名（<T3D名>.png，不再像原工具那样重复两遍），双击列表项可单独改名；
    名字不合规（非 ASCII 字母数字下划线、数字开头）或列表内重名时标红，并拦下导出/导入
  - 主窗口底部“导入UE4”：批量导入列表全部项，不落地到输出目录；目标目录与 Sprite 所在目录
    相同且同名时，贴图名自动加 _Tex 后缀

与原工具的差异（不影响裁切结果）：
  - 支持一次处理多个 T3D；除上面的确认框外不弹窗，结果都写进日志区
  - 原工具用线程处理；这里在主线程逐个处理，每个文件之间刷新界面
  - 批量处理时图集只解码一次
"""
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PIL import Image

from PySide6.QtCore import Qt, QRectF, QEvent, QObject
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QBrush
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout,
    QGroupBox, QProgressBar, QPlainTextEdit, QFileDialog, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QApplication, QStyledItemDelegate, QStyleOptionViewItem,
    QStyle, QMessageBox, QInputDialog,
)

from export_dir_mixin import ExportDirMixin
from theme import T
from utils import pil_to_qpixmap
from widgets import CheckerLabel
from ue4_sync import RESULT_TIMEOUT
from atlas_split_core import (
    parse_t3d_file, compute_crop_box, crop_sprite, read_source_texture_name,
    check_atlas_match, validate_asset_name, read_sprite_asset_ref, ue4_import_name,
    UE4_CLASH_SUFFIX,
)

T3D_EXTS = (".t3d", ".txt")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tga")

Box = Tuple[int, int, int, int]


def _is_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _is_t3d(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in T3D_EXTS


class _KeepForegroundDelegate(QStyledItemDelegate):
    """选中时保留条目自己的文字颜色（标红的项选中后仍是红色）。

    全局样式表 QListWidget::item:selected 指定了选中文字色，样式表优先级高于调色板，
    改 palette 盖不住；所以有自定义颜色的条目：背景/选中底色照常交给 style 画（文字留空），
    文字由这里用条目自己的颜色画。没有自定义颜色的条目完全走默认绘制。
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        fg = index.data(Qt.ForegroundRole)
        if fg is None:
            super().paint(painter, option, index)
            return
        color = fg.color() if isinstance(fg, QBrush) else QColor(fg)

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, widget)
        text = opt.text
        opt.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        painter.save()
        painter.setFont(opt.font)
        painter.setPen(color)
        elided = opt.fontMetrics.elidedText(text, Qt.ElideRight, text_rect.width())
        painter.drawText(text_rect, int(opt.displayAlignment) | Qt.TextSingleLine, elided)
        painter.restore()


@dataclass
class T3DEntry:
    """列表中的一个 T3D 文件"""
    path: str
    info: Optional[dict] = None        # 解析结果；None 表示解析失败
    error: str = ""                    # 解析失败原因
    texture_name: Optional[str] = None  # T3D 引用的贴图名
    box: Optional[Box] = None          # 在当前图集上的裁切区域（与实际输出一致）
    mismatch: str = ""                 # 与当前图集对不上的原因
    custom_name: str = ""              # 用户改的导出名；空 = 用 T3D 名
    name_error: str = ""               # 导出名不合规/重名的原因
    sprite_dir: Optional[str] = None   # Sprite 在 UE 中的目录（判断导入 UE4 时是否同名冲突）
    sprite_asset: Optional[str] = None  # Sprite 在 UE 中的资产名

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def export_name(self) -> str:
        """导出文件名 / UE4 资产名（不含扩展名）"""
        return self.custom_name or self.name

    @property
    def is_bad(self) -> bool:
        return bool(self.error or self.mismatch or self.name_error)

    @property
    def region_bad(self) -> bool:
        """预览框标红只看区域本身（解析失败/对不上），命名问题不影响预览"""
        return bool(self.error or self.mismatch)


class _DropTarget(QObject):
    """让某个区域只接收一类文件：accept(path) 为真的文件才可拖入，拖入后交给 on_drop(paths)"""

    def __init__(self, widget: QWidget, accept: Callable[[str], bool],
                 on_drop: Callable[[List[str]], None]):
        super().__init__(widget)
        self._accept = accept
        self._on_drop = on_drop
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    def _paths(self, event) -> List[str]:
        md = event.mimeData()
        if not md.hasUrls():
            return []
        paths = [os.path.normpath(u.toLocalFile()) for u in md.urls() if u.isLocalFile()]
        return [p for p in paths if self._accept(p)]

    def eventFilter(self, obj, event):
        t = event.type()
        if t in (QEvent.DragEnter, QEvent.DragMove):
            if self._paths(event):
                event.acceptProposedAction()
            else:
                event.ignore()
            return True
        if t == QEvent.Drop:
            paths = self._paths(event)
            if paths:
                event.acceptProposedAction()
                self._on_drop(paths)
            else:
                event.ignore()
            return True
        return False


class AtlasPreview(CheckerLabel):
    """图集预览（只读，不接收拖入/点击导入）：整图变暗，T3D 区域按原样显示并加细边框。
    滚轮缩放、右键拖动平移、双击适配窗口沿用 CheckerLabel。"""

    DIM_OPACITY = 0.25

    def __init__(self, parent=None):
        super().__init__(cell=12, parent=parent)
        self.setAcceptDrops(False)
        # (box, 对不上, 选中)
        self._regions: List[Tuple[Box, bool, bool]] = []

    def set_regions(self, regions: List[Tuple[Box, bool, bool]]):
        self._regions = regions
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source_pix is None or self._source_pix.isNull():
            p = QPainter(self)
            p.setPen(QColor(T.text_tertiary))
            font = p.font()
            font.setPixelSize(14)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignCenter, "图集预览\n在右侧导入原始图片后显示")
            p.end()

    def _draw_image(self, p: QPainter, pix_rect: QRectF, pm: QPixmap):
        if not self._regions:
            super()._draw_image(p, pix_rect, pm)
            return

        sx = pix_rect.width() / pm.width()
        sy = pix_rect.height() / pm.height()

        def dst_rect(box: Box) -> QRectF:
            left, top, right, bottom = box
            return QRectF(pix_rect.x() + left * sx, pix_rect.y() + top * sy,
                          (right - left) * sx, (bottom - top) * sy)

        # 整图变暗
        p.save()
        p.setOpacity(self.DIM_OPACITY)
        p.drawPixmap(pix_rect, pm, QRectF(0, 0, pm.width(), pm.height()))
        p.restore()

        # T3D 区域按原样再画一遍
        for box, _, _ in self._regions:
            left, top, right, bottom = box
            p.drawPixmap(dst_rect(box), pm, QRectF(left, top, right - left, bottom - top))

        # 细边框：普通=强调色，对不上=红色，选中=加粗（未对不上时用黄色）；选中项最后画，压在上面
        p.save()
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setBrush(Qt.NoBrush)
        for box, bad, selected in sorted(self._regions, key=lambda r: r[2]):
            color = T.error if bad else (T.warning if selected else T.accent)
            pen = QPen(QColor(color))
            pen.setCosmetic(True)
            pen.setWidth(2 if selected else 1)
            p.setPen(pen)
            p.drawRect(dst_rect(box))
        p.restore()


class AtlasSplitTab(ExportDirMixin, QWidget):
    """图集拆分：原始图集 + 多个 T3D → 单张 Sprite PNG"""

    _export_dir_cache_name = "atlas_split_last_export_dir.txt"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[T3DEntry] = []
        self._atlas_path: str = ""
        self._atlas_size: Optional[Tuple[int, int]] = None
        self._running = False
        self._drop_targets: List[_DropTarget] = []
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # 左：大图预览（只读）
        self.preview = AtlasPreview()
        self.preview.setMinimumSize(420, 360)
        self.preview.setStyleSheet("border-radius:10px; border:1px solid #383850;")

        # 右：控制区（所有资源都从这里添加）
        panel = QWidget()
        panel.setMinimumWidth(360)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)

        hint = QLabel("图片拖到「原始图片」，T3D 拖到「T3D 文件」。点“导出”按每个 T3D 的区域裁出 PNG 到输出目录，"
                      "点底部“导入UE4”直接导入 UE4，两者互不依赖。导出名默认为 T3D 名，双击列表项可改名。"
                      "左侧仅用于预览：滚轮缩放，右键拖动，双击复位。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{T.text_tertiary};")
        side.addWidget(hint)

        # 原始图片：只收图片
        atlas_group = QGroupBox("原始图片")
        atlas_row = QHBoxLayout(atlas_group)
        self.edit_atlas = QLineEdit()
        self.edit_atlas.setReadOnly(True)
        self.edit_atlas.setAcceptDrops(False)  # 交给分组框统一接收
        self.edit_atlas.setPlaceholderText("拖入图集图片到这里，或点击浏览")
        btn_atlas = QPushButton("浏览")
        btn_atlas.clicked.connect(self.browse_image_file)
        atlas_row.addWidget(self.edit_atlas, 1)
        atlas_row.addWidget(btn_atlas)
        side.addWidget(atlas_group)
        self._drop_targets.append(_DropTarget(atlas_group, _is_image, self._on_atlas_dropped))

        # T3D 列表：只收 T3D
        t3d_group = QGroupBox("T3D 文件")
        t3d_box = QVBoxLayout(t3d_group)
        self.list_t3d = QListWidget()
        self.list_t3d.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_t3d.setItemDelegate(_KeepForegroundDelegate(self.list_t3d))
        self.list_t3d.setMinimumHeight(140)
        self.list_t3d.itemSelectionChanged.connect(self._update_preview_regions)
        self.list_t3d.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_t3d.installEventFilter(self)
        t3d_box.addWidget(self.list_t3d, 1)
        self.lbl_t3d_count = QLabel()
        self.lbl_t3d_count.setStyleSheet(f"color:{T.text_tertiary};")
        t3d_box.addWidget(self.lbl_t3d_count)
        t3d_btns = QHBoxLayout()
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self.browse_t3d_files)
        btn_remove = QPushButton("移除选中")
        btn_remove.clicked.connect(self._remove_selected)
        btn_clear_list = QPushButton("清空列表")
        btn_clear_list.clicked.connect(self._clear_list)
        t3d_btns.addWidget(btn_add)
        t3d_btns.addWidget(btn_remove)
        t3d_btns.addWidget(btn_clear_list)
        t3d_box.addLayout(t3d_btns)
        side.addWidget(t3d_group, 2)
        self._drop_targets.append(_DropTarget(t3d_group, _is_t3d, self._add_t3d_files))

        # 输出目录：只收文件夹
        out_group = QGroupBox("输出目录")
        out_row = QHBoxLayout(out_group)
        self.edit_output = QLineEdit(self._default_output_dir())
        self.edit_output.setAcceptDrops(False)
        btn_output = QPushButton("浏览")
        btn_output.clicked.connect(self.browse_output_dir)
        out_row.addWidget(self.edit_output, 1)
        out_row.addWidget(btn_output)
        side.addWidget(out_group)
        self._drop_targets.append(_DropTarget(out_group, os.path.isdir, self._on_output_dropped))

        # 操作按钮
        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("导出")
        self.btn_export.setToolTip("把列表里全部 T3D 裁成 PNG，保存到上面的输出目录")
        self.btn_export.setProperty("class", "primary")
        self.btn_export.clicked.connect(self.export_local)
        btn_open = QPushButton("打开输出目录")
        btn_open.clicked.connect(self.open_output_dir)
        btn_row.addWidget(self.btn_export, 1)
        btn_row.addWidget(btn_open)
        side.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        side.addWidget(self.progress)

        side.addWidget(QLabel("处理日志:"))
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        side.addWidget(self.log_text, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.preview)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([720, 420])
        root.addWidget(splitter)

        self._refresh_count()

    def _default_output_dir(self) -> str:
        return self._load_last_export_dir() or os.getcwd()

    def log(self, message: str):
        self.log_text.appendPlainText(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---------------- 导入：图集 ----------------
    def browse_image_file(self):
        if self._running:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择原始图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.tga);;所有文件 (*.*)"
        )
        if filename:
            self._load_atlas(filename)

    def _on_atlas_dropped(self, paths: List[str]):
        self._load_atlas(paths[0])  # 一次只用一张图集

    def _load_atlas(self, path: str):
        if self._running:
            return
        try:
            with Image.open(path) as img:
                img.load()
                size = img.size
                pix = pil_to_qpixmap(img)
        except Exception as e:
            # 读不了图属于导入失败，不写日志；同时卸掉旧图集，避免路径框为空却仍按旧图集导出
            self._atlas_path = ""
            self._atlas_size = None
            self.preview.clear()
            self.edit_atlas.clear()
            self.edit_atlas.setToolTip("")
            self.edit_atlas.setPlaceholderText(f"无法读取图片：{os.path.basename(path)}（{e}）")
            self._recompute_entries()
            return
        self._atlas_path = path
        self._atlas_size = size
        self.edit_atlas.setText(path)
        self.edit_atlas.setToolTip(f"{path}\n{size[0]}x{size[1]}")
        self.preview.setPixmap(pix)
        self._recompute_entries()

    # ---------------- 导入：T3D ----------------
    def browse_t3d_files(self):
        if self._running:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择T3D文件", "", "T3D文件 (*.t3d);;文本文件 (*.txt);;所有文件 (*.*)"
        )
        if files:
            self._add_t3d_files(files)

    def _add_t3d_files(self, paths: List[str]):
        """加入列表（已在列表中的跳过）；导入时静默解析，解析失败的项在列表中标红"""
        if self._running:
            return
        existed = {os.path.normcase(os.path.abspath(e.path)) for e in self._entries}
        added = 0
        for p in sorted(paths, key=lambda s: os.path.basename(s).lower()):
            key = os.path.normcase(os.path.abspath(p))
            if key in existed:
                continue
            existed.add(key)
            entry = T3DEntry(path=os.path.abspath(p))
            lines: List[str] = []
            entry.info = parse_t3d_file(entry.path, lines.append)
            if entry.info is None:
                entry.error = lines[-1] if lines else "解析失败"
            else:
                entry.texture_name = read_source_texture_name(entry.path)
            entry.sprite_dir, entry.sprite_asset = read_sprite_asset_ref(entry.path)
            self._entries.append(entry)
            added += 1
        if added:
            self._recompute_entries()

    def _recompute_entries(self):
        """按当前图集重新计算每个 T3D 的区域、对不上状态和命名问题，然后刷新列表和预览"""
        for e in self._entries:
            e.box = None
            e.mismatch = ""
            if e.info is None or self._atlas_size is None:
                continue
            w, h = self._atlas_size
            try:
                e.box = compute_crop_box(w, h, e.info)
            except ZeroDivisionError:
                e.mismatch = "纹理尺寸为 0，无法计算区域"
                continue
            e.mismatch = check_atlas_match(e.info, e.texture_name, self._atlas_path, w, h)
        self._validate_names()
        self._rebuild_list()

    def _validate_names(self):
        """导出名合规 + 列表内不重名（不区分大小写：Windows 文件名和 UE4 资产名都不区分）"""
        groups = {}
        for e in self._entries:
            e.name_error = validate_asset_name(e.export_name)
            groups.setdefault(e.export_name.lower(), []).append(e)
        for same in groups.values():
            if len(same) > 1:
                for e in same:
                    others = "、".join(o.name for o in same if o is not e)
                    msg = f"导出名 {e.export_name} 与 {others} 重名"
                    e.name_error = f"{e.name_error}；{msg}" if e.name_error else msg

    def _rebuild_list(self):
        selected = {self.list_t3d.row(it) for it in self.list_t3d.selectedItems()}
        self.list_t3d.blockSignals(True)
        self.list_t3d.clear()
        for i, e in enumerate(self._entries):
            label = e.name if not e.custom_name else f"{e.name}  →  {e.custom_name}"
            if e.info is not None:
                text = f"{label}    {e.info['source_width']:g}x{e.info['source_height']:g}"
            else:
                text = f"{label}    解析失败"
            item = QListWidgetItem(text)
            tip = f"{e.path}\n导出名：{e.export_name}.png（双击改名）"
            if e.texture_name:
                tip += f"\n引用贴图：{e.texture_name}"
            if e.error:
                tip += f"\n解析失败：{e.error}"
            if e.mismatch:
                tip += f"\n与当前图集对不上：{e.mismatch}"
            if e.name_error:
                tip += f"\n命名问题：{e.name_error}"
            item.setToolTip(tip)
            if e.is_bad:
                item.setForeground(QColor(T.error))
            self.list_t3d.addItem(item)
            if i in selected:
                item.setSelected(True)
        self.list_t3d.blockSignals(False)
        self._refresh_count()
        self._update_preview_regions()

    def _update_preview_regions(self):
        selected = {self.list_t3d.row(it) for it in self.list_t3d.selectedItems()}
        regions = [(e.box, e.region_bad, i in selected)
                   for i, e in enumerate(self._entries) if e.box is not None]
        self.preview.set_regions(regions)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        row = self.list_t3d.row(item)
        if 0 <= row < len(self._entries):
            self._rename_entry(row)

    def _rename_entry(self, row: int):
        """单独修改某一项的导出名；留空 = 恢复为 T3D 名"""
        if self._running:
            return
        e = self._entries[row]
        text, ok = QInputDialog.getText(
            self, "修改导出名",
            f"T3D：{e.name}\n导出名（只能用英文字母、数字、下划线，不能以数字开头；留空恢复为 T3D 名）：",
            QLineEdit.Normal, e.export_name)
        if not ok:
            return
        self._set_custom_name(row, text)

    def _set_custom_name(self, row: int, text: str):
        e = self._entries[row]
        text = (text or "").strip()
        if text.lower().endswith(".png"):
            text = text[:-4]
        e.custom_name = "" if text == e.name else text
        self._validate_names()
        self._rebuild_list()

    def _refresh_count(self):
        n = len(self._entries)
        bad = sum(1 for e in self._entries if e.is_bad)
        text = f"共 {n} 个" + (f"，{bad} 个有问题（红色，悬停查看原因）" if bad else "") + "；双击改导出名"
        self.lbl_t3d_count.setText(text if n else "可一次拖入多个 .t3d 文件")

    def _remove_selected(self):
        if self._running:
            return
        rows = {self.list_t3d.row(it) for it in self.list_t3d.selectedItems()}
        if not rows:
            return
        self._entries = [e for i, e in enumerate(self._entries) if i not in rows]
        self.list_t3d.clearSelection()
        self._validate_names()
        self._rebuild_list()

    def _clear_list(self):
        if self._running:
            return
        self._entries = []
        self._validate_names()
        self._rebuild_list()

    def eventFilter(self, obj, event):
        if obj is self.list_t3d and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self._remove_selected()
                return True
        return super().eventFilter(obj, event)

    # ---------------- 输出目录 ----------------
    def browse_output_dir(self):
        if self._running:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.edit_output.text().strip())
        if directory:
            self.edit_output.setText(directory)

    def _on_output_dropped(self, paths: List[str]):
        if not self._running:
            self.edit_output.setText(paths[0])

    # ---------------- 按钮 ----------------
    def open_output_dir(self):
        output_dir = self.edit_output.text().strip()
        if os.path.exists(output_dir):
            try:
                os.startfile(output_dir)
            except Exception as e:
                self.log(f"无法打开输出目录: {str(e)}")
        else:
            self.log("警告: 输出目录不存在！")

    def export_local(self):
        """“导出”按钮：把列表中全部 T3D 裁成 PNG 存到输出目录；不弹窗（对不上的确认框除外），结果写进日志"""
        if self._running:
            return
        output_dir = self.edit_output.text().strip()
        if not self._entries:
            self.log("错误: 请先导入T3D文件！")
            return
        if not self._atlas_path or not os.path.exists(self._atlas_path):
            self.log("错误: 请选择有效的图片文件！")
            return
        if not output_dir:
            self.log("错误: 请选择有效的输出目录！")
            return
        if not self._precheck("导出"):
            return

        self._running = True
        self.btn_export.setEnabled(False)
        try:
            self._export_all(output_dir)
        finally:
            self._running = False
            self.btn_export.setEnabled(True)

    def _precheck(self, verb: str) -> bool:
        """导出/导入前的共同检查：命名有问题直接拦下（写日志），对不上的弹窗确认。返回 True 表示继续"""
        bad_names = [e for e in self._entries if e.name_error]
        if bad_names:
            if self.log_text.toPlainText():
                self.log("")
            self.log(f"错误: 有 {len(bad_names)} 个导出名有问题（列表中标红），双击改名后再{verb}：")
            for e in bad_names:
                self.log(f"  {e.name}（导出名 {e.export_name}）：{e.name_error}")
            return False
        return self._confirm_mismatch(verb)

    def _confirm_mismatch(self, verb: str = "导出") -> bool:
        """列表里有与当前图集对不上的 T3D 时弹窗确认；没有则直接通过。返回 True 表示继续执行"""
        bad = [e for e in self._entries if e.mismatch]
        if not bad:
            return True
        shown = bad[:10]
        lines = [f"· {e.name}：{e.mismatch}" for e in shown]
        if len(bad) > len(shown):
            lines.append(f"……另有 {len(bad) - len(shown)} 个")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("T3D 与图集对不上")
        box.setText(f"有 {len(bad)} 个 T3D 与当前图集对不上（列表中标红），"
                    f"按坐标裁出来的很可能不是想要的图。\n\n是否仍然{verb}全部 {len(self._entries)} 个？")
        box.setInformativeText("\n".join(lines))
        btn_force = box.addButton(f"仍然{verb}", QMessageBox.AcceptRole)
        btn_cancel = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(btn_cancel)
        box.exec()
        return box.clickedButton() is btn_force

    @staticmethod
    def _crop_entry(atlas: Image.Image, e: T3DEntry) -> Tuple[Optional[Image.Image], str]:
        """导出和导入UE4共用：用添加时解析好的数据（与预览一致）裁出区域，返回 (图, 失败原因)"""
        if e.info is None:
            return None, e.error or "解析失败"
        try:
            region = crop_sprite(atlas, e.info)
        except ZeroDivisionError:
            return None, "纹理尺寸为 0，无法计算区域"
        if region is None:
            return None, "无效的截取区域"
        return region, ""

    def _export_all(self, output_dir: str):
        """日志只写每个文件的导出结果和最终汇总"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            atlas = Image.open(self._atlas_path)
            atlas.load()  # 只解码一次，所有 T3D 共用
        except Exception as e:
            self.log(f"处理过程中发生错误: {str(e)}")
            return

        entries = list(self._entries)
        n = len(entries)
        ok = 0
        failed = 0
        self.progress.setValue(0)
        if self.log_text.toPlainText():
            self.log("")
        try:
            for i, e in enumerate(entries):
                out_file = os.path.join(output_dir, f"{e.export_name}.png")
                region, reason = self._crop_entry(atlas, e)
                if region is not None:
                    try:
                        region.save(out_file, "PNG")  # 保持图集原模式，与原工具一致
                    except Exception as ex:
                        region, reason = None, f"保存失败: {str(ex)}"

                if region is not None:
                    ok += 1
                    line = f"成功  {e.name} → {os.path.basename(out_file)}"
                    if e.mismatch:
                        line += f"（注意：与当前图集对不上：{e.mismatch}）"
                else:
                    failed += 1
                    line = f"失败  {e.name}：{reason}"
                self.log(line)
                self.progress.setValue(int((i + 1) * 100 / n))
                QApplication.processEvents()  # 让日志和进度条逐个刷新
        finally:
            atlas.close()

        self.log(f"导出完成：成功 {ok} 个，失败 {failed} 个，输出到 {output_dir}")
        if ok:
            self._save_last_export_dir(output_dir)

    # ---------------- 导入 UE4（由主窗口底部“导入UE4”按钮调用）----------------
    def export_to_ue4_batch(self, target_path: str, sync_mgr) -> None:
        """把列表里全部 T3D 裁出来逐个导入 UE4（不落地到输出目录，临时文件用完即删）。

        - 资产名 = 导出名；目标目录里有同名 Sprite 时自动加 UE4_CLASH_SUFFIX
        - 结果逐行写日志 + 汇总，不弹窗（对不上的确认框除外）
        - 某一项等满超时仍无响应，视为 UE4 不在线，中止剩余项（否则每项都要卡满超时时间）
        target_path / UE4 是否就绪由主窗口检查好再传进来
        """
        if self._running:
            return
        if not self._entries:
            self.log("错误: 请先导入T3D文件！")
            return
        if not self._atlas_path or not os.path.exists(self._atlas_path):
            self.log("错误: 请选择有效的图片文件！")
            return
        if not self._precheck("导入"):
            return

        sprites = [(e.sprite_dir, e.sprite_asset) for e in self._entries]
        final_names = {id(e): ue4_import_name(e.export_name, target_path, sprites) for e in self._entries}
        # 加后缀后可能与别的项撞名（如另一项本来就叫 X_Tex），撞了就拦下
        seen = {}
        for e in self._entries:
            seen.setdefault(final_names[id(e)].lower(), []).append(e)
        clashes = [same for same in seen.values() if len(same) > 1]
        if clashes:
            if self.log_text.toPlainText():
                self.log("")
            self.log("错误: 加后缀后有重名，请双击改名后再导入：")
            for same in clashes:
                self.log("  " + final_names[id(same[0])] + "：" + "、".join(x.name for x in same))
            return

        try:
            atlas = Image.open(self._atlas_path)
            atlas.load()
        except Exception as ex:
            self.log(f"处理过程中发生错误: {str(ex)}")
            return

        tmp_dir = tempfile.mkdtemp(prefix="pipi_atlas_ue4_")
        entries = list(self._entries)
        n = len(entries)
        ok = failed = skipped = 0
        self._running = True
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        if self.log_text.toPlainText():
            self.log("")
        try:
            for i, e in enumerate(entries):
                name = final_names[id(e)]
                timed_out = False
                region, reason = self._crop_entry(atlas, e)
                success = False
                if region is not None:
                    try:
                        if region.mode not in ("RGB", "RGBA"):
                            region = region.convert("RGBA")
                        png = os.path.join(tmp_dir, f"{name}.png")
                        region.save(png, "PNG")
                        t0 = time.monotonic()
                        success, reason = sync_mgr.send_import_request(
                            png_path=png, target_path=target_path, asset_name=name)
                        timed_out = not success and time.monotonic() - t0 >= RESULT_TIMEOUT
                        try:
                            os.remove(png)
                        except OSError:
                            pass
                    except Exception as ex:
                        success, reason = False, f"处理过程中发生错误: {str(ex)}"

                if success:
                    ok += 1
                    line = f"成功  {e.name} → {target_path.rstrip('/')}/{name}"
                    notes = []
                    if name != e.export_name:
                        notes.append(f"目标目录有同名 Sprite，已加后缀 {UE4_CLASH_SUFFIX}")
                    if e.mismatch:
                        notes.append(f"与当前图集对不上：{e.mismatch}")
                    if notes:
                        line += "（注意：" + "；".join(notes) + "）"
                else:
                    failed += 1
                    line = f"失败  {e.name}：{reason}"
                self.log(line)
                self.progress.setValue(int((i + 1) * 100 / n))
                QApplication.processEvents()
                if timed_out and i + 1 < n:
                    skipped = n - i - 1
                    self.log(f"已中止：UE4 无响应，剩余 {skipped} 个未导入")
                    break
        finally:
            atlas.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._running = False
            self.btn_export.setEnabled(True)

        summary = f"导入UE4完成：成功 {ok} 个，失败 {failed} 个"
        if skipped:
            summary += f"，未导入 {skipped} 个"
        self.log(summary + f"，目标 {target_path}")
