import sys
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageEnhance, ImageOps

from export_dir_mixin import ExportDirMixin

from PySide6.QtCore import Qt, QRect, QRectF, QPoint, QPointF, QRegularExpression, QSize, QThread, QTimer, Signal
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QColor,
    QRegularExpressionValidator,
    QFontMetrics,
    QIcon,
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
    QColorDialog,
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
from tab_transfer import (
    TAB_TEXTURE, TAB_SPRITE, TAB_FLOWMAP, TAB_GRAYGROWTH, TAB_VIEWER,
    TAB_NAMES, TAB_HINTS, build_send_menu, pil_to_temp_png, qpixmap_to_pil,
)
from version import __version__
from ue4_sync import get_sync_manager

# =========================
# 规则：输入框仅允许 A-Z a-z 0-9 _ （导出名由输入框保证）
# =========================
VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp"]

from utils import pil_to_qpixmap, to_bw_rgba

from dialogs import CropDialog, MaskThresholdDialog

from widgets import DropLabel, CheckerLabel, StackedTextTabBar


class MainWindow(ExportDirMixin, QMainWindow):
    # 跨线程安全信号：后台线程 emit，主线程自动接收（Qt 信号槽机制）
    _ue4_export_signal = Signal(dict)

    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("PPEditor")
        # 设置窗口左上角图标（兼容打包环境，多路径回退查找）
        _ico_name = "TextureToolGUI.ico"
        _candidates = []
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _candidates.append(os.path.join(_this_dir, _ico_name))
        if getattr(sys, 'frozen', False):
            # 打包环境：ico 在 exe 同级目录和 _MEIPASS 中
            _candidates.append(os.path.join(os.path.dirname(sys.executable), _ico_name))
            _candidates.append(os.path.join(getattr(sys, '_MEIPASS', ''), _ico_name))
        _ico_path = next((p for p in _candidates if os.path.exists(p)), None)
        if _ico_path:
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
                image: url("__CHK_MARK__");
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
        # 白色对勾图标（用于 QCheckBox checked 状态）
        _chk_mark_b64 = (
            b"iVBORw0KGgoAAAANSUhEUgAAAAwAAAAMCAYAAABWdVznAAAAQ0lEQVR4nNVO"
            b"OQoAMAgz/v/P6aLggcWOzRLFHIr8CZL0WbdiGus2GQCaIQoiYOJkqNXO"
            b"UZwM8TA1tZdqWt1H3BqecQATASf9lQz7/wAAAABJRU5ErkJggg=="
        )
        _tmp_chk = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        _tmp_chk.write(_b64.b64decode(_chk_mark_b64))
        _tmp_chk.close()
        import atexit as _atexit
        _atexit.register(lambda p=_tmp_dn.name: os.path.exists(p) and os.remove(p))
        _atexit.register(lambda p=_tmp_chk.name: os.path.exists(p) and os.remove(p))
        _style = _style.replace("__COMBO_DN_ARROW__", _tmp_dn.name.replace("\\", "/"))
        _style = _style.replace("__CHK_MARK__", _tmp_chk.name.replace("\\", "/"))
        self.setStyleSheet(_style)

        self.src_path: Optional[str] = None
        self.source_color: Optional[Image.Image] = None  # 当前基准图（图像调整确认后更新）
        self.master_color: Optional[Image.Image] = None  # 当前基础（裁切后会替换为裁切结果）

        # UE4 来源信息：按 Tab 索引独立存储，每个 Tab 有自己的来源信息
        # 格式：{tab_index: {"asset_path": str, "asset_name": str}}
        # 这样切换 Tab 时导出信息互不干扰
        self._ue4_source_info_per_tab: dict = {}  # {0: {...}, 1: {...}, 2: {...}, 3: {...}}

        # UE4 目标路径：按 Tab 索引独立存储，切换 Tab 时自动恢复对应路径
        self._ue4_target_path_per_tab: dict = {}  # {0: "/Game/...", 2: "/Game/...", ...}

        self.is_bw: bool = False
        self.has_unmult: bool = False
        self.custom_bg_color: Optional[Tuple[int, int, int]] = None
        self.target_size: Optional[Tuple[int, int]] = None
        self._pre_crop_color: Optional[Image.Image] = None  # 图像调整前的原始图（用于重置）

        self.working_img: Optional[Image.Image] = None
        self._preview_thumb: Optional[Image.Image] = None  # 预览用缩略图
        self._preview_dirty: bool = True  # 标记全尺寸预览图是否需要重新生成
        self._preview_full_cache: Optional[Image.Image] = None  # 全尺寸预览图缓存
        self._PREVIEW_MAX = 1024  # 预览缩略图最大边长

        self.output_basename: Optional[str] = None

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)

        # 用自定义 TabBar（竖排文字），注意：必须先 setTabBar 再 setShape
        tabs.setTabBar(StackedTextTabBar())
        tabs.tabBar().setShape(QTabBar.RoundedWest)
        self._tabs = tabs  # 保存引用，供跨 tab 切换使用

        # ── 全局 UE4 联动区域（所有 tab 共享，放在 tab 内容区下方） ──
        ue4_bar = QWidget()
        ue4_bar.setStyleSheet(
            "background-color: #252535; border-top: 1px solid #383850;"
        )
        ue4_bar_layout = QHBoxLayout(ue4_bar)
        ue4_bar_layout.setContentsMargins(12, 6, 12, 6)
        ue4_bar_layout.setSpacing(8)

        ue4_icon_label = QLabel("🔗")
        ue4_icon_label.setFixedWidth(20)
        ue4_icon_label.setAlignment(Qt.AlignCenter)
        ue4_bar_layout.addWidget(ue4_icon_label)

        ue4_bar_layout.addWidget(QLabel("UE4路径："))
        self.ue4_target_path = QLineEdit()
        self.ue4_target_path.setPlaceholderText("/Game/Art/UI/Textures")
        # 启动时加载 Tab 0 的缓存路径（默认显示 Tab 0）
        init_path = self._load_ue4_target_path(0)
        self.ue4_target_path.setText(init_path)
        if init_path:
            self._ue4_target_path_per_tab[0] = init_path
        ue4_bar_layout.addWidget(self.ue4_target_path, 1)

        self.btn_export_ue4 = QPushButton("导入UE4")
        self.btn_export_ue4.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e; font-weight:700;"
            "padding:6px 18px; border-radius:7px;"
        )
        self.btn_export_ue4.clicked.connect(self.export_to_ue4)
        ue4_bar_layout.addWidget(self.btn_export_ue4)

        # 将 tabs + UE4 联动区域组合为中央容器
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(tabs, 1)
        central_layout.addWidget(ue4_bar, 0)
        self.setCentralWidget(central)
        self._ue4_bar = ue4_bar  # 保存引用

        # ── 左下角 bug 按钮（覆盖在主窗口上，绝对定位） ──
        self._bug_btn = QPushButton(self)
        self._bug_btn.setFixedSize(32, 32)
        self._bug_btn.setCursor(Qt.PointingHandCursor)
        self._bug_btn.setToolTip("关于 / 检查更新")
        self._bug_btn.clicked.connect(self._show_about_dialog)

        _svg_name = "bug.svg"
        _svg_candidates = []
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _svg_candidates.append(os.path.join(_this_dir, _svg_name))
        if getattr(sys, 'frozen', False):
            _svg_candidates.append(os.path.join(os.path.dirname(sys.executable), _svg_name))
            _svg_candidates.append(os.path.join(getattr(sys, '_MEIPASS', ''), _svg_name))
        _svg_path = next((p for p in _svg_candidates if os.path.exists(p)), None)
        if _svg_path:
            self._bug_btn.setIcon(QIcon(_svg_path))
            self._bug_btn.setIconSize(QSize(22, 22))

        self._bug_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #313244;
                border-color: #45475a;
            }
            QPushButton:pressed {
                background-color: #45475a;
            }
        """)
        self._bug_btn.raise_()  # 确保在最上层

        # ── UE4 联动栏折叠按钮（绝对定位，在 bug 按钮下方） ──
        self._ue4_toggle_btn = QPushButton(self)
        self._ue4_toggle_btn.setFixedSize(32, 32)
        self._ue4_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._ue4_toggle_btn.setToolTip("展开 UE4 联动栏")
        self._ue4_toggle_btn.clicked.connect(self._toggle_ue4_bar)

        _ue4_svg_name = "ue4_toggle.svg"
        _ue4_svg_candidates = []
        _ue4_svg_candidates.append(os.path.join(_this_dir, _ue4_svg_name))
        if getattr(sys, 'frozen', False):
            _ue4_svg_candidates.append(os.path.join(os.path.dirname(sys.executable), _ue4_svg_name))
            _ue4_svg_candidates.append(os.path.join(getattr(sys, '_MEIPASS', ''), _ue4_svg_name))
        _ue4_svg_path = next((p for p in _ue4_svg_candidates if os.path.exists(p)), None)
        if _ue4_svg_path:
            self._ue4_toggle_btn.setIcon(QIcon(_ue4_svg_path))
            self._ue4_toggle_btn.setIconSize(QSize(22, 22))

        self._ue4_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #313244;
                border-color: #45475a;
            }
            QPushButton:pressed {
                background-color: #45475a;
            }
        """)
        self._ue4_toggle_btn.raise_()

        # 启动时根据记忆状态决定 UE4 联动栏是否显示（默认折叠）
        _ue4_bar_visible = self._load_ue4_bar_collapsed_state()
        ue4_bar.setVisible(_ue4_bar_visible)
        self._update_ue4_toggle_tooltip(_ue4_bar_visible)

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

        # 保存各 tab 引用，供 UE4 联动接口使用
        self._sprite_tab = sprite_tab
        self._flowmap_tab = flowmap_tab
        self._growth_tab = growth_tab
        self._image_viewer_tab = image_viewer_tab

        # 连接全能看图的「转移至贴图修改」信号（保留兼容）
        image_viewer_tab.transfer_to_texture.connect(self._on_transfer_from_viewer)

        # 连接各板块的跨板块通信信号
        sprite_tab.transfer_signal.connect(self._on_tab_transfer)
        flowmap_tab.transfer_signal.connect(self._on_tab_transfer)
        growth_tab.transfer_signal.connect(self._on_tab_transfer)
        image_viewer_tab.transfer_signal.connect(self._on_tab_transfer)

        # 连接 tab 切换信号，更新 UE4 按钮状态
        tabs.currentChanged.connect(self._on_tab_changed)
        # Left
        left_layout = QVBoxLayout()
        self.preview_label = CheckerLabel(cell=12)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border-radius:10px; border:1px solid #383850;")
        self.preview_label.setMinimumSize(680, 560)
        self.preview_label._on_drop_callback = self.load_image
        self.preview_label._parent_window = self
        self.preview_label._right_click_callback = self._on_preview_right_click

        left_layout.addWidget(self.preview_label, 1)

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

        # 图像调整 + 重置（80%/20%）
        crop_row = QHBoxLayout()
        btn_crop = QPushButton("图像调整")
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

        # 添加自定义底色 + 重置（80%/20%）
        custom_bg_row = QHBoxLayout()
        btn_custom_bg = QPushButton("添加自定义底色")
        btn_custom_bg.setMinimumHeight(34)
        btn_custom_bg.setStyleSheet("text-align:center;")
        btn_custom_bg.setMinimumWidth(340)
        btn_custom_bg.clicked.connect(self.apply_custom_bg_color)

        btn_reset_custom_bg = QPushButton("重置")
        btn_reset_custom_bg.setMinimumHeight(34)
        btn_reset_custom_bg.clicked.connect(self.reset_custom_bg)

        custom_bg_row.addWidget(btn_custom_bg, 4)
        custom_bg_row.addWidget(btn_reset_custom_bg, 1)

        # 去除黑底(Unmult) + 重置（80%/20%）
        unmult_row = QHBoxLayout()
        btn_unmult = QPushButton("去除黑底")
        btn_unmult.setMinimumHeight(34)
        btn_unmult.setStyleSheet("text-align:center;")
        btn_unmult.setMinimumWidth(340)
        btn_unmult.clicked.connect(self.apply_unmult)

        btn_reset_unmult = QPushButton("重置")
        btn_reset_unmult.setMinimumHeight(34)
        btn_reset_unmult.clicked.connect(self.reset_unmult)

        unmult_row.addWidget(btn_unmult, 4)
        unmult_row.addWidget(btn_reset_unmult, 1)

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
        right_layout.addLayout(custom_bg_row)
        right_layout.addLayout(unmult_row)
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

        # ── 隐藏菜单栏（帮助功能已移至左下角 bug 按钮） ──
        self.menuBar().setVisible(False)

        # ── 后台检查更新 ──
        self._start_update_checker()

        # ── 启动 UE4 同步（创建 Named Mutex） ──
        sync_mgr = get_sync_manager()
        sync_mgr.start()

        # ── 启动 UE4 导出监听（接收"发送到皮皮"的贴图） ──
        # 使用信号槽实现线程安全的跨线程通信（后台线程 emit → 主线程 slot）
        self._ue4_export_signal.connect(self._load_from_ue4_safe)
        sync_mgr.start_export_listener(self._on_ue4_export_received)
    # ---------------- 关于弹窗 ----------------
    def _show_about_dialog(self):
        """点击 bug 按钮后弹出「关于」对话框，包含版本信息和检查更新按钮"""
        dlg = QDialog(self)
        dlg.setWindowTitle("关于 PPEditor")
        dlg.setFixedSize(360, 320)
        dlg.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                border: 1px solid #383850;
                border-radius: 12px;
            }
            QLabel {
                color: #cdd6f4;
                background: transparent;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 7px;
                padding: 8px 20px;
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
            }
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(28, 24, 28, 20)

        # 应用名称
        title_label = QLabel("PPEditor")
        title_label.setStyleSheet("font-size: 22px; font-weight: 700; color: #89b4fa;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 版本号
        ver_label = QLabel(f"版本 v{__version__}")
        ver_label.setStyleSheet("font-size: 14px; color: #a6adc8;")
        ver_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(ver_label)

        layout.addSpacing(6)

        # 描述
        desc_label = QLabel(
            "皮皮贴图修改器 —— 一站式游戏贴图工具集\n\n"
            "功能：贴图修改 / 精灵图制作 / 法线绘制\n"
            "　　　灰度图生成 / 全能看图"
        )
        desc_label.setStyleSheet("font-size: 12px; color: #6c7086;")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # 作者
        author_label = QLabel("© evanlumier")
        author_label.setStyleSheet("font-size: 11px; color: #585b70;")
        author_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(author_label)

        layout.addStretch()

        # 检查更新按钮
        btn_check = QPushButton("检查更新")
        btn_check.setCursor(Qt.PointingHandCursor)
        layout.addWidget(btn_check)

        # 状态标签（检查更新反馈）
        self._about_status_label = QLabel("")
        self._about_status_label.setAlignment(Qt.AlignCenter)
        self._about_status_label.setWordWrap(True)
        self._about_status_label.setStyleSheet("font-size: 12px; color: #a6adc8;")
        self._about_status_label.hide()
        layout.addWidget(self._about_status_label)

        def _on_check_clicked():
            btn_check.setEnabled(False)
            btn_check.setText("正在检查...")
            self._about_status_label.hide()
            self._about_dlg = dlg  # 保存对话框引用，用于回调时更新
            self._about_check_btn = btn_check
            self._manual_check_update()

        btn_check.clicked.connect(_on_check_clicked)

        dlg.exec()

        # 对话框关闭后清理引用
        self._about_dlg = None
        self._about_check_btn = None
        self._about_status_label = None

    def _manual_check_update(self):
        """用户手动点击「检查更新」"""
        self._start_update_checker(manual=True)

    # ---------------- 在线更新 ----------------
    def _start_update_checker(self, manual: bool = False):
        """
        在后台线程中检查 GitHub Release 是否有新版本。
        manual=True 时为用户手动触发，会显示「已是最新版本」或「检查失败」提示；
        manual=False 时为启动自动检查，静默失败不打扰用户。
        """
        self._manual_update_check = manual

        # 如果上一个检查线程仍在运行，先等待它结束
        if hasattr(self, '_update_thread') and self._update_thread is not None:
            if self._update_thread.isRunning():
                self._update_thread.wait(2000)
            # 断开旧线程的所有信号连接，防止信号累积
            try:
                self._update_thread.update_found.disconnect()
                self._update_thread.no_update.disconnect()
                self._update_thread.check_failed.disconnect()
            except RuntimeError:
                pass

        class _UpdateThread(QThread):
            update_found = Signal(dict)
            no_update = Signal()
            check_failed = Signal()
            def __init__(self_t, parent, force_check):
                super().__init__(parent)
                self_t._force = force_check
            def run(self_t):
                try:
                    from updater import check_for_update
                    result = check_for_update(force=self_t._force)
                    if result:
                        self_t.update_found.emit(result)
                    else:
                        self_t.no_update.emit()
                except Exception:
                    self_t.check_failed.emit()

        self._update_thread = _UpdateThread(self, force_check=manual)
        self._update_thread.update_found.connect(self._on_update_found)
        self._update_thread.no_update.connect(self._on_no_update)
        self._update_thread.check_failed.connect(self._on_check_failed)
        self._update_thread.start()

    def _on_no_update(self):
        """没有新版本时的回调（仅手动检查时提示）"""
        if getattr(self, '_manual_update_check', False):
            self._manual_update_check = False
            # 在关于对话框内显示反馈
            if getattr(self, '_about_status_label', None) and getattr(self, '_about_dlg', None):
                self._about_status_label.setText("✅ 已是最新版本")
                self._about_status_label.setStyleSheet("font-size: 12px; color: #a6e3a1;")
                self._about_status_label.show()
                if self._about_check_btn:
                    self._about_check_btn.setText("检查更新")
                    self._about_check_btn.setEnabled(True)
            else:
                QMessageBox.information(
                    self, "检查更新",
                    f"当前版本 v{__version__} 已是最新版本。"
                )

    def _on_check_failed(self):
        """检查更新失败的回调（仅手动检查时提示）"""
        if getattr(self, '_manual_update_check', False):
            self._manual_update_check = False
            # 在关于对话框内显示反馈
            if getattr(self, '_about_status_label', None) and getattr(self, '_about_dlg', None):
                self._about_status_label.setText("❌ 无法连接更新服务器，请检查网络")
                self._about_status_label.setStyleSheet("font-size: 12px; color: #f38ba8;")
                self._about_status_label.show()
                if self._about_check_btn:
                    self._about_check_btn.setText("检查更新")
                    self._about_check_btn.setEnabled(True)
            else:
                QMessageBox.warning(
                    self, "检查更新",
                    "无法连接到更新服务器，请检查网络后重试。"
                )

    def _get_skipped_version_path(self) -> str:
        """获取跳过版本记录文件路径"""
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "skipped_version.txt")

    def _get_skipped_version(self) -> str:
        """读取用户选择跳过的版本号"""
        try:
            with open(self._get_skipped_version_path(), "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _set_skipped_version(self, version: str):
        """保存用户选择跳过的版本号"""
        try:
            with open(self._get_skipped_version_path(), "w", encoding="utf-8") as f:
                f.write(version)
        except Exception:
            pass

    def _on_update_found(self, info: dict):
        """收到新版本信息后弹窗提示用户"""
        new_version = info.get("version", "")

        # 自动检查时，如果用户已跳过此版本，则静默忽略
        if not getattr(self, '_manual_update_check', False):
            if new_version and new_version == self._get_skipped_version():
                return

        # 关闭关于对话框（如果还开着）
        if getattr(self, '_about_dlg', None):
            self._about_dlg.accept()

        changelog = info.get("changelog", "暂无更新说明")
        if len(changelog) > 600:
            changelog = changelog[:600] + "\n..."

        # ── 【关卡 1 · UI 引导】协议代际不兼容：引导用户手动下完整包 ──
        # updater.check_for_update 判断远程 __min_compatible_version__ 高于本地版本时，
        # 会在返回 dict 里塞 require_full_install=True。这时增量更新不安全，
        # 弹一个"引导下载完整安装包"的弹窗，避免自动走增量替换踩坑。
        if info.get("require_full_install"):
            guide_msg = QMessageBox(self)
            guide_msg.setWindowTitle("发现新版本（需手动安装）")
            guide_msg.setIcon(QMessageBox.Warning)
            guide_msg.setText(
                f"发现新版本 v{new_version}\n"
                f"当前版本 v{__version__}\n\n"
                "本次更新涉及架构性变更，无法通过在线增量更新完成。\n"
                "请前往 GitHub 手动下载完整安装包并覆盖安装。\n\n"
                f"更新内容：\n{changelog}"
            )
            btn_open = guide_msg.addButton("打开下载页", QMessageBox.AcceptRole)
            btn_cancel = guide_msg.addButton("取消", QMessageBox.RejectRole)
            guide_msg.setDefaultButton(btn_open)
            guide_msg.exec()
            if guide_msg.clickedButton() == btn_open:
                try:
                    import webbrowser
                    webbrowser.open(
                        "https://github.com/evanlumier/PiPiTextureEditor/releases/latest"
                    )
                except Exception:
                    # 打开浏览器失败也不影响主程序继续运行
                    pass
            # 不管用户选哪个，都不再走"立即更新"流程
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("发现新版本")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            f"发现新版本 v{new_version}\n"
            f"当前版本 v{__version__}\n\n"
            f"更新内容：\n{changelog}"
        )
        btn_update = msg.addButton("立即更新", QMessageBox.AcceptRole)
        btn_skip = msg.addButton("跳过此版本", QMessageBox.RejectRole)
        btn_later = msg.addButton("稍后提醒", QMessageBox.NoRole)
        msg.setDefaultButton(btn_update)

        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_update:
            self._do_update(info["download_url"])
        elif clicked == btn_skip:
            self._set_skipped_version(new_version)
            # 清除缓存中的新版本信息，避免后续自动检查再次触发弹窗信号
            from updater import _update_check_cache, _CACHE_NO_UPDATE
            _update_check_cache["result"] = _CACHE_NO_UPDATE

    def _do_update(self, download_url: str):
        """执行下载和更新流程（下载阶段显示详细信息，安装阶段显示进度条）"""
        import threading

        progress = QProgressDialog("准备更新...", "取消", 0, 0, self)
        progress.setWindowTitle("更新中")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumWidth(380)
        progress.show()
        QApplication.processEvents()

        # 用 threading.Event 作为取消信号，后台线程会检查它
        self._stop_event = threading.Event()

        def on_cancel():
            self._stop_event.set()
        progress.canceled.connect(on_cancel)

        # ── 后台线程：负责下载 + 应用更新 ──
        class _UpdateWorkerThread(QThread):
            # 下载进度信号（传递 dict 包含 downloaded_str, speed_str, eta_str 等）
            download_progress = Signal(object)
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

        def on_download_progress(info):
            """下载阶段：显示已下载大小、速度、预计剩余时间等详细信息"""
            if not self._stop_event.is_set() and isinstance(info, dict):
                downloaded_str = info.get("downloaded_str", "")
                speed_str = info.get("speed_str", "")
                elapsed_str = info.get("elapsed_str", "")
                eta_str = info.get("eta_str", "")
                total_str = info.get("total_str", "")
                percent = info.get("percent", -1)
                
                # 如果有总大小信息（urllib 下载），显示百分比进度条
                if percent >= 0:
                    if progress.maximum() == 0:
                        progress.setRange(0, 100)
                    progress.setValue(int(percent * 0.5))  # 下载占前50%
                    label = (f"正在下载更新... {percent}%\n"
                             f"已下载：{downloaded_str} / {total_str}\n"
                             f"速度：{speed_str}　　预计剩余：{eta_str}")
                else:
                    # curl 下载没有总大小信息，使用跑马灯模式
                    label = (f"正在下载更新...\n"
                             f"已下载：{downloaded_str}　　速度：{speed_str}\n"
                             f"已用时间：{elapsed_str}")
                
                progress.setLabelText(label)
                QApplication.processEvents()

        def on_download_done():
            """下载完成，切换到应用更新阶段（恢复为确定进度条）"""
            if not self._stop_event.is_set():
                progress.setRange(0, 100)
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

        def on_finished_ok(bat_path):
            """更新完成，在主线程中启动 bat 脚本并退出"""
            progress.setValue(100)
            progress.setLabelText("更新完成，正在重启...")
            QApplication.processEvents()
            # apply_update 返回的是 bat 脚本路径，需要通过 cmd.exe /c 启动
            import subprocess
            from updater import get_app_dir
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE，隐藏 CMD 窗口
            subprocess.Popen(
                ["cmd.exe", "/c", bat_path],
                cwd=get_app_dir(),
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            QApplication.quit()

        def on_finished_err(err_msg):
            progress.close()
            if not self._stop_event.is_set():
                retry_msg = QMessageBox(self)
                retry_msg.setWindowTitle("更新失败")
                retry_msg.setIcon(QMessageBox.Warning)
                retry_msg.setText(
                    f"更新时出错：\n{err_msg}\n\n"
                    "如反复失败，建议前往 GitHub 手动下载完整安装包。\n"
                    "程序将继续使用当前版本。"
                )
                btn_retry = retry_msg.addButton("重试", QMessageBox.AcceptRole)
                btn_open = retry_msg.addButton("下载完整包", QMessageBox.ActionRole)
                btn_close = retry_msg.addButton("关闭", QMessageBox.RejectRole)
                retry_msg.setDefaultButton(btn_retry)
                retry_msg.exec()
                clicked = retry_msg.clickedButton()
                if clicked == btn_retry:
                    self._do_update(download_url)
                elif clicked == btn_open:
                    # 引导用户前往 Release 页手动下载完整包（关卡 2 二次校验失败时的兜底出口）
                    try:
                        import webbrowser
                        webbrowser.open(
                            "https://github.com/evanlumier/PiPiTextureEditor/releases/latest"
                        )
                    except Exception:
                        pass

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

    # ---------------- UE4 导出接收回调 ----------------
    def _on_ue4_export_received(self, export_data: dict):
        """
        当 UE4 发送贴图到皮皮时的回调（在后台线程 PiPi-ExportListener 中调用）。
        通过 Qt 信号槽机制安全地将数据传递到主线程处理。
        """
        tga_path = export_data.get("tga_path", "")
        asset_name = export_data.get("asset_name", "")
        if not tga_path or not os.path.isfile(tga_path):
            return

        # 通过信号槽跨线程安全地传递到主线程（Qt 自动排队到主线程事件循环）
        self._ue4_export_signal.emit(export_data)

    def _load_from_ue4_safe(self, export_data: dict):
        """信号槽接收端（在主线程中执行），从 export_data 中提取路径并加载。"""
        tga_path = export_data.get("tga_path", "")
        asset_name = export_data.get("asset_name", "")
        asset_path = export_data.get("asset_path", "")
        if not tga_path or not os.path.isfile(tga_path):
            return

        # 根据资产类型路由到对应 Tab（优先通过像素内容分析判断）
        texture_type = self._detect_ue4_texture_type(asset_name, asset_path, tga_path)
        self._load_from_ue4(tga_path, asset_name, asset_path, texture_type)

    @staticmethod
    def _detect_ue4_texture_type(asset_name: str, asset_path: str, tga_path: str = "") -> str:
        """
        综合判断 UE4 贴图类型，优先通过图片像素内容分析，名字/路径作为辅助。

        法线贴图的像素特征：
        - B 通道（z 分量）均值通常很高（> 180），因为大部分法线朝上（z ≈ 1）
        - R 和 G 通道（x, y 分量）均值接近 128（中性值）
        - 三通道标准差相对较低（法线图颜色分布集中）

        返回值：
            "normal"  - 法线贴图
            "default" - 普通贴图（默认）
        """
        import numpy as np

        # ── 第一优先级：像素内容分析 ──
        if tga_path and os.path.isfile(tga_path):
            try:
                img = Image.open(tga_path)

                # ── 前置闸口：透明通道排除 ──
                # 法线图不会有透明通道，如果图片存在大量非完全不透明像素，
                # 则直接跳过像素分析，降级到名字/路径判断
                if img.mode in ("RGBA", "LA", "PA"):
                    alpha = np.array(img.split()[-1], dtype=np.uint8)
                    non_opaque_ratio = np.count_nonzero(alpha < 255) / alpha.size
                    if non_opaque_ratio > 0.05:
                        # 超过 5% 的像素有透明度，不可能是法线图
                        raise Exception("has_alpha_skip_pixel_analysis")

                # 转为 RGB 进行分析
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img, dtype=np.float32)

                r_mean = arr[:, :, 0].mean()
                g_mean = arr[:, :, 1].mean()
                b_mean = arr[:, :, 2].mean()

                # 法线图判定条件：
                # 1. B 通道均值 > 180（法线 z 分量偏高）
                # 2. R 和 G 通道均值在 100~160 范围内（接近中性 128）
                # 3. B 通道均值明显高于 R 和 G（至少高 30）
                is_normal_by_pixel = (
                    b_mean > 180
                    and 80 < r_mean < 175
                    and 80 < g_mean < 175
                    and b_mean - r_mean > 30
                    and b_mean - g_mean > 30
                )

                if is_normal_by_pixel:
                    return "normal"

            except Exception:
                pass  # 图片分析失败，降级到名字判断

        # ── 第二优先级：名字/路径辅助判断 ──
        name = asset_name.strip()
        # UE4 法线贴图命名惯例：以 _N 结尾（如 T_Rock_N）
        if name.endswith("_N") or name.endswith("_Normal"):
            return "normal"
        # 路径中包含 Normal 目录
        if "Normal" in asset_path or "/N/" in asset_path:
            return "normal"

        return "default"

    def _load_from_ue4(self, tga_path: str, asset_name: str, asset_path: str = "", texture_type: str = "default"):
        """
        在主线程中加载 UE4 发送过来的贴图，根据类型路由到对应 Tab。

        参数：
            tga_path: TGA 文件路径
            asset_name: 资产名称
            asset_path: UE4 中的资产目录路径（如 /Game/Art/Characters/Hero）
            texture_type: 贴图类型（"normal" / "default"）
        """
        try:
            if texture_type == "normal":
                # 法线贴图 → 路由到法线绘制 Tab（index 2）
                self._load_normal_from_ue4(tga_path, asset_name)
            else:
                # 默认 → 加载到贴图修改 Tab（index 0）
                self._tabs.setCurrentIndex(0)
                self.load_image(tga_path)

            # 记录 UE4 来源信息（用于导出时覆盖原文件）
            # 根据贴图类型确定目标 Tab 索引
            target_tab_idx = 2 if texture_type == "normal" else 0

            if asset_path and asset_path.startswith("/Game"):
                # UE4 端发送的 asset_path 可能是以下格式之一：
                # 1. 完整资产路径：/Game/Art/Tex/T_123_3
                # 2. UE4 ObjectPath：/Game/Art/Tex/T_123_3.T_123_3
                # 3. 纯目录路径：/Game/Art/Tex
                # 需要提取目录部分作为 target_path（send_import_request 期望目录路径）

                # 先处理 ObjectPath 格式（去掉 .AssetName 后缀）
                clean_path = asset_path.split(".")[0] if "." in asset_path else asset_path
                clean_path = clean_path.rstrip("/")

                # 提取最后一段
                last_segment = clean_path.split("/")[-1] if "/" in clean_path else ""

                # 如果最后一段等于 asset_name，说明是完整路径，需要去掉资产名
                if asset_name and last_segment == asset_name:
                    dir_path = "/".join(clean_path.split("/")[:-1])
                else:
                    dir_path = clean_path

                # 确保 dir_path 不为空
                if not dir_path or dir_path == "/Game":
                    dir_path = clean_path

                import logging
                logging.getLogger("PiPi").debug(
                    f"UE4 asset_path 解析: 原始={asset_path}, 清理后={clean_path}, "
                    f"目录={dir_path}, 资产名={asset_name}"
                )

                # 按 Tab 索引独立存储来源信息
                self._ue4_source_info_per_tab[target_tab_idx] = {
                    "asset_path": dir_path,
                    "asset_name": asset_name,
                }
                # 按 Tab 索引独立存储目标路径
                self._ue4_target_path_per_tab[target_tab_idx] = dir_path
                # 自动填充 UE4 目标路径文本框（填入目录路径）
                self.ue4_target_path.setText(dir_path)
                self._save_ue4_target_path(target_tab_idx, dir_path)
                # 更新命名预览，显示 UE4 原始资产名称
                self.update_name_preview()
            else:
                # 清除对应 Tab 的来源信息
                self._ue4_source_info_per_tab.pop(target_tab_idx, None)

            # 将窗口带到前台
            self.raise_()
            self.activateWindow()
        except Exception as e:
            self.statusBar().showMessage(f"UE4 贴图加载失败: {e}", 5000)
        finally:
            # 清理 TGA 临时文件（已加载到内存，不再需要磁盘文件）
            try:
                if os.path.exists(tga_path):
                    os.remove(tga_path)
            except OSError:
                pass

    def _load_normal_from_ue4(self, tga_path: str, asset_name: str):
        """
        从 UE4 接收法线贴图并加载到法线绘制 Tab 的法线结果区（normal_map）。
        - 将法线图 RGB 解码为 float32 法线向量写入 canvas.normal_map
        - 自动打开"显示全图"以便用户查看导入结果
        - 参考图区域保持不变（用于放置正常图片）
        """
        # 切换到法线绘制 Tab
        self._tabs.setCurrentIndex(2)
        import numpy as np
        try:
            img = Image.open(tga_path).convert("RGB")
            w, h = img.size

            # 将图片 resize 到画布尺寸
            canvas = self._flowmap_tab.canvas
            cw, ch = canvas._cw, canvas._ch
            if w != cw or h != ch:
                img = img.resize((cw, ch), Image.LANCZOS)

            # 将 RGB packed 格式解码为 float32 法线向量
            # packed: r = x*0.5+0.5, g = y*0.5+0.5, b = z*0.5+0.5
            # 解码: x = r/255*2-1, y = g/255*2-1, z = b/255*2-1
            arr = np.array(img, dtype=np.float32) / 255.0
            nm = arr * 2.0 - 1.0  # HxWx3, 值域 [-1, 1]

            # UE4 导出的法线贴图是 DirectX 格式（G 通道翻转，Y 轴朝下）
            # 如果当前画布是 DirectX 模式，需要翻转 G 通道回来
            # 因为 canvas.normal_map 存储的是标准法线向量（Y 轴朝上）
            # DirectX 格式的 G 通道 = -Y，所以需要翻转
            if canvas.mode_dx:
                nm[:, :, 1] = -nm[:, :, 1]

            # normalize 确保向量长度为 1
            length = np.sqrt(np.sum(nm ** 2, axis=2, keepdims=True))
            length = np.maximum(length, 1e-6)
            nm = nm / length

            # 保存 undo 快照，然后写入 normal_map
            canvas._push_undo()
            canvas.normal_map = nm
            canvas._flow_cache_dirty = True
            canvas._normal_vis_dirty = True
            canvas.update()

            # 通知左侧结果区更新
            if canvas.on_normal_updated:
                canvas.on_normal_updated(canvas.normal_map)

            # 自动打开"显示全图"以便用户查看导入的法线图
            self._flowmap_tab.chk_show_all.setChecked(True)
        except Exception as e:
            self.statusBar().showMessage(f"UE4 法线贴图加载失败: {e}", 5000)

    # ---------------- file ----------------
    def load_image(self, path: str):
        try:
            img = Image.open(path).convert("RGBA")
            self.src_path = path

            # 本地打开新图片时，清除贴图修改 Tab 的 UE4 来源信息（断开与 UE4 原始资产的关联）
            # 注意：从 UE4 导入时，_load_from_ue4 会在 load_image 之后重新设置来源信息
            self._ue4_source_info_per_tab.pop(0, None)

            self.source_color = img
            self._pre_crop_color = img.copy()  # 保留图像调整前的原始图（用于重置）
            self.master_color = img.copy()

            self.is_bw = False
            self.has_unmult = False
            self.custom_bg_color = None
            self.target_size = None
            self.output_basename = None

            # reset sliders/spin to 100
            _block_widgets = (self.brightness_slider, self.contrast_slider)
            try:
                for w in _block_widgets:
                    w.blockSignals(True)

                self.brightness_slider.setValue(100)
                self.contrast_slider.setValue(100)
                self.brightness_spin.setText("100%")
                self.contrast_spin.setText("100%")
            finally:
                for w in _block_widgets:
                    w.blockSignals(False)

            self.name_input.blockSignals(True)
            try:
                self.name_input.setText("")
            finally:
                self.name_input.blockSignals(False)
            self.update_name_preview()

            self.rebuild_working()
            self.set_enabled(True)
            self.update_info()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：\n{e}")

    def _on_transfer_from_viewer(self, tmp_png_path: str):
        """从全能看图 tab 接收转移过来的图片（临时 PNG 文件）"""
        self._receive_to_texture(tmp_png_path)

    def _on_tab_transfer(self, tmp_png_path: str, target_tab: int):
        """统一处理各板块发出的跨板块通信信号"""
        self._route_to_tab(target_tab, tmp_png_path)

    # ── 跨板块通信：发送 & 接收 ──────────────────────────────────────

    def _on_preview_right_click(self, global_pos):
        """贴图修改板块：右键单击预览区弹出发送菜单"""
        # 没有图片时不弹菜单
        if self.preview_label.pixmap() is None:
            return
        menu = build_send_menu(self, TAB_TEXTURE, self._send_image_to)
        menu.exec(global_pos)

    def _send_image_to(self, target_tab: int):
        """贴图修改板块：将当前预览图发送到目标板块"""
        pm = self.preview_label.pixmap()
        pil_img = qpixmap_to_pil(pm)
        if pil_img is None:
            return
        tmp_path = pil_to_temp_png(pil_img, prefix="texture_send_")
        if tmp_path is None:
            return
        self._route_to_tab(target_tab, tmp_path)

    def _route_to_tab(self, target_tab: int, tmp_png_path: str):
        """统一路由：将临时 PNG 文件发送到目标板块并切换"""
        try:
            if target_tab == TAB_TEXTURE:
                self._receive_to_texture(tmp_png_path)
            elif target_tab == TAB_SPRITE:
                self._receive_to_sprite(tmp_png_path)
            elif target_tab == TAB_FLOWMAP:
                self._receive_to_flowmap(tmp_png_path)
            elif target_tab == TAB_GRAYGROWTH:
                self._receive_to_growth(tmp_png_path)
            elif target_tab == TAB_VIEWER:
                self._receive_to_viewer(tmp_png_path)
        except Exception as e:
            QMessageBox.critical(self, "发送失败", f"发送图片时出错：\n{e}")
            # 清理临时文件
            try:
                if os.path.exists(tmp_png_path):
                    os.remove(tmp_png_path)
            except OSError:
                pass

    def _receive_to_texture(self, tmp_png_path: str):
        """接收图片到贴图修改板块"""
        try:
            self.load_image(tmp_png_path)
            self._tabs.setCurrentIndex(TAB_TEXTURE)
        except Exception as e:
            QMessageBox.critical(self, "接收失败", f"加载图片时出错：\n{e}")
        finally:
            try:
                if os.path.exists(tmp_png_path):
                    os.remove(tmp_png_path)
            except OSError:
                pass

    def _receive_to_sprite(self, tmp_png_path: str):
        """接收图片到精灵图板块（追加为新帧）"""
        try:
            self._sprite_tab.add_paths([tmp_png_path])
            self._tabs.setCurrentIndex(TAB_SPRITE)
        except Exception as e:
            QMessageBox.critical(self, "接收失败", f"加载图片到精灵图时出错：\n{e}")

    def _receive_to_flowmap(self, tmp_png_path: str):
        """接收图片到法线绘制板块（作为参考图）"""
        try:
            self._flowmap_tab.drop_ref._load(tmp_png_path)
            self._tabs.setCurrentIndex(TAB_FLOWMAP)
        except Exception as e:
            QMessageBox.critical(self, "接收失败", f"加载参考图时出错：\n{e}")
        finally:
            try:
                if os.path.exists(tmp_png_path):
                    os.remove(tmp_png_path)
            except OSError:
                pass

    def _receive_to_growth(self, tmp_png_path: str):
        """接收图片到灰度图板块（作为单图素材）"""
        try:
            self._growth_tab._load_single_from_path(tmp_png_path)
            self._tabs.setCurrentIndex(TAB_GRAYGROWTH)
        except Exception as e:
            QMessageBox.critical(self, "接收失败", f"加载素材图时出错：\n{e}")
        finally:
            try:
                if os.path.exists(tmp_png_path):
                    os.remove(tmp_png_path)
            except OSError:
                pass

    def _receive_to_viewer(self, tmp_png_path: str):
        """接收图片到全能看图板块（完全替换当前状态）"""
        try:
            self._image_viewer_tab.receive_image(tmp_png_path)
            self._tabs.setCurrentIndex(TAB_VIEWER)
        except Exception as e:
            QMessageBox.critical(self, "接收失败", f"加载图片到全能看图时出错：\n{e}")
        finally:
            try:
                if os.path.exists(tmp_png_path):
                    os.remove(tmp_png_path)
            except OSError:
                pass

    # ---------------- crop ----------------
    def open_crop_dialog(self):
        if self.source_color is None:
            return
        dlg = CropDialog(self.source_color, self)
        if dlg.exec() == QDialog.Accepted and dlg.result_img is not None:
            result = dlg.result_img.convert("RGBA")
            # 图像调整是破坏性操作（改变尺寸/几何），确认后更新基准图
            self.source_color = result
            self.master_color = result.copy()
            self.rebuild_working()

    def reset_crop(self):
        # 重置图像调整：回到图像调整前的原始状态
        if self._pre_crop_color is None:
            return
        self.source_color = self._pre_crop_color.copy()
        self.master_color = self._pre_crop_color.copy()
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

    # ---------------- custom bg color ----------------
    def apply_custom_bg_color(self):
        """弹出取色器，选择自定义底色并合成"""
        if self.master_color is None:
            return
        color = QColorDialog.getColor(Qt.black, self, "选择底色")
        if color.isValid():
            self.custom_bg_color = (color.red(), color.green(), color.blue())
            self.rebuild_working()

    def reset_custom_bg(self):
        if self.master_color is None:
            return
        self.custom_bg_color = None
        self.rebuild_working()

    # ---------------- unmult ----------------
    def apply_unmult(self):
        """去除黑底（Unmult）：根据 Alpha 通道反推原始颜色，消除预乘黑底"""
        if self.master_color is None:
            return
        self.has_unmult = True
        self.rebuild_working()

    def reset_unmult(self):
        if self.master_color is None:
            return
        self.has_unmult = False
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

    # ===== 导出路径记忆（继承自 ExportDirMixin）=====
    _export_dir_cache_name = "last_export_dir.txt"

    def _get_default_export_dir(self) -> str:
        return os.path.dirname(self.src_path) if self.src_path else ""

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
        # 优先显示 UE4 来源名称（如果有且用户未手动命名）
        tab0_source = self._ue4_source_info_per_tab.get(0)
        if not self.output_basename and tab0_source and tab0_source.get("asset_name"):
            ue4_name = tab0_source["asset_name"]
            self.name_preview.setText(f"预览：{ue4_name}（UE4 原始名称）")
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
        if self.output_basename:
            return self.output_basename
        # 如果有 UE4 来源信息，优先使用 UE4 资产名称
        tab0_source = self._ue4_source_info_per_tab.get(0)
        if tab0_source and tab0_source.get("asset_name"):
            return tab0_source["asset_name"]
        return self.original_base()

    def validate_export_name(self, name: str) -> bool:
        return bool(VALID_NAME_RE.fullmatch(name))

    # ---------------- pipeline ----------------
    def rebuild_working(self):
        if self.master_color is None:
            return

        base = self.master_color

        if self.is_bw:
            base = to_bw_rgba(base)

        if self.has_unmult:
            # Unmult（去除黑底）：Alpha = min(原始Alpha, max(R, G, B))
            # 原理：纯黑(0,0,0)→透明，亮色→不透明，与 AE Unmult 插件一致
            # 使用 min() 确保不会覆盖已有的透明度（如羽化渐变）
            import numpy as np
            arr = np.array(base, dtype=np.uint8)
            rgb = arr[:, :, :3]
            original_alpha = arr[:, :, 3]
            unmult_alpha = np.max(rgb, axis=2)  # max(R, G, B) 作为 unmult alpha
            # 取两者较小值：保留羽化等已有透明度，同时去除黑底
            new_alpha = np.minimum(original_alpha, unmult_alpha)
            result = np.dstack([rgb, new_alpha])
            base = Image.fromarray(result, "RGBA")

        if self.target_size is not None:
            w, h = self.target_size
            base = base.resize((int(w), int(h)), resample=Image.LANCZOS)

        self.working_img = base

        # 生成预览缩略图（限制最大边长，避免大图卡顿）
        w, h = base.size
        max_side = self._PREVIEW_MAX
        if w > max_side or h > max_side:
            ratio = min(max_side / w, max_side / h)
            tw, th = int(w * ratio), int(h * ratio)
            self._preview_thumb = base.resize((tw, th), resample=Image.LANCZOS)
        else:
            self._preview_thumb = base

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

    @property
    def preview_img(self) -> Optional[Image.Image]:
        """懒加载全尺寸预览图（带亮度/对比度+底色），仅在导出/遮罩等需要时才计算"""
        if self.working_img is None:
            return None
        if self._preview_dirty or self._preview_full_cache is None:
            b = self.brightness_slider.value() / 100.0
            c = self.contrast_slider.value() / 100.0
            full = self.working_img.copy()
            full = ImageEnhance.Brightness(full).enhance(b)
            full = ImageEnhance.Contrast(full).enhance(c)
            # 底色合成放在亮度/对比度之后，确保底色不受调整影响
            if self.custom_bg_color is not None:
                bg = Image.new("RGBA", full.size, (*self.custom_bg_color, 255))
                bg.paste(full, mask=full.split()[3])
                bg.putalpha(255)
                full = bg
            self._preview_full_cache = full
            self._preview_dirty = False
        return self._preview_full_cache

    def update_preview(self):
        if self.working_img is None:
            self.preview_label.clear()
            return

        # 标记全尺寸缓存已过期，下次访问 preview_img 时才重新生成
        self._preview_dirty = True

        b = self.brightness_slider.value() / 100.0
        c = self.contrast_slider.value() / 100.0

        # 用缩略图做预览渲染（大幅提升大图性能）
        thumb = self._preview_thumb if self._preview_thumb is not None else self.working_img
        img = thumb.copy()
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        # 底色合成放在亮度/对比度之后，确保底色不受调整影响
        if self.custom_bg_color is not None:
            bg = Image.new("RGBA", img.size, (*self.custom_bg_color, 255))
            bg.paste(img, mask=img.split()[3])
            bg.putalpha(255)
            img = bg

        pix = pil_to_qpixmap(img)
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
                    bg = Image.new("RGB", out.size, (255, 255, 255))
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
                bg = Image.new("RGB", out.size, (255, 255, 255))
                bg.paste(out, mask=out.split()[-1])
                bg.save(path, quality=95)
            else:
                out.save(path)
            self._save_last_export_dir(path)
            QMessageBox.information(self, "完成", f"已导出：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")

    # ---------------- UE4 联动栏折叠/展开 ----------------
    def _toggle_ue4_bar(self):
        """切换 UE4 联动栏的显示/隐藏状态"""
        visible = not self._ue4_bar.isVisible()

        # 锁定窗口大小，防止 setVisible 触发布局系统撑大/缩小窗口
        current_h = self.height()
        self.setFixedHeight(current_h)

        self._ue4_bar.setVisible(visible)

        # 解除固定高度限制，允许用户后续自由调整窗口大小
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)

        self._update_ue4_toggle_tooltip(visible)
        self._save_ue4_bar_collapsed_state(visible)
        # 重新定位按钮（因为 ue4_bar 高度变化）
        self._reposition_bug_btn()

    def _update_ue4_toggle_tooltip(self, bar_visible: bool):
        """根据当前状态更新折叠按钮的 tooltip"""
        if bar_visible:
            self._ue4_toggle_btn.setToolTip("收起 UE4 联动栏")
        else:
            self._ue4_toggle_btn.setToolTip("展开 UE4 联动栏")

    def _load_ue4_bar_collapsed_state(self) -> bool:
        """加载 UE4 联动栏的显示状态，返回 True 表示可见，False 表示折叠。默认折叠。"""
        try:
            appdata = os.getenv("APPDATA") or ""
            folder = os.path.join(appdata, "GUITextureEditor")
            filepath = os.path.join(folder, "ue4_bar_visible.txt")
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip() == "1"
        except Exception:
            return False  # 默认折叠

    def _save_ue4_bar_collapsed_state(self, visible: bool):
        """保存 UE4 联动栏的显示状态"""
        try:
            appdata = os.getenv("APPDATA") or ""
            folder = os.path.join(appdata, "GUITextureEditor")
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "ue4_bar_visible.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("1" if visible else "0")
        except Exception:
            pass

    # ---------------- UE4 联动 ----------------
    def _get_ue4_target_path_cache(self, tab_idx: int = 0) -> str:
        """UE4 目标路径缓存文件路径（按 Tab 索引独立存储）"""
        appdata = os.getenv("APPDATA") or ""
        folder = os.path.join(appdata, "GUITextureEditor")
        os.makedirs(folder, exist_ok=True)
        # Tab 0 保持原文件名兼容旧版本，其他 Tab 加后缀
        if tab_idx == 0:
            return os.path.join(folder, "ue4_target_path.txt")
        return os.path.join(folder, f"ue4_target_path_tab{tab_idx}.txt")

    def _load_ue4_target_path(self, tab_idx: int = 0) -> str:
        """加载上次保存的 UE4 目标路径（按 Tab 索引）"""
        try:
            with open(self._get_ue4_target_path_cache(tab_idx), "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _save_ue4_target_path(self, tab_idx: int, path: str):
        """保存 UE4 目标路径（按 Tab 索引）"""
        try:
            with open(self._get_ue4_target_path_cache(tab_idx), "w", encoding="utf-8") as f:
                f.write(path)
        except Exception:
            pass

    # ── 贴图修改 tab 的 UE4 联动接口 ──────────────────────────────────
    def get_ue4_export_image(self) -> Optional[Image.Image]:
        """贴图修改 tab：返回当前预览图（带亮度/对比度），没有则返回 None。"""
        return self.preview_img

    def get_ue4_export_name(self) -> str:
        """贴图修改 tab：返回导出到 UE4 时的资产名称。
        优先级：用户手动命名 > UE4 来源名称 > 文件原始名称
        （逻辑与 get_export_basename 一致，直接委托）
        """
        return self.get_export_basename()

    # ── Tab 切换回调 ──────────────────────────────────────────────────
    def _on_tab_changed(self, index: int):
        """Tab 切换时：保存离开 Tab 的路径 → 恢复目标 Tab 的路径 → 更新按钮状态。"""
        # 保存离开 Tab 的当前路径（用户可能手动修改了路径文本框）
        prev_idx = getattr(self, '_prev_tab_idx', None)
        if prev_idx is not None:
            current_text = self.ue4_target_path.text().strip()
            if current_text:
                self._ue4_target_path_per_tab[prev_idx] = current_text

        # 恢复目标 Tab 的 UE4 目标路径（如果有独立存储的路径）
        saved_path = self._ue4_target_path_per_tab.get(index, "")
        if saved_path:
            self.ue4_target_path.setText(saved_path)

        # 记录当前 Tab 索引，供下次切换时保存路径
        self._prev_tab_idx = index
        self._update_ue4_btn_state()

    def _update_ue4_btn_state(self):
        """根据当前 tab 是否有可导出图片来更新 UE4 按钮状态。"""
        tab_widget = self._get_current_ue4_tab()
        if tab_widget is None:
            # 全能看图 tab 没有导出功能
            self.btn_export_ue4.setEnabled(False)
            self.btn_export_ue4.setToolTip("当前页签不支持导入 UE4")
        else:
            self.btn_export_ue4.setEnabled(True)
            self.btn_export_ue4.setToolTip("")

    def _get_current_ue4_tab(self):
        """
        获取当前 tab 对应的 UE4 联动对象。
        贴图修改 tab 返回 self（MainWindow），其他 tab 返回 tab 实例。
        全能看图 tab 返回 None（不支持导出）。
        """
        idx = self._tabs.currentIndex()
        if idx == 0:
            return self  # 贴图修改 tab，接口在 MainWindow 上
        elif idx == 1:
            return self._sprite_tab
        elif idx == 2:
            return self._flowmap_tab
        elif idx == 3:
            return self._growth_tab
        else:
            return None  # 全能看图 tab，不支持

    def export_to_ue4(self):
        """将当前 tab 的图片导入到 UE4（所有 tab 共享此方法）"""
        # 获取当前 tab 的 UE4 联动对象
        tab_obj = self._get_current_ue4_tab()
        if tab_obj is None:
            QMessageBox.warning(self, "提示", "当前页签不支持导入 UE4")
            return

        # 获取可导出图片
        export_img = tab_obj.get_ue4_export_image()
        if export_img is None:
            QMessageBox.warning(self, "提示", "当前没有可导出的图片，请先生成或导入内容")
            return

        export_name = tab_obj.get_ue4_export_name()

        # 非贴图修改 Tab：来源信息存储在 MainWindow 中，Tab 自身无法感知，
        # 需要在此处补充来源名称覆盖（仅当 Tab 没有用户自定义名称时）
        current_tab_idx = self._tabs.currentIndex()
        current_source_info = self._ue4_source_info_per_tab.get(current_tab_idx)
        if current_tab_idx != 0 and current_source_info and current_source_info.get("asset_name"):
            tab_has_custom_name = getattr(tab_obj, '_output_basename', None)
            if not tab_has_custom_name:
                export_name = current_source_info["asset_name"]
        target_path = self.ue4_target_path.text().strip()
        if not target_path:
            QMessageBox.warning(self, "提示", "请填写 UE4 目标路径（如 /Game/Art/UI/Textures）")
            return

        if not target_path.startswith("/Game"):
            QMessageBox.warning(self, "提示", "UE4 路径必须以 /Game 开头")
            return

        # ── UE4 来源信息检查：改名警告 + 路径变更检测 ──
        if current_source_info:
            source_name = current_source_info.get("asset_name", "")
            source_path = current_source_info.get("asset_path", "")

            name_changed = source_name and export_name != source_name
            path_changed = source_path and target_path != source_path

            if name_changed or path_changed:
                # 构建提示信息
                msg_parts = []
                if name_changed:
                    msg_parts.append(f"• 资产名称已更改：{source_name} → {export_name}")
                if path_changed:
                    msg_parts.append(f"• 目标路径已更改：{source_path} → {target_path}")

                msg = (
                    "检测到导出信息与 UE4 原始来源不同：\n\n"
                    + "\n".join(msg_parts) + "\n\n"
                    "⚠️ 使用新名称/路径导出将创建新资产，原资产仍保留在 UE4 中。\n"
                    "如果使用 Perforce，请注意手动处理原资产，避免服务器残留。\n\n"
                    "请选择导出方式："
                )

                box = QMessageBox(self)
                box.setWindowTitle("导出确认")
                box.setText(msg)
                box.setIcon(QMessageBox.Warning)
                btn_original = box.addButton(
                    f"覆盖原资产（{source_name}）", QMessageBox.AcceptRole)
                btn_new = box.addButton(
                    "使用当前设置导出", QMessageBox.DestructiveRole)
                btn_cancel = box.addButton("取消", QMessageBox.RejectRole)
                box.setDefaultButton(btn_original)
                box.exec_()

                clicked = box.clickedButton()
                if clicked == btn_cancel:
                    return
                elif clicked == btn_original:
                    # 使用原始名称和路径覆盖
                    export_name = source_name
                    target_path = source_path
                    self.ue4_target_path.setText(source_path)
                elif clicked == btn_new:
                    # 用户确认使用新名称/路径，清除当前 Tab 的来源信息（后续不再弹窗）
                    self._ue4_source_info_per_tab.pop(current_tab_idx, None)

        # 保存路径（按当前 Tab 索引独立存储）
        self._ue4_target_path_per_tab[current_tab_idx] = target_path
        self._save_ue4_target_path(current_tab_idx, target_path)

        # 检查同步管理器
        sync_mgr = get_sync_manager()
        if not sync_mgr.is_running():
            sync_mgr.start()

        if not sync_mgr.is_ue4_available():
            QMessageBox.warning(
                self, "UE4 未就绪",
                "无法找到 UE4 Sync 目录。\n"
                "请确认：\n"
                "1. UE4 编辑器已启动\n"
                "2. AssetWorkflowToolkit 插件已加载\n"
                "3. 皮皮联动功能已启用"
            )
            return

        # 导出 PNG 到临时文件
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="pipi_ue4_")
        png_path = os.path.join(tmp_dir, f"{export_name}.png")

        try:
            # 确保导出为 RGB/RGBA 格式，UE4 不一定能正确处理单通道灰度 PNG
            if export_img.mode == "L":
                export_img = export_img.convert("RGB")
            elif export_img.mode == "LA":
                export_img = export_img.convert("RGBA")
            export_img.save(png_path, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出临时 PNG 失败：\n{e}")
            return

        # 发送导入请求
        self.btn_export_ue4.setEnabled(False)
        self.btn_export_ue4.setText("正在导入...")
        QApplication.processEvents()

        try:
            success, message = sync_mgr.send_import_request(
                png_path=png_path,
                target_path=target_path,
                asset_name=export_name,
            )

            if success:
                QMessageBox.information(self, "导入成功", f"贴图已导入 UE4：\n{target_path}/{export_name}")
            else:
                QMessageBox.warning(self, "导入失败", f"{message}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入 UE4 时出错：\n{e}")
        finally:
            self.btn_export_ue4.setEnabled(True)
            self.btn_export_ue4.setText("导入UE4")

            # 清理临时文件
            try:
                if os.path.exists(png_path):
                    os.remove(png_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def _reposition_bug_btn(self):
        """将 bug 按钮和 UE4 折叠按钮固定在窗口左下角（从底部向上排列）"""
        if not hasattr(self, '_bug_btn'):
            return
        tab_bar = self._tabs.tabBar()
        bar_pos = tab_bar.mapTo(self, QPoint(0, 0))
        # 计算 UE4 联动栏高度
        ue4_bar_h = self._ue4_bar.height() if (hasattr(self, '_ue4_bar') and self._ue4_bar.isVisible()) else 0
        # 水平居中于 tab bar 列
        x = bar_pos.x() + (tab_bar.width() - self._bug_btn.width()) // 2

        # 从底部向上排列：UE4 折叠按钮在最底部，bug 按钮在其上方
        bottom_margin = 10
        if hasattr(self, '_ue4_toggle_btn'):
            toggle_btn = self._ue4_toggle_btn
            toggle_y = self.height() - toggle_btn.height() - bottom_margin - ue4_bar_h
            toggle_btn.move(x, toggle_y)
            toggle_btn.raise_()
            # bug 按钮在折叠按钮上方
            bug_y = toggle_y - self._bug_btn.height() - 4
        else:
            bug_y = self.height() - self._bug_btn.height() - bottom_margin - ue4_bar_h

        self._bug_btn.move(x, bug_y)
        self._bug_btn.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_bug_btn()
        # 防抖：避免拖拽调整窗口大小时频繁重绘
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(50)
            self._resize_timer.timeout.connect(self._on_resize_done)
        if self.working_img is not None:
            self._resize_timer.start()

    def _on_resize_done(self):
        """防抖结束后执行一次预览刷新"""
        if self.working_img is not None:
            self.update_preview()

    def showEvent(self, event):
        super().showEvent(event)
        # 首次显示时 tab bar 布局可能尚未完成，延迟一帧再定位
        QTimer.singleShot(0, self._reposition_bug_btn)

    def closeEvent(self, event):
        """窗口关闭时，确保后台线程安全结束，防止退出崩溃"""
        # 停止 UE4 同步（释放 Named Mutex）
        try:
            sync_mgr = get_sync_manager()
            if sync_mgr.is_running():
                sync_mgr.stop()
        except Exception:
            pass

        # 先统一发送停止信号，让所有后台线程尽快退出
        if hasattr(self, '_stop_event'):
            self._stop_event.set()

        # 等待更新检查线程结束
        if hasattr(self, '_update_thread') and self._update_thread is not None:
            if self._update_thread.isRunning():
                self._update_thread.quit()
                self._update_thread.wait(3000)
        # 等待下载线程结束
        if hasattr(self, '_dl_thread') and self._dl_thread is not None:
            if self._dl_thread.isRunning():
                self._dl_thread.quit()
                self._dl_thread.wait(3000)
        super().closeEvent(event)


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
    # 日志文件写到 exe 所在根目录（打包环境）或脚本上级目录（开发环境 app/ 下）
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        _this = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(_this) if os.path.basename(_this) == "app" else _this
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
    # 切换工作目录到脚本所在目录，确保资源文件的相对路径正确
    # （launcher.py 已经 chdir 到 app/ 目录，这里做兜底确认）
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        app = QApplication(sys.argv)

        # 禁用所有文本输入框的右键上下文菜单（Undo/Redo/Cut/Copy/Paste）
        from PySide6.QtCore import QEvent, QObject

        class _NoContextMenuFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.ContextMenu:
                    if isinstance(obj, (QLineEdit, QSpinBox)):
                        return True  # 拦截，不弹出菜单
                return super().eventFilter(obj, event)

        _ctx_filter = _NoContextMenuFilter(app)
        app.installEventFilter(_ctx_filter)
        # 给整个应用设置图标（任务栏图标，多路径回退查找）
        _ico_name = "TextureToolGUI.ico"
        _candidates = []
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _candidates.append(os.path.join(_this_dir, _ico_name))
        if getattr(sys, 'frozen', False):
            # 打包环境：ico 在 exe 同级目录和 _MEIPASS 中
            _candidates.append(os.path.join(os.path.dirname(sys.executable), _ico_name))
            _candidates.append(os.path.join(getattr(sys, '_MEIPASS', ''), _ico_name))
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
