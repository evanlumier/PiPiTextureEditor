# -*- coding: utf-8 -*-
"""
theme.py — 皮皮贴图修改器 主题系统

职责：
  1. 定义所有颜色/字体/圆角/间距的变量
  2. 提供 generate_qss() 函数，生成完整的全局 QSS 样式表
  3. 提供 T（当前主题实例）供代码中动态引用颜色
  4. 支持多套主题切换（改 current_theme 即可）

使用方式：
  from theme import T, generate_qss
  # 启动时：
  app.setStyleSheet(generate_qss())
  # 代码中引用颜色：
  label.setStyleSheet(f"color: {T.success};")
"""


class Theme:
    """主题配置基类 — 所有主题必须实现这些属性"""

    # ── 背景层级（从深到浅）──
    bg_base: str        # 最底层（窗口背景）
    bg_surface: str     # 面板/侧边栏
    bg_overlay: str     # 卡片/弹窗
    bg_elevated: str    # hover / 浮起元素
    bg_input: str       # 输入框背景

    # ── 文字 ──
    text_primary: str   # 主文字
    text_secondary: str # 次要文字
    text_tertiary: str  # 三级文字（提示/占位符）
    text_disabled: str  # 禁用态文字
    text_hint: str      # 极淡提示文字

    # ── 主色调 ──
    accent: str         # 主强调色
    accent_hover: str   # 强调色 hover
    accent_pressed: str # 强调色 pressed
    success: str        # 成功/确认
    error: str          # 错误/删除/危险
    warning: str        # 警告
    info: str           # 信息/链接

    # ── 边框/分割线 ──
    border: str         # 常规边框
    border_subtle: str  # 微妙分割线
    border_focus: str   # 聚焦边框

    # ── 尺寸 ──
    radius: int = 8           # 默认圆角 (px)
    radius_sm: int = 4        # 小圆角
    radius_lg: int = 10       # 大圆角
    font_family: str = '"Segoe UI", "Microsoft YaHei UI", sans-serif'
    font_size: int = 12       # 基础字号 (px)
    font_size_sm: int = 11    # 小字号
    font_size_xs: int = 10    # 极小字号
    font_size_lg: int = 13    # 大字号
    font_size_title: int = 22 # 标题字号
    spacing: int = 8          # 基础间距 (px)


class CatppuccinMocha(Theme):
    """
    Catppuccin Mocha 主题 — 当前项目默认配色
    保持与现有 UI 完全一致的视觉效果
    """
    # 背景
    bg_base = "#1e1e2e"
    bg_surface = "#181825"
    bg_overlay = "#313244"
    bg_elevated = "#45475a"
    bg_input = "#1e1e2e"

    # 文字
    text_primary = "#cdd6f4"
    text_secondary = "#a6adc8"
    text_tertiary = "#6c7086"
    text_disabled = "#585b70"
    text_hint = "#45475a"

    # 主色调
    accent = "#89b4fa"
    accent_hover = "#b4d0fb"
    accent_pressed = "#5e9bf7"
    success = "#a6e3a1"
    error = "#f38ba8"
    warning = "#f9e2af"
    info = "#89dceb"

    # 边框
    border = "#45475a"
    border_subtle = "#313244"
    border_focus = "#89b4fa"

    # 尺寸
    radius = 8
    radius_sm = 4
    radius_lg = 10


class FigmaDark(Theme):
    """
    Figma 深色主题 — 新风格
    特点：纯灰度层级区分，蓝色强调，极简克制
    """
    # 背景
    bg_base = "#1e1e1e"
    bg_surface = "#2c2c2c"
    bg_overlay = "#383838"
    bg_elevated = "#444444"
    bg_input = "#2a2a2a"

    # 文字
    text_primary = "#ffffff"
    text_secondary = "#b3b3b3"
    text_tertiary = "#7a7a7a"
    text_disabled = "#555555"
    text_hint = "#444444"

    # 主色调
    accent = "#0d99ff"
    accent_hover = "#2ba6ff"
    accent_pressed = "#0a7dd4"
    success = "#14ae5c"
    error = "#f24822"
    warning = "#ff8c00"
    info = "#a259ff"

    # 边框
    border = "#444444"
    border_subtle = "#363636"
    border_focus = "#0d99ff"

    # 尺寸
    radius = 6
    radius_sm = 4
    radius_lg = 8


# ══════════════════════════════════════════════════════════════════════════════
# 全局当前主题实例
# ══════════════════════════════════════════════════════════════════════════════
# 默认主题：CatppuccinMocha（与项目原始 UI 视觉一致）
# 如需切换到 Figma 深色风格，改成：current_theme: Theme = FigmaDark()
current_theme: Theme = CatppuccinMocha()

# 便捷别名：在各文件中 from theme import T 即可引用颜色
T = current_theme


def set_theme(theme: Theme):
    """切换当前主题（需要在 app.setStyleSheet 之前调用）"""
    global current_theme, T
    current_theme = theme
    T = theme


# ══════════════════════════════════════════════════════════════════════════════
# QSS 生成
# ══════════════════════════════════════════════════════════════════════════════
def generate_qss(theme: Theme = None) -> str:
    """
    根据主题配置生成完整的全局 QSS 样式表。
    调用方式：app.setStyleSheet(generate_qss())
    """
    t = theme or current_theme
    return f"""
/* ═══════════════════════════════════════════════════════════════════════
   皮皮贴图修改器 — 全局主题样式表
   由 theme.py generate_qss() 自动生成，请勿手动编辑
   ═══════════════════════════════════════════════════════════════════════ */

/* ── 全局基础 ── */
QWidget {{
    font-family: {t.font_family};
    font-size: {t.font_size}px;
    color: {t.text_primary};
}}

QMainWindow {{
    background-color: {t.bg_base};
}}

/* ── 按钮 ── */
QPushButton {{
    background-color: {t.bg_overlay};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius}px;
    padding: 7px 16px;
    font-size: {t.font_size}px;
    font-weight: 500;
    color: {t.text_primary};
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: {t.bg_elevated};
    border-color: {t.border};
}}
QPushButton:pressed {{
    background-color: {t.bg_surface};
    border-color: {t.border};
}}
QPushButton:disabled {{
    color: {t.text_disabled};
    background-color: {t.bg_surface};
    border-color: transparent;
}}

/* 主要按钮（通过 setProperty("class", "primary") 启用） */
QPushButton[class="primary"] {{
    background-color: {t.accent};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[class="primary"]:hover {{
    background-color: {t.accent_hover};
}}
QPushButton[class="primary"]:pressed {{
    background-color: {t.accent_pressed};
}}

/* 危险按钮 */
QPushButton[class="danger"] {{
    background-color: {t.error};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[class="danger"]:hover {{
    background-color: #ff6b5a;
}}

/* ── 输入框 ── */
QLineEdit {{
    background-color: {t.bg_input};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_sm}px;
    padding: 6px 10px;
    font-size: {t.font_size}px;
    color: {t.text_primary};
    selection-background-color: {t.accent};
    selection-color: #ffffff;
}}
QLineEdit:hover {{
    border-color: {t.border};
}}
QLineEdit:focus {{
    border-color: {t.border_focus};
    border-width: 2px;
    padding: 5px 9px;
}}

/* ── 下拉框 ── */
QComboBox {{
    background-color: {t.bg_input};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_sm}px;
    padding: 5px 10px;
    font-size: {t.font_size}px;
    color: {t.text_primary};
    min-height: 22px;
}}
QComboBox:hover {{
    border-color: {t.border};
}}
QComboBox:focus {{
    border-color: {t.border_focus};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {t.text_secondary};
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_overlay};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    selection-background-color: {t.bg_elevated};
    selection-color: {t.text_primary};
    color: {t.text_primary};
    padding: 4px;
    outline: none;
}}

/* ── 滑块 ── */
QSlider::groove:horizontal {{
    height: 3px;
    background: {t.border};
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    width: 12px;
    height: 12px;
    margin: -5px 0;
    background: #ffffff;
    border: 1px solid {t.border};
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    border-color: {t.accent};
}}
QSlider::sub-page:horizontal {{
    background: {t.accent};
    border-radius: 1px;
}}
QSlider::groove:vertical {{
    width: 3px;
    background: {t.border};
    border-radius: 1px;
}}
QSlider::handle:vertical {{
    width: 12px;
    height: 12px;
    margin: 0 -5px;
    background: #ffffff;
    border: 1px solid {t.border};
    border-radius: 6px;
}}
QSlider::handle:vertical:hover {{
    border-color: {t.accent};
}}
QSlider::sub-page:vertical {{
    background: {t.accent};
    border-radius: 1px;
}}

/* ── SpinBox ── */
QSpinBox, QDoubleSpinBox {{
    background-color: {t.bg_input};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: 3px 6px;
    padding-right: 20px;
    font-size: {t.font_size}px;
    color: {t.text_primary};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t.border_focus};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    height: 11px;
    border: none;
    border-left: 1px solid {t.border_subtle};
    border-top-right-radius: {t.radius_sm}px;
    background: {t.bg_overlay};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background: {t.bg_elevated};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    height: 11px;
    border: none;
    border-left: 1px solid {t.border_subtle};
    border-bottom-right-radius: {t.radius_sm}px;
    background: {t.bg_overlay};
}}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {t.bg_elevated};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {t.text_secondary};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {t.text_secondary};
}}

/* ── CheckBox ── */
QCheckBox {{
    spacing: 8px;
    font-size: {t.font_size}px;
    color: {t.text_primary};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: {t.radius_sm}px;
    border: 1.5px solid {t.border};
    background: transparent;
}}
QCheckBox::indicator:hover {{
    border-color: {t.text_secondary};
}}
QCheckBox::indicator:checked {{
    background: {t.accent};
    border-color: {t.accent};
}}
QCheckBox::indicator:checked:hover {{
    background: {t.accent_hover};
    border-color: {t.accent_hover};
}}

/* ── RadioButton ── */
QRadioButton {{
    spacing: 6px;
    font-size: {t.font_size}px;
    color: {t.text_secondary};
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 1px solid {t.border};
    background: {t.bg_input};
}}
QRadioButton::indicator:checked {{
    background: {t.accent};
    border-color: {t.accent};
}}

/* ── GroupBox ── */
QGroupBox {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_lg}px;
    margin-top: 16px;
    padding: 14px 12px 10px 12px;
    font-size: {t.font_size}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {t.text_secondary};
    font-size: {t.font_size_sm}px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

/* ── TabWidget / TabBar ── */
QTabWidget::pane {{
    border: none;
    background-color: {t.bg_base};
}}
QTabBar {{
    background-color: {t.bg_surface};
}}
QTabBar::tab {{
    background-color: transparent;
    border: none;
    border-right: 2px solid transparent;
    padding: 10px 6px;
    color: {t.text_tertiary};
    font-size: {t.font_size}px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background-color: {t.bg_overlay};
    color: {t.text_primary};
    border-right: 2px solid {t.accent};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background-color: {t.bg_overlay};
    color: {t.text_secondary};
}}

/* ── 滚动条 ── */
QScrollBar:vertical {{
    width: 6px;
    background: transparent;
    margin: 2px 0;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(255, 255, 255, 0.3);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    height: 6px;
    background: transparent;
    margin: 0 2px;
}}
QScrollBar::handle:horizontal {{
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: rgba(255, 255, 255, 0.3);
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ── QScrollArea ── */
QScrollArea {{
    border: none;
    background: transparent;
}}

/* ── 标签 ── */
QLabel {{
    background: transparent;
    border: none;
}}

/* ── 分割线 ── */
QFrame[frameShape="4"] {{
    color: {t.border_subtle};
    max-height: 1px;
}}
QFrame[frameShape="5"] {{
    color: {t.border_subtle};
    max-width: 1px;
}}

/* ── QSplitter ── */
QSplitter::handle {{
    background: {t.border_subtle};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── ToolTip ── */
QToolTip {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: none;
    border-radius: {t.radius_sm}px;
    padding: 5px 10px;
    font-size: {t.font_size_sm}px;
}}

/* ── QMenu ── */
QMenu {{
    background-color: {t.bg_overlay};
    border: 1px solid {t.border};
    border-radius: {t.radius}px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: {t.radius_sm}px;
    color: {t.text_primary};
}}
QMenu::item:selected {{
    background-color: {t.bg_elevated};
}}
QMenu::separator {{
    height: 1px;
    background: {t.border_subtle};
    margin: 4px 8px;
}}

/* ── QProgressBar ── */
QProgressBar {{
    background-color: {t.bg_surface};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    font-size: 0px;
}}
QProgressBar::chunk {{
    background-color: {t.accent};
    border-radius: 3px;
}}

/* ── QTextEdit / QPlainTextEdit ── */
QTextEdit, QPlainTextEdit {{
    background-color: {t.bg_input};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: 4px;
    color: {t.text_primary};
    font-size: {t.font_size}px;
    selection-background-color: {t.accent};
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {t.border_focus};
}}

/* ── QDialog ── */
QDialog {{
    background-color: {t.bg_base};
}}
"""
