# -*- mode: python ; coding: utf-8 -*-
"""
皮皮贴图修改器 —— PyInstaller 打包配置
使用方法：
    pyinstaller 皮皮贴图修改器.spec
注意事项：
    1. 打包前请确保已安装 opencv-python：pip install opencv-python
    2. 打包后输出目录为 dist/PPTextureEditor/
    3. exe 文件名为「皮皮贴图修改器.exe」，请勿修改
"""

import os
import sys

block_cipher = None

# ── 项目根目录 ──
SPEC_DIR = os.path.dirname(os.path.abspath(SPECPATH))

a = Analysis(
    ['launcher.py'],
    pathex=[SPEC_DIR],
    binaries=[],
    datas=[
        # 打包资源文件
        ('TextureToolGUI.ico', '.'),
        ('bug.svg', '.'),
        ('api_check.json', '.'),
        ('api_check2.json', '.'),
        ('release_check.json', '.'),
        ('release_check2.json', '.'),
    ],
    hiddenimports=[
        # ── 核心依赖 ──
        'cv2',                      # opencv-python（视频导入功能）
        'numpy',
        'PIL',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # ── 项目模块 ──
        'Texture_tool_GUI_with_tabs',
        'growth_gray_tab',
        'growth_algorithms',
        'flowmap_tab',
        'sprite_sheet_tab',
        'image_viewer_tab',
        'updater',
        'version',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',          # 不需要 tkinter（launcher.py 中仅作兜底，ctypes 优先）
        'matplotlib',       # 不需要
        'scipy',            # 不需要
        'pandas',           # 不需要
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='皮皮贴图修改器',          # ← exe 文件名，请勿修改
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                   # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='TextureToolGUI.ico',       # 应用图标
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PPTextureEditor',          # ← 输出文件夹名，请勿修改
)
