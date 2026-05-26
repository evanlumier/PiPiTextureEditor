# -*- mode: python ; coding: utf-8 -*-
"""
皮皮贴图修改器 —— PyInstaller 打包配置（拆包架构）
使用方法：
    pyinstaller 皮皮贴图修改器.spec
    然后运行 build.ps1 复制 app/ 目录到 dist/

★ 架构说明：
    打包后目录结构为 exe + _internal/ + app/
    - exe 和 _internal/ 由 PyInstaller 生成，几乎不变（只要不升级依赖库）
    - app/ 目录包含业务代码，每次发版只更新这个目录
    - 这样 exe 的哈希不变，iOA 白名单不会失效

注意事项：
    1. 打包前请确保已安装 opencv-python：pip install opencv-python
    2. 打包后输出目录为 dist/PPTextureEditor/
    3. exe 文件名为「皮皮贴图修改器.exe」，请勿修改
    4. 打包完成后需要运行 build.ps1 或手动复制 app/ 到 dist/PPTextureEditor/app/
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ── 确保 Pillow 完整打包（包含所有 C 扩展和子模块）──
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')

# ── 确保 OpenCV 完整打包（包含 FFmpeg DLL，解码 MOV/MP4 等视频格式必需）──
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')

# ── 项目根目录 ──
SPEC_DIR = os.path.dirname(os.path.abspath(SPECPATH))

# ── 收集 app/ 目录下所有 .py 文件作为额外的分析入口 ──
# 这样 PyInstaller 会自动分析它们的依赖（标准库 + 第三方库），
# 确保打包后 _internal/ 中包含所有需要的模块。
import glob
app_dir = os.path.join(SPEC_DIR, 'app')
app_scripts = glob.glob(os.path.join(app_dir, '*.py'))

a = Analysis(
    ['launcher.py'] + app_scripts,
    pathex=[SPEC_DIR, app_dir],
    binaries=pil_binaries + cv2_binaries,
    datas=[
        # 只打包图标文件到 exe 同级目录（供 launcher 和 Windows 任务栏使用）
        ('TextureToolGUI.ico', '.'),
        # 业务资源文件（bug.svg, json 等）已移至 app/ 目录，通过复制方式部署
    ] + pil_datas + cv2_datas,
    hiddenimports=[
        # ── 第三方依赖（必须打包进 _internal/）──
        'cv2',                      # opencv-python（视频导入功能）
        'numpy',
        'PIL',                      # Pillow 图像处理
        'PIL.Image',
        'PIL.ImageEnhance',
        'PIL.ImageOps',
        'PIL.ImageDraw',
        'PIL.ImageFilter',
        'PIL.ImageFont',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtSvg',           # SVG 渲染支持（image_viewer_tab 使用）
        'fitz',                     # PyMuPDF（PDF 查看支持）
        'pymupdf',                  # PyMuPDF 底层模块
        'psd_tools',                # psd-tools（PSD/PSB 查看支持）
        # ── 标准库模块（app/ 中使用但 launcher.py 未直接引用）──
        'uuid',
        'threading',
        'json',
        'hashlib',
        'ssl',
        'dataclasses',
        # ── 项目模块不再打包（它们在 app/ 目录下作为源码部署）──
        # 'Texture_tool_GUI_with_tabs', 'updater', 'version', ...
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
