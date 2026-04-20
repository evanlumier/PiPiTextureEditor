# PiPiTextureEditor - 皮皮贴图修改器

一个功能强大的贴图处理工具，支持多种贴图格式的查看、编辑和转换。

## 功能特性

- 🖼️ 多标签页界面：支持同时处理多个贴图文件
- 🔄 流场图处理：专业的流场贴图分析和可视化
- 🌱 生长算法：智能贴图生成和扩展功能
- 🎮 Sprite Sheet 支持：游戏精灵表编辑和管理
- 📊 图像查看器：高质量的图像预览和缩放
- 🔄 自动更新：GitHub Release 在线更新支持

## 安装依赖

```bash
pip install -r requirements.txt
```

## 开发环境设置

1. 克隆项目仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 运行主程序：`python launcher.py`

## 打包发布

### 使用 PowerShell 一键打包
```powershell
.\build.ps1
```

### 手动打包步骤
1. 使用 PyInstaller：`pyinstaller 皮皮贴图修改器.spec`
2. 复制 app/ 目录到 `dist/PPTextureEditor/app/`
3. 验证输出目录结构：
   - `皮皮贴图修改器.exe`
   - `_internal/` 目录
   - `app/` 目录（业务代码）

### 打包注意事项
- 确保已安装 opencv-python：`pip install opencv-python`
- exe 文件名必须保持为「皮皮贴图修改器.exe」
- 打包架构采用拆包设计，避免 iOA 白名单失效问题

## iOA 白名单处理指南

### 白名单配置（一劳永逸方案）
本项目的拆包架构设计实现了 iOA 白名单的永久有效：

**首次部署时，请管理员将以下内容加入白名单：**
- ✅ `皮皮贴图修改器.exe`（完整权限）
- ✅ `_internal/` 目录（读写执行权限）
- ✅ `app/` 目录（读写执行权限）

**白名单优势：**
- 🎯 **一劳永逸**：主 exe 文件哈希值固定，白名单不会失效
- 🔄 **无缝更新**：未来版本更新只需替换 `app/` 目录内容
- ⚡ **无需重新申请**：更新后无需重新加白名单

### 白名单材料准备
1. 运行打包脚本：`\build.ps1`
2. 将生成的 `dist/PPTextureEditor/` 目录压缩为 ZIP 文件
3. 将 ZIP 文件提供给管理员进行白名单配置

## 版本管理

版本号在 `app/version.py` 中定义，每次发布新版本时修改 `__version__` 值。

## 自动更新

项目支持 GitHub Release 自动更新：
- 发布新版本时上传 zip 压缩包作为 Release Asset
- 使用英文文件名格式：`PPTextureEditor_vX.X.X.zip`
- 更新器配置在 `app/updater.py` 中

## 项目结构

```
GUIDev/
├── launcher.py          # 启动器（异常兜底和文件名检查）
├── 皮皮贴图修改器.spec   # PyInstaller 打包配置
├── build.ps1           # 一键打包脚本
├── requirements.txt    # Python 依赖
├── app/               # 业务代码目录
│   ├── Texture_tool_GUI_with_tabs.py  # 主界面
│   ├── flowmap_tab.py                 # 流场图标签页
│   ├── growth_gray_tab.py            # 生长算法标签页  
│   ├── sprite_sheet_tab.py           # Sprite Sheet 标签页
│   ├── image_viewer_tab.py           # 图像查看器
│   ├── growth_algorithms.py          # 生长算法实现
│   ├── updater.py                    # 自动更新模块
│   ├── version.py                    # 版本管理
│   └── bug.svg                       # 错误图标
├── TextureToolGUI.ico  # 应用图标
└── README.md           # 项目说明
```

## 技术支持

如遇启动问题，错误日志将保存在 `error_log.txt` 中，请将日志文件发送给开发者。