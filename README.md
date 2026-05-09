# PiPiTextureEditor - 皮皮贴图修改器

一个功能强大的贴图处理工具，支持多种贴图格式的查看、编辑和转换，并与 UE4 编辑器深度联动。

## 功能特性

- 🖼️ **贴图修改**：亮度/对比度调整、一键黑白、遮罩生成、裁切/旋转、去除黑底、自定义底色、尺寸调整
- 🎬 **精灵图制作**：序列帧合并为 Sprite Sheet，支持视频导入
- 🧭 **法线绘制**：可视化法线贴图绘制，支持 DirectX/OpenGL 格式切换
- 🌱 **灰度图生成**：智能生长算法生成灰度贴图
- 🔍 **全能看图**：高质量图像预览，支持缩放/平移/通道查看
- 🔗 **UE4 联动**：与 UE4 编辑器双向通信，支持从 UE4 导入贴图、编辑后一键导回

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
- 打包架构采用拆包设计：exe 和 _internal 不变，更新只替换 app/ 目录

## 版本管理

版本号在 `app/version.py` 中定义，每次发布新版本时修改 `__version__` 值。

## 自动更新

项目通过腾讯工蜂（git.woa.com）Release 进行自动更新：
- 发布新版本时上传 zip 压缩包作为 Release Asset
- 使用英文文件名格式：`PPTextureEditor_vX.X.X.zip`
- 更新器配置在 `app/updater.py` 中
- 用户需连接公司网络才能检查和下载更新

## 项目结构

```
GUIDev/
├── launcher.py              # 启动器（异常兜底和文件名检查）
├── 皮皮贴图修改器.spec       # PyInstaller 打包配置
├── build.ps1               # 一键打包脚本
├── requirements.txt        # Python 依赖
├── app/                    # 业务代码目录
│   ├── Texture_tool_GUI_with_tabs.py  # 主界面（MainWindow）
│   ├── widgets.py                     # 通用 UI 控件
│   ├── dialogs.py                     # 对话框组件
│   ├── utils.py                       # 图像工具函数
│   ├── flowmap_tab.py                 # 法线绘制标签页
│   ├── growth_gray_tab.py            # 灰度图生成标签页
│   ├── sprite_sheet_tab.py           # 精灵图制作标签页
│   ├── image_viewer_tab.py           # 全能看图标签页
│   ├── growth_algorithms.py          # 生长算法实现
│   ├── ue4_sync.py                   # UE4 联动通信模块
│   ├── export_dir_mixin.py           # 导出路径记忆 Mixin
│   ├── updater.py                    # 自动更新模块（工蜂源）
│   ├── version.py                    # 版本管理
│   └── bug.svg                       # 关于按钮图标
├── TextureToolGUI.ico      # 应用图标
└── README.md               # 项目说明
```

## 技术支持

如遇启动问题，错误日志将保存在 `error_log.txt` 中，请将日志文件发送给开发者。