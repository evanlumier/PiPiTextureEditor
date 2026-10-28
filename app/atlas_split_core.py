# -*- coding: utf-8 -*-
"""
atlas_split_core.py - 图集拆分核心逻辑（不依赖 Qt）

裁切算法 1:1 复刻 Sprite2imgV1.1.exe（内部脚本 ImageCutterV0.1.py）：
从 UE 导出的 Paper2D Sprite .T3D 文本中读取区域坐标，从原始图集上裁出该区域。
parse_t3d_file / compute_crop_box 与原工具逐行一致（含已知不足，保证裁切像素与原工具相同）：
  - 每个 T3D 只取第一个匹配（re.search）
  - 正则 "SourceUV=" 也能匹配到 "BakedSourceUV="，取的是文本中先出现的那个
  - 坐标用 int() 截断而非四舍五入
  - 解析日志/报错文字与原工具一致（板块取最后一行作为失败原因）

原工具的输出命名（<T3D名>_<T3D名>.png）已不再使用：板块默认导出名为 <T3D名>，可单独改名。

文件后半部分为新增功能（原工具没有）：读取 T3D 引用的贴图名、检查 T3D 与当前图集是否对得上、
导出命名校验、读取 Sprite 在 UE 中的位置、导入 UE4 时的同名后缀规则。

log 参数：接收一行日志文本的回调（对应原工具 self.log）。
"""
import os
import re
from typing import Callable, Iterable, Optional, Tuple

from PIL import Image

LogFn = Callable[[str], None]

# 与原工具完全相同的正则
_RE_TEXTURE_DIM = r"SourceTextureDimension=\(X=([\d.-]+),Y=([\d.-]+)\)"
_RE_SOURCE_UV = r"SourceUV=\(X=([\d.-]+),Y=([\d.-]+)\)"
_RE_BAKED_UV = r"BakedSourceUV=\(X=([\d.-]+),Y=([\d.-]+)\)"
_RE_SOURCE_DIM = r"SourceDimension=\(X=([\d.-]+),Y=([\d.-]+)\)"
_RE_BAKED_DIM = r"BakedSourceDimension=\(X=([\d.-]+),Y=([\d.-]+)\)"


def parse_t3d_file(file_path: str, log: LogFn) -> Optional[dict]:
    """解析T3D文件，提取UV坐标和尺寸信息（原工具 parse_t3d_file）"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        log("正在解析T3D文件...")

        texture_match = re.search(_RE_TEXTURE_DIM, content)
        if not texture_match:
            log("错误: 未找到SourceTextureDimension")
            return None
        texture_width = float(texture_match.group(1))
        texture_height = float(texture_match.group(2))
        log(f"纹理尺寸: {texture_width} x {texture_height}")

        uv_match = re.search(_RE_SOURCE_UV, content)
        if not uv_match:
            uv_match = re.search(_RE_BAKED_UV, content)
            if not uv_match:
                log("错误: 未找到SourceUV或BakedSourceUV")
                return None
        source_u = float(uv_match.group(1))
        source_v = float(uv_match.group(2))
        log(f"UV坐标: ({source_u}, {source_v})")

        dim_match = re.search(_RE_SOURCE_DIM, content)
        if not dim_match:
            dim_match = re.search(_RE_BAKED_DIM, content)
            if not dim_match:
                log("错误: 未找到SourceDimension或BakedSourceDimension")
                return None
        source_width = float(dim_match.group(1))
        source_height = float(dim_match.group(2))
        log(f"源尺寸: {source_width} x {source_height}")

        sprite_name = os.path.splitext(os.path.basename(file_path))[0]
        sprite_info = {
            "name": sprite_name,
            "texture_width": texture_width,
            "texture_height": texture_height,
            "source_u": source_u,
            "source_v": source_v,
            "source_width": source_width,
            "source_height": source_height,
        }
        log("T3D文件解析成功！")
        return sprite_info
    except Exception as e:
        log(f"解析T3D文件时出错: {str(e)}")
        return None


def compute_crop_box(img_width: int, img_height: int, sprite_info: dict, log: Optional[LogFn] = None):
    """按图片实际尺寸缩放坐标并钳制到图片范围内，返回 (left, top, right, bottom)。

    与原工具 extract_sprite_from_image 中的计算逐行一致；texture_width/height 为 0 时
    会抛 ZeroDivisionError，由调用方捕获。
    """
    scale_x = img_width / sprite_info["texture_width"]
    scale_y = img_height / sprite_info["texture_height"]

    left = int(sprite_info["source_u"] * scale_x)
    top = int(sprite_info["source_v"] * scale_y)
    right = int((sprite_info["source_u"] + sprite_info["source_width"]) * scale_x)
    bottom = int((sprite_info["source_v"] + sprite_info["source_height"]) * scale_y)

    if log:
        log(f"截取区域: 位置({left}, {top}) 尺寸({right - left}x{bottom - top})")

    left = max(0, min(left, img_width - 1))
    top = max(0, min(top, img_height - 1))
    right = max(left + 1, min(right, img_width))
    bottom = max(top + 1, min(bottom, img_height))
    return left, top, right, bottom


def crop_sprite(image: Image.Image, sprite_info: dict) -> Optional[Image.Image]:
    """按 compute_crop_box 的区域在内存中裁出 Sprite（与原工具裁切像素一致）；区域无效返回 None。

    纹理尺寸为 0 时抛 ZeroDivisionError，由调用方捕获。
    """
    w, h = image.size
    left, top, right, bottom = compute_crop_box(w, h, sprite_info)
    if right > left and bottom > top:
        return image.crop((left, top, right, bottom))
    return None


# ══════════════════════════════════════════════════════════════════════
# 以下为新增功能（原工具没有）：检查 T3D 与当前图集是否对得上
# ══════════════════════════════════════════════════════════════════════

# T3D 中引用贴图的写法（真实样本）：
#   SourceTexture="/Game/.../Textures/T_Story_Atlas_0.T_Story_Atlas_0"
#   BakedSourceTexture="/Script/Engine.Texture2D'/Game/.../T_Story_Atlas_0.T_Story_Atlas_0'"
# 另兼容旧式写法 SourceTexture=Texture2D'/Game/.../T_Atlas.T_Atlas'
# 先取 SourceTexture，没有再取 BakedSourceTexture；前者的后顾断言保证不会匹配到 BakedSourceTexture=，
# 紧跟等号也不会匹配到 SourceTextureDimension=
_RE_SOURCE_TEXTURE = r'(?<![A-Za-z])SourceTexture=("[^"\r\n]*"|\S+)'
_RE_BAKED_SOURCE_TEXTURE = r'BakedSourceTexture=("[^"\r\n]*"|\S+)'

ASPECT_TOLERANCE = 0.01  # 宽高比允许 1% 误差（图集可以被整体缩放）


def _texture_asset_name(ref: str) -> Optional[str]:
    """对象引用 → 资产名：'/Game/A/T_Atlas.T_Atlas' 或 "Texture2D'/Game/A/T_Atlas.T_Atlas'" → 'T_Atlas'"""
    ref = ref.strip().strip('"')
    if "'" in ref:
        parts = ref.split("'")
        if len(parts) >= 2:
            ref = parts[1]
    ref = ref.rsplit("/", 1)[-1]
    name = ref.rsplit(".", 1)[-1]
    if not name or name == "None":
        return None
    return name


def read_source_texture_name(file_path: str) -> Optional[str]:
    """读取 T3D 中引用的贴图资产名，没有则返回 None"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None
    for pattern in (_RE_SOURCE_TEXTURE, _RE_BAKED_SOURCE_TEXTURE):
        m = re.search(pattern, content)
        if m:
            name = _texture_asset_name(m.group(1))
            if name:
                return name
    return None


def check_atlas_match(sprite_info: dict, texture_name: Optional[str], atlas_path: str,
                      img_width: int, img_height: int) -> str:
    """检查 T3D 是否属于当前图集，对得上返回空字符串，否则返回原因。

    两条判据（任一不符即视为对不上）：
      1. T3D 引用的贴图名 ≠ 图集文件名（不区分大小写；T3D 里没有贴图名时跳过此条）
      2. T3D 记录的纹理宽高比 ≠ 图集宽高比（允许整体缩放，所以只比宽高比不比尺寸）
    """
    reasons = []
    atlas_stem = os.path.splitext(os.path.basename(atlas_path))[0]
    if texture_name and texture_name.lower() != atlas_stem.lower():
        reasons.append(f"引用的贴图是 {texture_name}，当前图集是 {atlas_stem}")

    tw, th = sprite_info["texture_width"], sprite_info["texture_height"]
    if tw > 0 and th > 0 and img_width > 0 and img_height > 0:
        t3d_ratio = tw / th
        img_ratio = img_width / img_height
        if abs(t3d_ratio - img_ratio) / img_ratio > ASPECT_TOLERANCE:
            reasons.append(f"纹理比例 {tw:g}x{th:g} 与图集 {img_width}x{img_height} 不一致")
    return "；".join(reasons)


# ══════════════════════════════════════════════════════════════════════
# 导出命名 & UE4 导入（新增）
# ══════════════════════════════════════════════════════════════════════

# 资产名规则：UE4 资源名只认 ASCII；只允许字母/数字/下划线，不能以数字开头
_RE_VALID_ASSET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# 第一行 Begin Object 的 ExportPath 即 Sprite 自身：
#   ExportPath="/Script/Paper2D.PaperSprite'/Game/.../Frames/T_X.T_X'"
_RE_EXPORT_PATH = r"ExportPath=\"[^'\"]*'(/[^'\"]+)'\""

# 导入 UE4 时，目标目录里已有同名 Sprite（PaperSprite 资产）会冲突，自动给贴图名加这个后缀
UE4_CLASH_SUFFIX = "_Tex"


def validate_asset_name(name: str) -> str:
    """检查导出名能否同时作为文件名和 UE4 资产名；合法返回空字符串，否则返回原因"""
    if not name:
        return "名字为空"
    if name[0].isdigit():
        return "不能以数字开头"
    if not _RE_VALID_ASSET_NAME.fullmatch(name):
        bad = "".join(sorted({c for c in name if not (c.isascii() and (c.isalnum() or c == "_"))}))
        return f"含不允许的字符「{bad}」（只能用英文字母、数字、下划线）"
    return ""


def read_sprite_asset_ref(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """读取 Sprite 自身在 UE 中的 (所在目录, 资产名)，如 ('/Game/UI/Frames', 'T_X')；读不到返回 (None, None)"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None, None
    m = re.search(_RE_EXPORT_PATH, content)
    if not m:
        return None, None
    obj_path = m.group(1).split(":", 1)[0]          # 去掉子对象 ":BodySetup_3"
    package, _, asset = obj_path.partition(".")
    if "/" not in package:
        return None, None
    directory, _, pkg_name = package.rpartition("/")
    return (directory or None), (asset or pkg_name or None)


def normalize_ue_dir(path: str) -> str:
    """UE 目录比较用：统一斜杠、去掉末尾斜杠、不区分大小写"""
    return path.replace("\\", "/").rstrip("/").lower()


def ue4_import_name(export_name: str, target_dir: str, sprites: Iterable[Tuple[Optional[str], Optional[str]]]) -> str:
    """导入 UE4 时的最终资产名：目标目录里有同名 Sprite 时加 UE4_CLASH_SUFFIX，否则原样返回。

    sprites：列表中所有 T3D 对应 Sprite 的 (目录, 资产名)
    """
    target = normalize_ue_dir(target_dir)
    key = export_name.lower()
    for d, n in sprites:
        if d and n and normalize_ue_dir(d) == target and n.lower() == key:
            return export_name + UE4_CLASH_SUFFIX
    return export_name
