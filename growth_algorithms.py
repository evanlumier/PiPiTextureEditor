# -*- coding: utf-8 -*-
"""
growth_algorithms.py
序列帧 → 生长灰度图 核心算法模块

不依赖 scipy，仅使用 PIL + numpy。
所有中间数据均为 numpy float32，H×W。
"""

import os
import re
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ─────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────

def natural_sort_key(path: str) -> list:
    """
    自然排序 key：将文件名中的数字部分作为整数比较，
    支持 frame1, frame2 ... frame10, frame11 顺序。
    风格与 sprite_sheet_tab._natural_sort_key 保持一致。
    """
    name = os.path.basename(path).lower()
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p for p in parts]


# ─────────────────────────────────────────────────────────────────────
# 单帧 → 占位图（presence map）
# ─────────────────────────────────────────────────────────────────────

def rgba_to_presence_map(
    img: Image.Image,
    source_mode: str = "auto",
    blur_radius: float = 0.0,
    _force_mode: str = "",
) -> Tuple[np.ndarray, str]:
    """
    将单帧 PIL RGBA 图像转换为 H×W float32 占位图（0~1）。

    source_mode:
        "auto"       — 先做单帧判定（alpha 不全为 255 时用 alpha，否则用 luminance）；
                       若调用方已通过跨帧分析确定了更优模式，可用 _force_mode 覆盖。
        "alpha"      — 强制使用 alpha 通道 / 255
        "luminance"  — 强制使用 0.299R + 0.587G + 0.114B，归一化到 0~1

    _force_mode:
        仅在 source_mode=="auto" 时生效，由 generate_growth_gray_from_sequence
        的跨帧分析结果传入，取值 "alpha" 或 "luminance"，覆盖单帧 auto 判定。

    blur_radius > 0 时，先对图像做 GaussianBlur（PIL 实现，无 scipy 依赖），
    再提取占位图，可平滑噪点边缘。

    返回：(presence_map: H×W float32, actual_mode: str)
        actual_mode 为实际使用的模式："alpha" 或 "luminance"
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # 可选模糊（在提取通道前做，避免边缘锯齿）
    if blur_radius > 0.0:
        # PIL GaussianBlur radius 参数约等于 sigma
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.array(img, dtype=np.float32)   # H×W×4，值域 0~255

    if source_mode == "alpha":
        presence = arr[:, :, 3] / 255.0
        actual_mode = "alpha"

    elif source_mode == "luminance":
        presence = (
            arr[:, :, 0] * 0.299
            + arr[:, :, 1] * 0.587
            + arr[:, :, 2] * 0.114
        ) / 255.0
        actual_mode = "luminance"

    else:  # "auto"
        # 若跨帧分析已给出更优判定，直接使用
        if _force_mode in ("alpha", "luminance"):
            if _force_mode == "alpha":
                presence = arr[:, :, 3] / 255.0
            else:
                presence = (
                    arr[:, :, 0] * 0.299
                    + arr[:, :, 1] * 0.587
                    + arr[:, :, 2] * 0.114
                ) / 255.0
            actual_mode = _force_mode
        else:
            # 单帧 fallback：alpha 不全为 255 时用 alpha，否则用 luminance
            alpha_ch = arr[:, :, 3]
            if alpha_ch.max() < 255.0:
                presence = alpha_ch / 255.0
                actual_mode = "alpha"
            else:
                presence = (
                    arr[:, :, 0] * 0.299
                    + arr[:, :, 1] * 0.587
                    + arr[:, :, 2] * 0.114
                ) / 255.0
                actual_mode = "luminance"

    return presence.astype(np.float32), actual_mode


# ─────────────────────────────────────────────────────────────────────
# 单调包络
# ─────────────────────────────────────────────────────────────────────

def build_monotonic_envelope(
    frame_maps: List[np.ndarray],
) -> List[np.ndarray]:
    """
    对一组占位图序列做单调递增包络（cumulative max）。

    规则：
        E[0] = M[0]
        E[i] = maximum(E[i-1], M[i])

    返回与输入等长的包络列表，每帧均为 H×W float32。
    """
    if not frame_maps:
        return []

    envelopes: List[np.ndarray] = []
    running_max = frame_maps[0].copy()
    envelopes.append(running_max.copy())

    for m in frame_maps[1:]:
        np.maximum(running_max, m, out=running_max)
        envelopes.append(running_max.copy())

    return envelopes


# ─────────────────────────────────────────────────────────────────────
# 首次过阈时间图
# ─────────────────────────────────────────────────────────────────────

def compute_first_hit_time(
    envelope_maps: List[np.ndarray],
    threshold: float = 0.2,
) -> np.ndarray:
    """
    对每个像素，在单调包络序列中找第一次 >= threshold 的帧索引，
    将其归一化为 [0, 1] 作为灰度值。

    命中第 i 帧（0-based）时：gray = i / (N - 1)
    未命中区域：gray = 1.0

    实现：将所有帧堆叠为 (N, H, W) 数组，用向量化操作一次性完成，
    避免 H×W×N 的 Python 三重循环。

    返回 H×W float32。
    """
    N = len(envelope_maps)
    if N == 0:
        raise ValueError("envelope_maps 不能为空")

    H, W = envelope_maps[0].shape

    # 堆叠为 (N, H, W)
    stack = np.stack(envelope_maps, axis=0)   # float32, (N, H, W)

    # 布尔掩码：哪些帧 >= threshold
    hit_mask = stack >= threshold              # (N, H, W) bool

    # 是否有任何帧命中
    any_hit = hit_mask.any(axis=0)            # (H, W) bool

    # argmax 在 bool 数组上返回第一个 True 的索引
    # 对于全 False 的像素，argmax 返回 0（但 any_hit 会过滤掉）
    first_hit_idx = hit_mask.argmax(axis=0).astype(np.float32)  # (H, W)

    if N > 1:
        gray = first_hit_idx / float(N - 1)
    else:
        gray = np.zeros((H, W), dtype=np.float32)

    # 未命中区域设为 1.0
    gray[~any_hit] = 1.0

    return gray.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# 最终 Mask
# ─────────────────────────────────────────────────────────────────────

def final_mask_from_sequence(
    envelope_last: np.ndarray,
    mask_threshold: float = 0.05,
) -> np.ndarray:
    """
    从最后一帧单调包络生成最终 mask。
    mask = (envelope_last >= mask_threshold)，值为 0.0 或 1.0，float32。
    """
    return (envelope_last >= mask_threshold).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────

def generate_growth_gray_from_sequence(
    frames: List[Image.Image],
    source_mode: str = "auto",
    presence_blur: float = 0.0,
    hit_threshold: float = 0.2,
    mask_threshold: float = 0.05,
    invert: bool = False,
) -> dict:
    """
    将一组序列帧自动转换为生长灰度图。

    参数：
        frames          — PIL RGBA 图像列表（已按时间顺序排好）
        source_mode     — "auto" / "alpha" / "luminance"
        presence_blur   — 单帧模糊半径（0 = 不模糊）
        hit_threshold   — 首次过阈判定阈值（0~1）
        mask_threshold  — 最终 mask 阈值（0~1）
        invert          — 是否反转灰度（gray = 1 - gray）

    返回 dict：
        "gray_map"        : H×W float32，最终灰度图（mask 外为 0）
        "mask_map"        : H×W float32，0/1 mask
        "first_presence"  : H×W float32，第一帧占位图（用于预览）
        "last_envelope"   : H×W float32，最后一帧单调包络（用于预览）
        "frame_count"     : int，帧数
    """
    if not frames:
        raise ValueError("frames 列表不能为空")

    N = len(frames)

    # ── 跨帧 auto 判定（在生成占位图前先分析整组帧）──────────────────
    # 目标：如果 RGB 亮度在序列中接近恒定，但 Alpha 明显变化，
    #       则强制判定为 alpha 模式，不受单帧 alpha.max()==255 的干扰。
    #
    # 实现：对每帧采样（最多取 16 帧均匀采样，避免大序列过慢）
    #   lum_std_mean  = 各帧亮度图均值的标准差（跨帧变化量）
    #   alpha_std_mean = 各帧 alpha 图均值的标准差（跨帧变化量）
    #   判定条件：
    #     lum_std_mean  < LUM_STABLE_THRESH   （亮度几乎不变）
    #     alpha_std_mean > ALPHA_VARY_THRESH   （alpha 明显变化）
    #   → 强制 alpha
    LUM_STABLE_THRESH  = 0.04   # 亮度跨帧均值标准差阈值（0~1）
    ALPHA_VARY_THRESH  = 0.04   # alpha 跨帧均值标准差阈值（0~1）

    _force_mode = ""            # 跨帧分析给出的强制模式（仅 auto 时使用）
    _cross_frame_triggered = False  # 是否触发了跨帧强制判定

    if source_mode == "auto" and N >= 2:
        # 均匀采样最多 16 帧
        sample_step = max(1, N // 16)
        sample_indices = list(range(0, N, sample_step))[:16]

        lum_means = []
        alpha_means = []
        for idx in sample_indices:
            f = frames[idx]
            if f.mode != "RGBA":
                f = f.convert("RGBA")
            a = np.array(f, dtype=np.float32)
            lum = (a[:, :, 0] * 0.299 + a[:, :, 1] * 0.587 + a[:, :, 2] * 0.114) / 255.0
            alp = a[:, :, 3] / 255.0
            lum_means.append(float(lum.mean()))
            alpha_means.append(float(alp.mean()))

        lum_std   = float(np.std(lum_means))
        alpha_std = float(np.std(alpha_means))

        if lum_std < LUM_STABLE_THRESH and alpha_std > ALPHA_VARY_THRESH:
            _force_mode = "alpha"
            _cross_frame_triggered = True

    # ── 每帧 → 占位图 ────────────────────────────────────────────────
    raw_results = [
        rgba_to_presence_map(f, source_mode, presence_blur, _force_mode)
        for f in frames
    ]
    presence_maps: List[np.ndarray] = [r[0] for r in raw_results]
    actual_modes: List[str] = [r[1] for r in raw_results]

    # 统计实际使用的模式（取多数帧的结果）
    alpha_count = actual_modes.count("alpha")
    lum_count = actual_modes.count("luminance")
    actual_mode_final = "alpha" if alpha_count >= lum_count else "luminance"

    # 计算 alpha 与亮度的差异（用于提示用户检查识别方式）
    # 取第一帧做差异检测：alpha 归一化值 vs 亮度归一化值，均值差
    arr0 = np.array(frames[0].convert("RGBA"), dtype=np.float32)
    alpha_norm = arr0[:, :, 3] / 255.0
    lum_norm = (
        arr0[:, :, 0] * 0.299
        + arr0[:, :, 1] * 0.587
        + arr0[:, :, 2] * 0.114
    ) / 255.0
    alpha_lum_diff = float(np.abs(alpha_norm - lum_norm).mean())

    # 2. 单调包络
    envelope_maps = build_monotonic_envelope(presence_maps)

    # 3. 首次过阈时间图
    gray = compute_first_hit_time(envelope_maps, hit_threshold)

    # 4. 最终 mask（来自最后一帧包络）
    mask = final_mask_from_sequence(envelope_maps[-1], mask_threshold)

    # 5. mask 外清零
    gray = gray * mask

    # 6. 可选反转
    if invert:
        # 只在 mask 内反转，mask 外保持 0
        gray = np.where(mask > 0, 1.0 - gray, 0.0).astype(np.float32)

    return {
        "gray_map":               gray,
        "mask_map":               mask,
        "first_presence":         presence_maps[0],
        "last_envelope":          envelope_maps[-1],
        "frame_count":            N,
        "actual_mode":            actual_mode_final,    # 实际使用的识别方式
        "alpha_lum_diff":         alpha_lum_diff,        # alpha 与亮度的均值差（0~1）
        "cross_frame_triggered":  _cross_frame_triggered,  # 是否由跨帧分析强制切换
    }


# ─────────────────────────────────────────────────────────────────────
# 单图手绘路径 → seed_map
# ─────────────────────────────────────────────────────────────────────

def rasterize_stroke_to_seed(
    seed_map: np.ndarray,
    stroke_points: list,
    brush_radius: int = 8,
    brush_hardness: float = 1.0,
    time_start: float = 0.0,
    time_end: float = 1.0,
) -> np.ndarray:
    """
    将一条笔触（像素坐标列表）按时间进度写入 seed_map。

    参数：
        seed_map       — H×W float32，-1 = 未赋值，原地修改并返回
        stroke_points  — [(x, y), ...] 像素坐标列表（画布坐标）
        brush_radius   — 笔刷半径（像素）
        brush_hardness — 0~1，1 = 硬边（圆形覆盖），<1 = 边缘衰减
                         衰减公式：weight = max(0, 1 - dist/radius)^(1/hardness)
                         hardness=1 时退化为硬边（dist<=radius 全覆盖）
        time_start     — 笔触起点时间值（0~1）
        time_end       — 笔触终点时间值（0~1）

    返回修改后的 seed_map（原地修改，同时也返回引用）。
    """
    if not stroke_points:
        return seed_map

    H, W = seed_map.shape
    N = len(stroke_points)
    r = max(1, brush_radius)

    for i, (cx, cy) in enumerate(stroke_points):
        # 当前点的时间值（线性插值）
        t = time_start + (time_end - time_start) * (i / max(N - 1, 1))

        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)

        if brush_hardness >= 1.0:
            # 硬边：圆形内全覆盖
            inside = dist <= r
            seed_map[y0:y1, x0:x1][inside] = t
        else:
            # 软边：按距离衰减，只覆盖权重 > 0 的区域
            # 衰减：w = (1 - dist/r)^(1/hardness)，dist > r 时 w=0
            norm_dist = np.clip(dist / r, 0.0, 1.0)
            weight = (1.0 - norm_dist) ** (1.0 / max(brush_hardness, 0.01))
            # 只在权重 > 0 且新值比旧值"更有意义"时写入
            # 策略：权重 > 0 的区域直接覆盖（第一版简单方案）
            mask_w = weight > 0.0
            seed_map[y0:y1, x0:x1][mask_w] = t

    return seed_map


# ─────────────────────────────────────────────────────────────────────
# 收集 seed 点
# ─────────────────────────────────────────────────────────────────────

def collect_seed_points(
    seed_map: np.ndarray,
    mask_map: np.ndarray,
) -> tuple:
    """
    从 seed_map 中收集所有有效 seed 点（seed >= 0 且在 mask 内）。

    返回：
        coords  — (N_seed, 2) int32，每行 [y, x]
        values  — (N_seed,) float32，对应时间值 0~1
    """
    valid = (seed_map >= 0.0)
    if mask_map is not None:
        valid = valid & (mask_map > 0)

    ys, xs = np.where(valid)
    coords = np.stack([ys, xs], axis=1).astype(np.int32)   # (N, 2)
    values = seed_map[ys, xs].astype(np.float32)           # (N,)
    return coords, values


# ─────────────────────────────────────────────────────────────────────
# 距离加权传播
# ─────────────────────────────────────────────────────────────────────

def propagate_seed_to_gray(
    seed_map: np.ndarray,
    mask_map: np.ndarray,
    radius: int = 64,
    power: float = 2.0,
    fallback_nearest: bool = True,
    max_seeds: int = 4096,
) -> np.ndarray:
    """
    将 seed_map 中的种子点通过距离加权传播，填满 mask 内所有像素。

    算法（无 scipy，纯 numpy）：
        1. 收集所有 seed 点坐标和时间值
        2. 若 seed 点超过 max_seeds，随机采样（避免内存爆炸）
        3. 对 mask 内每个像素，计算到所有 seed 点的距离
        4. 取 radius 范围内的 seed 点做距离加权平均：
               wi = 1 / (dist^power + eps)
               gray = sum(wi * ti) / sum(wi)
        5. radius 内无 seed 时，若 fallback_nearest=True 则用最近 seed 值
        6. mask 外区域设为 0.0

    参数：
        seed_map         — H×W float32，-1 = 未赋值
        mask_map         — H×W float32，0/1 mask（None 则全图传播）
        radius           — 传播半径（像素）
        power            — 距离衰减指数（越大越局部）
        fallback_nearest — radius 内无 seed 时是否 fallback 到最近 seed
        max_seeds        — 最大 seed 点数（超出则随机采样，防止内存溢出）

    返回 H×W float32 gray_map。
    """
    H, W = seed_map.shape

    # 构建输出
    gray = np.zeros((H, W), dtype=np.float32)

    # 确定 mask
    if mask_map is not None:
        mask = mask_map > 0
    else:
        mask = np.ones((H, W), dtype=bool)

    # 收集 seed 点
    coords, values = collect_seed_points(seed_map, mask_map)
    N_seed = len(coords)

    if N_seed == 0:
        # 没有 seed，返回全零
        return gray

    # 若 seed 点过多，随机采样
    if N_seed > max_seeds:
        idx = np.random.choice(N_seed, max_seeds, replace=False)
        coords = coords[idx]
        values = values[idx]
        N_seed = max_seeds

    # 获取 mask 内所有像素坐标
    mask_ys, mask_xs = np.where(mask)
    N_pixels = len(mask_ys)

    if N_pixels == 0:
        return gray

    # ── 核心：分块广播，避免一次性 (N_pixels × N_seed) 内存爆炸 ──
    # 每块处理 chunk_size 个像素
    # 内存估算：chunk_size × N_seed × 4 bytes
    # 目标：每块不超过 ~128MB → chunk_size = 128MB / (N_seed * 4)
    target_bytes = 128 * 1024 * 1024  # 128 MB
    chunk_size = max(1, int(target_bytes / (N_seed * 4)))
    chunk_size = min(chunk_size, N_pixels)

    # seed 坐标转为 float32 方便广播
    seed_y = coords[:, 0].astype(np.float32)   # (N_seed,)
    seed_x = coords[:, 1].astype(np.float32)   # (N_seed,)

    eps = 1e-6
    radius_f = float(radius)

    for start in range(0, N_pixels, chunk_size):
        end = min(start + chunk_size, N_pixels)

        py = mask_ys[start:end].astype(np.float32)   # (chunk,)
        px = mask_xs[start:end].astype(np.float32)   # (chunk,)

        # 广播计算距离：(chunk, N_seed)
        dy = py[:, None] - seed_y[None, :]   # (chunk, N_seed)
        dx = px[:, None] - seed_x[None, :]   # (chunk, N_seed)
        dist = np.sqrt(dy * dy + dx * dx)    # (chunk, N_seed)

        # radius 内的 seed
        in_radius = dist <= radius_f         # (chunk, N_seed) bool

        # 距离加权
        w = 1.0 / (dist ** power + eps)      # (chunk, N_seed)
        w_masked = w * in_radius             # radius 外权重清零

        w_sum = w_masked.sum(axis=1)         # (chunk,)
        has_seed = w_sum > eps               # (chunk,) bool

        # 加权平均
        gray_chunk = np.zeros(end - start, dtype=np.float32)
        if has_seed.any():
            gray_chunk[has_seed] = (
                (w_masked[has_seed] * values[None, :]).sum(axis=1)
                / w_sum[has_seed]
            )

        # fallback：radius 内无 seed → 用最近 seed
        if fallback_nearest and (~has_seed).any():
            nearest_idx = dist[~has_seed].argmin(axis=1)
            gray_chunk[~has_seed] = values[nearest_idx]

        gray[mask_ys[start:end], mask_xs[start:end]] = gray_chunk

    return gray.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# 平滑
# ─────────────────────────────────────────────────────────────────────

def smooth_gray_map(
    gray_map: np.ndarray,
    mask_map: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    对 gray_map 在 mask 内做简单均值平滑（box blur），不引入 scipy。

    实现：用 PIL ImageFilter.BoxBlur 做模糊，再用 mask 还原边界。
    mask 外区域保持 0。

    参数：
        gray_map    — H×W float32
        mask_map    — H×W float32，0/1 mask（None 则全图平滑）
        iterations  — 平滑迭代次数
        kernel_size — box blur 半径（像素）

    返回平滑后的 H×W float32。
    """
    if iterations <= 0:
        return gray_map.copy()

    result = gray_map.copy()

    if mask_map is not None:
        mask = (mask_map > 0).astype(np.float32)
    else:
        mask = np.ones_like(result)

    for _ in range(iterations):
        # 转 PIL L 模式做 box blur
        u8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(u8, mode="L")
        pil_blurred = pil_img.filter(ImageFilter.BoxBlur(kernel_size))
        blurred = np.array(pil_blurred, dtype=np.float32) / 255.0

        # 只在 mask 内应用平滑结果
        result = np.where(mask > 0, blurred, 0.0).astype(np.float32)

    return result
