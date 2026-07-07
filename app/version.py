"""
版本号定义
每次发布新版本时，只需修改此文件中的 __version__ 值。

—— v0.8.10 加固版新增 ——
__update_protocol__       : 更新协议代际。每次做破坏性变更（架构/结构性调整）时 +1。
                            旧客户端若发现服务端协议 > 自己，必须走完整包更新（不再走增量）。
__min_compatible_version__: 从该版本起可与本版本安全走增量更新。
                            低于此版本的旧客户端会被引导下载完整包。
"""

__version__ = "0.8.10"

# —— 更新协议元数据（v0.8.10 加固版起启用） ——
# 修改准则：只要 RELEASE.md 阶段 1.5「架构变更红线检查表」有任一项为"是"，
# 就必须把 __update_protocol__ 递增，并同步刷新 __min_compatible_version__。
__update_protocol__ = 2
__min_compatible_version__ = "0.8.10"
