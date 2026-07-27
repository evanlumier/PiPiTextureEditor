"""
undo.py —— 统一撤销/重做栈

逻辑收口，步骤各自存储。各 tab 持有独立的 UndoStack 实例，
快照类型由调用方决定（ndarray / list / 任意）。

约定：
  - UndoStack 内部不做 copy，调用方 push 前自行 copy 快照。
  - max_len 由各使用方按需指定，保留各自历史行为。
  - push 新快照时清空 redo 栈（标准 undo/redo 语义：新操作使 redo 失效）。
"""


class UndoStack:
    """通用撤销/重做栈。LIFO，超限丢弃最旧。"""

    __slots__ = ("_undo", "_redo", "_max")

    def __init__(self, max_len: int = 30):
        self._undo: list = []
        self._redo: list = []
        self._max = max_len

    def push(self, snapshot) -> None:
        """压入 undo 快照。超限时丢弃最旧的一项。同时清空 redo 栈。"""
        self._undo.append(snapshot)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def push_undo_only(self, snapshot) -> None:
        """压入 undo 快照但不清 redo 栈。仅供 redo 回压 undo 使用，保持链路对称。"""
        self._undo.append(snapshot)
        if len(self._undo) > self._max:
            self._undo.pop(0)

    def pop(self):
        """弹出 undo 栈最新快照。栈空返回 None。"""
        if not self._undo:
            return None
        return self._undo.pop()

    def push_redo(self, snapshot) -> None:
        """压入 redo 快照。超限时丢弃最旧的一项。"""
        self._redo.append(snapshot)
        if len(self._redo) > self._max:
            self._redo.pop(0)

    def pop_redo(self):
        """弹出 redo 栈最新快照。栈空返回 None。"""
        if not self._redo:
            return None
        return self._redo.pop()

    def clear(self) -> None:
        """清空 undo + redo 栈。"""
        self._undo.clear()
        self._redo.clear()

    def clear_redo(self) -> None:
        """仅清空 redo 栈。"""
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)

    def __bool__(self) -> bool:
        return bool(self._undo)
