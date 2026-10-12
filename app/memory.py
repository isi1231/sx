"""Short-term conversation memory with optional summary compression.

两种工作模式：
- 不传 summarizer：退化为纯滑窗，保留最近 max_messages 条
- 传入 summarizer：超出容量时保留最近 keep_recent 条，
  其余交给 summarizer 压缩成一段摘要（而不是直接丢弃）

## 为什么有「轮」和「条」两个单位（Day 8）

对外说「记住最近 5 轮」是**人**的说法；实现里窗口就是 10 条消息，这是**机器**的说法。
以前只有一个 `max_messages=10`，读代码的人得自己心算「=5 轮」，于是参数名一直在骗人。

现在：
- **`max_turns`**（推荐）—— 1 轮 = 1 组 `user` + `assistant` = 2 条
- **`max_messages`**（底层逃生舱）—— 直接给条数。

为什么还要留着条数入口？因为它能做回合做不到的事：**表达奇数窗口**。
「历史必须以 user 开头」这个不变量只在条数不能被 2 整除时才会被触发，
留一个条数入口，那条守卫就永远是被真实覆盖的活代码，而不是死分支。

本模块是纯逻辑，不碰网络 —— 摘要怎么生成由调用方注入。
"""

from typing import Any, Callable


Message = dict[str, Any]
Summarizer = Callable[[str, str], str]
"""摘要函数签名：(已有摘要, 新增对话文本) -> 新的完整摘要"""

MESSAGES_PER_TURN = 2
"""一轮对话占两条消息：一条 user、一条 assistant。"""

DEFAULT_MAX_TURNS = 5
"""默认记住最近 5 轮（= 10 条），与 Day 5 起的默认窗口一致。"""


def _resolve_max_messages(max_turns: int | None, max_messages: int | None) -> int:
    """把「轮」和「条」两种说法归一成条数。"""
    if max_turns is not None and max_messages is not None:
        raise ValueError("max_turns 和 max_messages 只能给一个")

    if max_messages is not None:
        limit = int(max_messages)
        if limit < 2:
            raise ValueError("max_messages 必须至少为 2")
        return limit

    turns = DEFAULT_MAX_TURNS if max_turns is None else int(max_turns)
    if turns < 1:
        raise ValueError("max_turns 必须至少为 1")
    return turns * MESSAGES_PER_TURN


def _resolve_keep_recent(
    keep_recent: int | None,
    keep_recent_turns: int | None,
    max_messages: int,
) -> int:
    if keep_recent is not None and keep_recent_turns is not None:
        raise ValueError("keep_recent 和 keep_recent_turns 只能给一个")

    if keep_recent_turns is not None:
        turns = int(keep_recent_turns)
        if turns < 1:
            raise ValueError("keep_recent_turns 必须至少为 1")
        keep = turns * MESSAGES_PER_TURN
    elif keep_recent is not None:
        keep = int(keep_recent)
    else:
        # 缺省保留一半 —— 窗口是 10 条时保留 5 条，和 Day 5 起的默认行为一致
        keep = max(2, max_messages // 2)

    if keep < 2:
        raise ValueError("keep_recent 必须至少为 2")
    if keep > max_messages:
        raise ValueError("keep_recent 不能大于 max_messages")
    return keep


def format_messages(messages: list[Message]) -> str:
    """把消息列表渲染成给摘要器看的纯文本。"""
    lines: list[str] = []

    for message in messages:
        role = message.get("role", "unknown")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


class ConversationMemory:
    """Keep a bounded conversation, compressing the oldest part into a summary.

    容量用 `max_turns` 表达（推荐）或 `max_messages`（底层逃生舱），二选一。
    """

    def __init__(
        self,
        max_turns: int | None = None,
        max_messages: int | None = None,
        keep_recent: int | None = None,
        keep_recent_turns: int | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        resolved_max = _resolve_max_messages(max_turns, max_messages)
        resolved_keep = _resolve_keep_recent(
            keep_recent, keep_recent_turns, resolved_max
        )

        self.max_messages = resolved_max
        self.keep_recent = resolved_keep
        self.summarizer = summarizer

        self._messages: list[Message] = []
        self._summary = ""
        self._compressed_count = 0

    # ---------- 容量的两种说法 ----------

    @property
    def max_turns(self) -> int:
        """窗口能容纳的完整轮数。条数模式下向下取整。"""
        return self.max_messages // MESSAGES_PER_TURN

    @property
    def keep_recent_turns(self) -> int:
        """压缩时保留的最近轮数。条数模式下向下取整。"""
        return self.keep_recent // MESSAGES_PER_TURN

    # ---------- 写入 ----------

    def add(self, message: Message) -> None:
        """Add one message and keep the memory within its bound."""
        if not isinstance(message, dict):
            raise TypeError("message 必须是字典")

        if "role" not in message:
            raise ValueError("message 必须包含 role 字段")

        self._messages.append(message.copy())
        self._trim()

    def add_user_message(self, content: str) -> None:
        self.add({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.add({"role": "assistant", "content": content})

    # ---------- 读取 ----------

    def get_messages(self) -> list[Message]:
        """Return a copy so callers cannot modify internal memory directly."""
        return [message.copy() for message in self._messages]

    def get_summary(self) -> str:
        """Return the compressed summary of everything already trimmed away."""
        return self._summary

    @property
    def compressed_count(self) -> int:
        """已经被压缩（或丢弃）的消息条数，用于观测。"""
        return self._compressed_count

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""
        self._compressed_count = 0

    def __len__(self) -> int:
        return len(self._messages)

    # ---------- 内部 ----------

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return

        if self.summarizer is None:
            self._trim_by_window()
            return

        cut = len(self._messages) - self.keep_recent
        overflow = self._messages[:cut]
        kept = self._messages[cut:]

        # 历史必须以 user 开头：把因截断而成为孤儿的 assistant 一并移出
        while kept and kept[0]["role"] != "user":
            overflow.append(kept.pop(0))

        self._compress(overflow)
        self._messages = kept

    def _trim_by_window(self) -> None:
        """没有摘要器时的退化路径：纯滑窗，保留最近 max_messages 条。"""
        kept = self._messages[-self.max_messages :]

        # 历史必须以 user 开头。按条数截断时，窗口头部可能剩下一条
        # 提问已被丢弃的 assistant 消息，模型会收到“没有对应提问的回答”。
        while kept and kept[0]["role"] != "user":
            kept.pop(0)

        self._messages = kept

    def _compress(self, messages: list[Message]) -> None:
        if not messages:
            return

        self._compressed_count += len(messages)

        text = format_messages(messages)
        if not text or self.summarizer is None:
            return

        self._summary = self.summarizer(self._summary, text)
