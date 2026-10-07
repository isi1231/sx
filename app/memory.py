"""Short-term conversation memory with optional summary compression.

两种工作模式：
- 不传 summarizer：退化为纯滑窗，保留最近 max_messages 条
- 传入 summarizer：超出容量时保留最近 keep_recent 条，
  其余交给 summarizer 压缩成一段摘要（而不是直接丢弃）

本模块是纯逻辑，不碰网络 —— 摘要怎么生成由调用方注入。
"""

from typing import Any, Callable


Message = dict[str, Any]
Summarizer = Callable[[str, str], str]
"""摘要函数签名：(已有摘要, 新增对话文本) -> 新的完整摘要"""


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
    """Keep a bounded conversation, compressing the oldest part into a summary."""

    def __init__(
        self,
        max_messages: int = 10,
        keep_recent: int | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        if max_messages < 2:
            raise ValueError("max_messages 必须至少为 2")

        if keep_recent is None:
            keep_recent = max(2, max_messages // 2)

        if keep_recent < 2:
            raise ValueError("keep_recent 必须至少为 2")

        if keep_recent > max_messages:
            raise ValueError("keep_recent 不能大于 max_messages")

        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.summarizer = summarizer

        self._messages: list[Message] = []
        self._summary = ""
        self._compressed_count = 0

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
