"""Short-term conversation memory for the Agent."""

from typing import Any


Message = dict[str, Any]


class ConversationMemory:
    """Keep a bounded list of recent conversation messages."""

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 2:
            raise ValueError("max_messages 必须至少为 2")

        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Add one message and keep only the newest messages."""
        if not isinstance(message, dict):
            raise TypeError("message 必须是字典")

        if "role" not in message:
            raise ValueError("message 必须包含 role 字段")

        self._messages.append(message.copy())
        self._trim()

    def add_user_message(self, content: str) -> None:
        self.add(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(self, content: str) -> None:
        self.add(
            {
                "role": "assistant",
                "content": content,
            }
        )

    def get_messages(self) -> list[Message]:
        """Return a copy so callers cannot modify internal memory directly."""
        return [message.copy() for message in self._messages]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return

        trimmed = self._messages[-self.max_messages :]

        # 历史必须以 user 开头。按条数截断时，窗口头部可能剩下一条
        # 提问已被丢弃的 assistant 消息，模型会收到“没有对应提问的回答”。
        while trimmed and trimmed[0]["role"] != "user":
            trimmed.pop(0)

        self._messages = trimmed