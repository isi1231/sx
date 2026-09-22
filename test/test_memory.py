import pytest

from app.memory import ConversationMemory


def test_memory_stores_messages():
    memory = ConversationMemory(max_messages=4)

    memory.add_user_message("你好")
    memory.add_assistant_message("你好，有什么可以帮助你？")

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_memory_keeps_recent_messages_only():
    memory = ConversationMemory(max_messages=4)

    for index in range(6):
        memory.add_user_message(f"消息 {index}")

    messages = memory.get_messages()

    assert len(messages) == 4
    assert messages[0]["content"] == "消息 2"
    assert messages[-1]["content"] == "消息 5"


def test_memory_clear():
    memory = ConversationMemory(max_messages=4)

    memory.add_user_message("你好")
    memory.clear()

    assert memory.get_messages() == []
    assert len(memory) == 0


def test_memory_rejects_invalid_limit():
    with pytest.raises(ValueError):
        ConversationMemory(max_messages=1)


def test_memory_returns_copy():
    memory = ConversationMemory(max_messages=4)
    memory.add_user_message("原始消息")

    messages = memory.get_messages()
    messages[0]["content"] = "被修改的消息"

    assert memory.get_messages()[0]["content"] == "原始消息"