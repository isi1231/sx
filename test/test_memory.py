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


def test_memory_window_never_starts_with_assistant():
    """不变量：裁剪后历史必须以 user 开头，奇数窗口同样成立。"""
    for limit in (2, 3, 4, 5, 7, 10, 11):
        memory = ConversationMemory(max_messages=limit)

        for index in range(10):
            memory.add_user_message(f"问题 {index}")
            memory.add_assistant_message(f"答案 {index}")

            messages = memory.get_messages()

            assert messages, f"max={limit} 第 {index} 轮记忆被清空"
            assert messages[0]["role"] == "user", (
                f"max={limit} 第 {index} 轮历史错位: "
                f"{[message['role'] for message in messages]}"
            )


def test_memory_drops_orphan_assistant_on_trim():
    """窗口长度不能整除消息数时，头部孤儿 assistant 必须被丢弃。"""
    memory = ConversationMemory(max_messages=4)

    memory.add_user_message("问题 1")
    memory.add_assistant_message("答案 1")
    memory.add_user_message("问题 2")
    memory.add_assistant_message("答案 2")
    memory.add_user_message("问题 3")

    messages = memory.get_messages()

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "问题 2"


def test_memory_odd_window_keeps_answer_with_its_question():
    """奇数窗口收缩时，必须保留成对的一轮，而不是留下半截。"""
    memory = ConversationMemory(max_messages=3)

    memory.add_user_message("问题 1")
    memory.add_assistant_message("答案 1")
    memory.add_user_message("问题 2")
    memory.add_assistant_message("答案 2")

    messages = memory.get_messages()

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "问题 2"
    assert messages[1]["content"] == "答案 2"


def test_memory_all_assistant_window_degrades_to_empty():
    """窗口内没有任何 user 消息时降级为空，不留 assistant 开头的历史。"""
    memory = ConversationMemory(max_messages=2)

    for index in range(3):
        memory.add_assistant_message(f"答案 {index}")

    assert memory.get_messages() == []