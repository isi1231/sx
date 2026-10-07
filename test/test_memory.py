import pytest

from app.memory import ConversationMemory, format_messages


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


# ============ Day 6：摘要压缩 ============


def make_recording_summarizer():
    """假摘要器：记录每次调用的入参，返回可预测的编号摘要。"""
    calls: list[tuple[str, str]] = []

    def summarize(old_summary: str, new_text: str) -> str:
        calls.append((old_summary, new_text))
        return f"摘要{len(calls)}"

    return summarize, calls


def test_format_messages_skips_empty_content():
    text = format_messages(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "在"},
        ]
    )

    assert text == "user: 你好\nassistant: 在"


def test_memory_without_summarizer_loses_old_content():
    """对照组：不给摘要器时，被裁掉的内容永久丢失。"""
    memory = ConversationMemory(max_messages=4)

    for index in range(6):
        memory.add_user_message(f"消息 {index}")

    contents = [message["content"] for message in memory.get_messages()]

    assert memory.get_summary() == ""
    assert memory.compressed_count == 0
    assert "消息 0" not in contents


def test_memory_summarizer_receives_overflow():
    summarize, calls = make_recording_summarizer()
    memory = ConversationMemory(max_messages=4, summarizer=summarize)

    # 满 4 条时还没超容量，不该压缩
    for index in range(4):
        memory.add_user_message(f"消息 {index}")

    assert calls == []
    assert memory.get_summary() == ""

    memory.add_user_message("消息 4")

    assert len(calls) == 1
    old_summary, new_text = calls[0]
    assert old_summary == ""
    assert "消息 0" in new_text
    assert "消息 4" not in new_text, "保留区的内容不该被压进摘要"


def test_memory_summary_is_cumulative():
    """第二次压缩必须把上一次的摘要带回去合并，而不是从零重写。"""
    summarize, calls = make_recording_summarizer()
    memory = ConversationMemory(max_messages=4, summarizer=summarize)

    for index in range(8):
        memory.add_user_message(f"消息 {index}")

    assert len(calls) >= 2
    assert calls[0][0] == ""
    assert calls[1][0] == "摘要1"


def test_memory_compressed_count_equals_disappeared_messages():
    summarize, _ = make_recording_summarizer()
    memory = ConversationMemory(max_messages=4, summarizer=summarize)

    for index in range(6):
        memory.add_user_message(f"消息 {index}")

    # 不变量：写进去的每条消息，要么还在窗口里，要么已经被压缩
    assert memory.compressed_count + len(memory) == 6
    assert memory.compressed_count == 3


def test_memory_summarizer_mode_keeps_window_bounded_and_paired():
    """摘要模式下，窗口依旧有界，且首条永远不是 assistant。"""
    for limit in (2, 3, 4, 5, 7, 10):
        summarize, _ = make_recording_summarizer()
        memory = ConversationMemory(max_messages=limit, summarizer=summarize)

        for index in range(12):
            memory.add_user_message(f"问题 {index}")
            memory.add_assistant_message(f"答案 {index}")

            messages = memory.get_messages()

            assert len(messages) <= limit, f"max={limit} 窗口溢出到 {len(messages)}"
            if messages:
                assert messages[0]["role"] == "user", (
                    f"max={limit} 第 {index} 轮错位: "
                    f"{[message['role'] for message in messages]}"
                )


def test_memory_summarizer_receives_orphan_assistant():
    summarize, calls = make_recording_summarizer()
    memory = ConversationMemory(max_messages=2, keep_recent=2, summarizer=summarize)

    memory.add_user_message("问题 1")
    memory.add_assistant_message("答案 1")
    memory.add_user_message("问题 2")

    # 溢出区里的孤儿 assistant 也进了摘要 —— 它的消失有据可查
    assert len(calls) == 1
    assert "答案 1" in calls[0][1]


def test_memory_clear_resets_summary():
    summarize, _ = make_recording_summarizer()
    memory = ConversationMemory(max_messages=4, summarizer=summarize)

    for index in range(6):
        memory.add_user_message(f"消息 {index}")

    assert memory.get_summary()

    memory.clear()

    assert memory.get_summary() == ""
    assert memory.compressed_count == 0
    assert memory.get_messages() == []


def test_memory_survives_summarizer_failure():
    """摘要器抛异常时不能把记忆搞坏（异常交给调用方决定怎么处理）。"""
    def broken(old_summary: str, new_text: str) -> str:
        raise RuntimeError("摘要服务挂了")

    memory = ConversationMemory(max_messages=4, summarizer=broken)

    for index in range(4):
        memory.add_user_message(f"消息 {index}")

    with pytest.raises(RuntimeError):
        memory.add_user_message("消息 4")

    # 失败前写入的消息仍然完好，没有被裁掉半截
    assert [message["content"] for message in memory.get_messages()] == [
        "消息 0",
        "消息 1",
        "消息 2",
        "消息 3",
        "消息 4",
    ]


def test_memory_rejects_invalid_keep_recent():
    with pytest.raises(ValueError):
        ConversationMemory(max_messages=4, keep_recent=1)

    with pytest.raises(ValueError):
        ConversationMemory(max_messages=4, keep_recent=5)


def test_memory_default_keep_recent_is_half():
    memory = ConversationMemory(max_messages=10)

    assert memory.keep_recent == 5


def test_summary_compression_keeps_what_hard_truncation_drops():
    """Day 6 的核心命题：同一段对话，滑窗把早期信息丢了，摘要把它留下。

    想看两边的实际内容：`python -m pytest -s -k keeps_what`
    """
    turns = ["我叫小明", "我在做猪场项目", "继续", "继续"]
    limit = 4

    def keep_facts(old_summary: str, new_text: str) -> str:
        """极简摘要器：捡出事实。

        注意它必须能读回自己上一次的输出（「用户叫小明」），
        否则第二次合并时就会把早期信息丢掉 —— 真摘要器也有这个要求。
        """
        merged = f"{old_summary}\n{new_text}"

        facts = []
        if "小明" in merged:
            facts.append("用户叫小明")
        if "猪场" in merged:
            facts.append("用户在做猪场项目")

        return "；".join(facts)

    window = ConversationMemory(max_messages=limit)
    compressed = ConversationMemory(max_messages=limit, summarizer=keep_facts)

    for turn in turns:
        for memory in (window, compressed):
            memory.add_user_message(turn)
            memory.add_assistant_message("好的。")

    window_contents = [message["content"] for message in window.get_messages()]

    print()
    print(f"[对照] 同样的 {len(turns)} 轮、同样的 max_messages={limit}")
    print(f"  纯滑窗   窗口内 = {window_contents}")
    print(f"           摘要   = {window.get_summary()!r}")
    print(f"           「小明」还在吗 = {'小明' in ' '.join(window_contents)}")
    print(f"  摘要压缩 窗口内 = {[m['content'] for m in compressed.get_messages()]}")
    print(f"           摘要   = {compressed.get_summary()!r}")
    print(f"           「小明」还在吗 = {'小明' in compressed.get_summary()}")

    # 滑窗：信息彻底蒸发
    assert window.get_summary() == ""
    assert window.compressed_count == 0
    assert "小明" not in " ".join(window_contents)

    # 摘要压缩：窗口一样小，但信息还在
    assert len(compressed) <= limit
    assert "小明" in compressed.get_summary()

    # 两种模式下窗口都是有界的；但只有摘要模式能给出「消息去哪了」的账
    assert len(window) <= limit
    assert compressed.compressed_count + len(compressed) == 2 * len(turns)
    assert window.compressed_count == 0, "纯滑窗没有摘要，也就无从记账"
