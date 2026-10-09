from types import SimpleNamespace

from app.memory import ConversationMemory
from app.summarizer import (
    HARD_LIMIT_FACTOR,
    TRUNCATED_MARK,
    build_summarizer,
    enforce_limit,
)


def make_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_summarizer_returns_stripped_content():
    seen = []

    def fake_llm(messages, tools):
        seen.append(messages)
        return make_response("  用户名叫小明。  ")

    summarize = build_summarizer(fake_llm)

    assert summarize("", "user: 我叫小明") == "用户名叫小明。"

    request = seen[0]
    assert request[0]["role"] == "user"
    assert "我叫小明" in request[0]["content"]
    assert "（还没有摘要）" in request[0]["content"]


def test_summarizer_merges_previous_summary():
    seen = []

    def fake_llm(messages, tools):
        seen.append(messages)
        return make_response("合并后的摘要")

    summarize = build_summarizer(fake_llm, max_chars=123)
    summarize("用户叫小明", "user: 我在学 Agent")

    prompt = seen[0][0]["content"]

    assert "用户叫小明" in prompt
    assert "我在学 Agent" in prompt
    assert "123" in prompt
    assert "（还没有摘要）" not in prompt


def test_summarizer_never_sends_tools():
    """摘要不该让模型有机会调工具，否则压缩过程会自己发起工具调用。"""
    seen = []

    def fake_llm(messages, tools):
        seen.append(tools)
        return make_response("ok")

    build_summarizer(fake_llm)("", "x")

    assert seen == [[]]


def test_summarizer_handles_empty_content():
    def fake_llm(messages, tools):
        return make_response(None)

    assert build_summarizer(fake_llm)("", "x") == ""


def test_summarizer_handles_whitespace_only_old_summary():
    seen = []

    def fake_llm(messages, tools):
        seen.append(messages)
        return make_response("摘要")

    build_summarizer(fake_llm)("   ", "user: 你好")

    assert "（还没有摘要）" in seen[0][0]["content"]


# ============ Day 7：代码侧硬上限 ============

def test_enforce_limit_keeps_short_text():
    assert enforce_limit("短摘要", 100) == "短摘要"


def test_enforce_limit_truncates_and_marks():
    long_text = "字" * 50

    result = enforce_limit(long_text, 20)

    assert result.startswith("字" * 20)
    assert result.endswith(TRUNCATED_MARK)


def test_enforce_limit_with_zero_means_no_limit():
    """limit <= 0 视为不设限 —— 方便做对照实验。"""
    text = "字" * 500

    assert enforce_limit(text, 0) == text


def test_summarizer_default_hard_limit_is_double_max_chars():
    """提示词说「不超过 400 字」是**软约束**，代码要兜一道硬上限。"""
    seen = []

    def fake_llm(messages, tools):
        seen.append(messages)
        return make_response("字" * 5000)

    summarize = build_summarizer(fake_llm, max_chars=400)
    result = summarize("", "user: 你好")

    assert len(result) == 400 * HARD_LIMIT_FACTOR + len(TRUNCATED_MARK)
    assert result.endswith(TRUNCATED_MARK), "模型不听话时，代码要兜住"
    assert "400" in seen[0][0]["content"], "提示词里仍然要说清软约束"


def test_summarizer_accepts_explicit_hard_limit():
    def fake_llm(messages, tools):
        return make_response("字" * 100)

    assert build_summarizer(fake_llm, max_chars=10, hard_limit=30)("", "x") == (
        "字" * 30 + TRUNCATED_MARK
    )


def test_summarizer_hard_limit_does_not_touch_short_output():
    def fake_llm(messages, tools):
        return make_response("用户叫小明。")

    assert build_summarizer(fake_llm, max_chars=400)("", "x") == "用户叫小明。"


def test_memory_summary_stays_bounded_end_to_end():
    """整套跑下来，摘要不会无限长大。"""
    def fake_llm(messages, tools):
        return make_response("废话" * 1000)

    memory = ConversationMemory(max_messages=4, summarizer=build_summarizer(fake_llm, max_chars=50))

    for index in range(20):
        memory.add_user_message(f"消息 {index}")
        memory.add_assistant_message("好的。")

    assert len(memory.get_summary()) <= 50 * HARD_LIMIT_FACTOR + len(TRUNCATED_MARK)
