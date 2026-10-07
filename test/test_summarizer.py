from types import SimpleNamespace

from app.summarizer import build_summarizer


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
