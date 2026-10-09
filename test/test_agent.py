import json
from types import SimpleNamespace

import pytest

from app.agent import (
    MAX_TOOL_LOG_ENTRIES,
    TOOL_RESULT_LIMIT,
    AgentSession,
    _clip,
    _message_to_dict,
    _read_usage,
    run_agent,
)
from app.trace import TraceRecorder, load_events, render_replay


def make_response(message, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )


def make_tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def answer(text, usage=None):
    return make_response(
        SimpleNamespace(role="assistant", content=text, tool_calls=None), usage
    )


def tool_round(*calls, usage=None):
    return make_response(
        SimpleNamespace(role="assistant", content=None, tool_calls=list(calls)),
        usage,
    )


# ============ Day 4：工具调用 ============


def test_agent_calls_tool_then_returns_final_answer():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", '{"expression":"2 + 3"}'))
        return answer("计算结果是 5。")

    assert run_agent("2 + 3 等于多少？", llm_call=fake_llm) == "计算结果是 5。"
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "tool"
    assert calls[1][-1]["content"] == "5"


def test_agent_stops_at_max_rounds():
    def fake_llm(messages, tools):
        return tool_round(make_tool_call("calculator", '{"expression":"1 + 1"}'))

    assert "超过限制" in run_agent("循环调用", llm_call=fake_llm, max_rounds=2)


def test_agent_rejects_broken_tool_arguments():
    """参数不是合法 JSON 时不执行工具，把错误文本送回模型让它自己纠。"""
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", "{不是 JSON"))
        return answer("我换个写法。")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)
    agent.ask("算一下")

    tool_messages = [m for m in calls[1] if m["role"] == "tool"]
    assert "参数不是合法 JSON" in tool_messages[0]["content"]


# ============ Day 5：多轮记忆 ============


def test_agent_remembers_previous_turn():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)

        if len(calls) == 1:
            return answer("你好，小明。")

        previous_user_messages = [
            message for message in messages if message.get("role") == "user"
        ]

        assert any("我叫小明" in message["content"] for message in previous_user_messages)

        return answer("你叫小明。")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)

    assert agent.ask("我叫小明") == "你好，小明。"
    assert agent.ask("我叫什么？") == "你叫小明。"


def test_agent_clear_memory():
    agent = AgentSession(llm_call=lambda messages, tools: answer("收到。"), max_messages=10)

    agent.ask("我叫小明")
    agent.clear_memory()

    assert agent.get_memory() == []


def test_agent_history_always_starts_with_user():
    """回归测试：窗口溢出后，发给模型的历史首条必须是 user。

    修复前 max_messages=10 从第 6 轮起会变成 assistant 开头，
    因为 _trim 在加完 user（此时长度为奇数）之后立即按条数截断。
    """
    seen = []

    def fake_llm(messages, tools):
        seen.append([message["role"] for message in messages])

        assert messages[1]["role"] == "user", (
            f"第 {len(seen)} 轮历史错位: "
            f"{[message['role'] for message in messages]}"
        )

        return answer("收到。")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)

    # 必须跑到超过窗口容量（10 条 = 5 轮）才能触发裁剪
    for index in range(20):
        agent.ask(f"第 {index} 个问题")

    assert len(seen) == 20
    assert all(roles[1] == "user" for roles in seen)


def test_agent_memory_stays_paired_after_many_turns():
    """回归测试：多轮之后记忆里不能出现孤儿的 assistant 开头。"""
    agent = AgentSession(llm_call=lambda messages, tools: answer("收到。"), max_messages=10)

    for index in range(20):
        agent.ask(f"第 {index} 个问题")
        assert agent.get_memory()[0]["role"] == "user"


# ============ Day 6：上下文工程 ============


def test_clip_flattens_and_truncates():
    assert _clip("a\n\n  b") == "a b"

    exact = "x" * TOOL_RESULT_LIMIT
    assert _clip(exact) == exact

    too_long = "y" * (TOOL_RESULT_LIMIT + 50)
    clipped = _clip(too_long)

    assert clipped.startswith("y" * TOOL_RESULT_LIMIT)
    assert "已截断" in clipped
    assert len(clipped) < len(too_long)


def test_agent_failed_request_leaves_memory_untouched():
    """修复点：请求失败时不能留下没有回答的孤儿 user 消息。"""
    def boom(messages, tools):
        raise RuntimeError("网络挂了")

    agent = AgentSession(llm_call=boom, max_messages=10)

    with pytest.raises(RuntimeError):
        agent.ask("我叫小明")

    assert agent.get_memory() == []


def test_agent_memory_stays_paired_after_failure_then_success():
    state = {"failed": False}

    def flaky(messages, tools):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("第一次失败")
        return answer("这次成功。")

    agent = AgentSession(llm_call=flaky, max_messages=10)

    with pytest.raises(RuntimeError):
        agent.ask("第一个问题")

    assert agent.ask("第二个问题") == "这次成功。"

    memory = agent.get_memory()
    assert [message["role"] for message in memory] == ["user", "assistant"]
    assert "第二个问题" in memory[0]["content"]
    assert "第一个问题" not in memory[0]["content"]


def test_agent_tool_log_enters_memory_and_reaches_next_turn():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", '{"expression":"2 + 3"}'))
        return answer("答案是 5。")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)

    assert agent.ask("2 + 3 等于多少") == "答案是 5。"

    # 答案给用户看的是干净正文，记忆里则带着工具流水
    assistant_in_memory = agent.get_memory()[1]["content"]
    assert assistant_in_memory.startswith("答案是 5。")
    assert "[本轮工具调用]" in assistant_in_memory
    assert "-> 5" in assistant_in_memory

    agent.ask("刚才那个结果是多少？")

    # 下一轮的上下文里确实带上了上一轮的工具结果
    assert any(
        message["role"] == "assistant"
        and "[本轮工具调用]" in (message["content"] or "")
        for message in calls[-1]
    )


def test_agent_clips_long_tool_result_in_memory(monkeypatch):
    long_text = "很长的工具结果。" * 100
    monkeypatch.setattr("app.agent.execute_tool", lambda name, arguments: long_text)

    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("search_notes", '{"query":"Agent"}'))
        return answer("找到了。")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)
    agent.ask("搜一下笔记")

    assistant_in_memory = agent.get_memory()[1]["content"]

    assert "已截断" in assistant_in_memory
    assert len(assistant_in_memory) < len(long_text)

    # 但发给模型的原始工具消息没有被削 —— 削的只是进记忆的那份
    tool_messages = [m for m in calls[1] if m["role"] == "tool"]
    assert tool_messages[0]["content"] == long_text


def test_agent_caps_tool_log_entries():
    tool_calls = [
        make_tool_call("calculator", '{"expression":"1 + 1"}', call_id=f"call-{index}")
        for index in range(MAX_TOOL_LOG_ENTRIES + 3)
    ]

    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(*tool_calls)
        return answer("都算完了。")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)
    agent.ask("算很多个 1+1")

    assistant_in_memory = agent.get_memory()[1]["content"]
    assert assistant_in_memory.count("calculator(") == MAX_TOOL_LOG_ENTRIES

    # 同一请求内的工具消息一条不少
    assert len([m for m in calls[1] if m["role"] == "tool"]) == len(tool_calls)


def test_agent_summary_is_injected_as_system_message():
    seen = []

    def fake_llm(messages, tools):
        seen.append(messages)
        return answer("收到。")

    def fake_summarizer(old_summary, new_text):
        return "用户名叫小明，正在学习 Agent。"

    agent = AgentSession(
        llm_call=fake_llm,
        max_messages=4,
        summarizer=fake_summarizer,
    )

    for index in range(6):
        agent.ask(f"第 {index} 个问题")

    assert agent.get_summary() == "用户名叫小明，正在学习 Agent。"
    assert agent.compressed_count > 0

    last = seen[-1]
    system_contents = [m["content"] for m in last if m["role"] == "system"]

    assert len(system_contents) == 2, "规则 + 摘要各占一条 system"
    assert "<conversation_summary>" in system_contents[1]
    assert "用户名叫小明" in system_contents[1]

    # 顺序：规则 → 摘要 → 历史 → 本轮提问
    assert [m["role"] for m in last[:3]] == ["system", "system", "user"]


def test_agent_without_summarizer_has_no_summary_message():
    """对照组：不传摘要器时，上下文里不该凭空多出一条 system。"""
    seen = []

    def fake_llm(messages, tools):
        seen.append(messages)
        return answer("收到。")

    agent = AgentSession(llm_call=fake_llm, max_messages=4)

    for index in range(6):
        agent.ask(f"第 {index} 个问题")

    assert agent.get_summary() == ""
    assert all(
        len([m for m in messages if m["role"] == "system"]) == 1
        for messages in seen
    )


def test_agent_asks_nothing_when_input_is_blank():
    def fake_llm(messages, tools):
        raise AssertionError("空白输入不该发起请求")

    agent = AgentSession(llm_call=fake_llm, max_messages=10)

    assert agent.ask("   ") == "请输入问题。"
    assert agent.get_memory() == []


# ============ Day 7：可观测性 ============

# ---- 先修好「消息形状」，trace 才可能落盘 ----


def test_message_to_dict_gives_same_keys_for_all_inputs():
    """回归测试：三种输入必须产出**同一套键**。

    修复前三个分支各返回一种形状：走 model_dump 的可能没有 content，
    走兜底的又没有 tool_call_id。下游没法按键取值。
    """
    tool_call = make_tool_call("calculator", '{"expression":"1+1"}')

    from_sdk_object = _message_to_dict(
        SimpleNamespace(role="assistant", content="在", tool_calls=None)
    )
    from_dict = _message_to_dict({"role": "assistant", "content": "在"})
    from_dumpable = _message_to_dict(
        SimpleNamespace(
            role="assistant",
            content="在",
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "content": "在",
                "tool_calls": None,
            },
        )
    )

    assert from_sdk_object == from_dict == from_dumpable == {
        "role": "assistant",
        "content": "在",
    }
    assert set(from_sdk_object) == {"role", "content"}

    del tool_call


def test_message_to_dict_always_has_role_and_content():
    """哪怕输入什么都没带，这两个键也必须在。"""
    assert _message_to_dict({}) == {"role": "assistant", "content": None}
    assert _message_to_dict({"content": "只有内容"}) == {
        "role": "assistant",
        "content": "只有内容",
    }


def test_message_to_dict_expands_tool_calls_into_plain_dicts():
    """SDK 的 tool_call 是对象，必须展开成 dict，否则 json.dumps 会炸。"""
    call = make_tool_call("calculator", '{"expression":"2 + 3"}', call_id="call-9")

    normalized = _message_to_dict(
        SimpleNamespace(role="assistant", content=None, tool_calls=[call])
    )

    assert normalized["content"] is None
    assert normalized["tool_calls"] == [
        {
            "id": "call-9",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression":"2 + 3"}'},
        }
    ]
    assert json.loads(json.dumps(normalized))["tool_calls"][0]["id"] == "call-9"


def test_message_to_dict_keeps_tool_message_fields():
    normalized = _message_to_dict(
        {"role": "tool", "tool_call_id": "call-1", "name": "calculator", "content": "5"}
    )

    assert normalized["role"] == "tool"
    assert normalized["tool_call_id"] == "call-1"
    assert normalized["name"] == "calculator"


# ---- token 用量读取 ----


def test_read_usage_returns_empty_without_usage():
    assert _read_usage(SimpleNamespace()) == {}


def test_read_usage_picks_up_cache_fields():
    usage = SimpleNamespace(
        prompt_tokens=812,
        completion_tokens=24,
        total_tokens=836,
        prompt_cache_hit_tokens=768,
        prompt_cache_miss_tokens=44,
    )

    read = _read_usage(SimpleNamespace(usage=usage))

    assert read["prompt_cache_hit_tokens"] == 768
    assert read["prompt_cache_miss_tokens"] == 44


def test_read_usage_drops_non_integer_fields():
    usage = {
        "prompt_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 8},
        "note": "不是数字",
    }

    assert _read_usage(SimpleNamespace(usage=usage)) == {"prompt_tokens": 10}


# ---- 打点 ----


def test_agent_without_tracer_creates_none():
    agent = AgentSession(llm_call=lambda messages, tools: answer("收到。"), max_messages=10)

    assert agent.tracer is None, "不传 tracer 就不该有额外开销"


def test_agent_trace_records_a_full_tool_round():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", '{"expression":"2 + 3"}'))
        return answer("答案是 5。")

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=fake_llm, max_messages=10, tracer=tracer)
    agent.ask("2 + 3 等于多少")

    assert [event["kind"] for event in tracer.events] == [
        "request",
        "response",
        "tool_call",
        "request",
        "response",
        "answer",
    ]


def test_agent_trace_snapshots_each_request():
    """不变量：每次 request 存的是**那一刻**的快照，不是会长大的活列表。

    如果 emit 存的是引用，第 1 个 request 会跟着变成 4 条消息 ——
    回放时就看不出「模型当时只看到 2 条」了。
    """
    calls = []

    def fake_llm(messages, tools):
        calls.append(len(messages))
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", '{"expression":"2 + 3"}'))
        return answer("答案是 5。")

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=fake_llm, max_messages=10, tracer=tracer)
    agent.ask("2 + 3 等于多少")

    requests = [event for event in tracer.events if event["kind"] == "request"]
    counts = [len(event["payload"]["model_messages"]) for event in requests]

    assert counts == [2, 4], f"两次请求的消息条数应分别是 2 和 4，实际 {counts}"


def test_agent_trace_records_tool_call_payload():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", '{"expression":"25 * 4 + 10"}'))
        return answer("110")

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=fake_llm, max_messages=10, tracer=tracer)
    agent.ask("算一下")

    tool_event = next(event for event in tracer.events if event["kind"] == "tool_call")
    payload = tool_event["payload"]

    assert payload["name"] == "calculator"
    assert payload["arguments"] == '{"expression":"25 * 4 + 10"}'
    assert payload["result"] == "110"
    assert payload["ok"] is True
    assert payload["latency_ms"] >= 0


def test_agent_trace_marks_failed_tool_call(monkeypatch):
    def boom(name, arguments):
        raise RuntimeError("工具内部炸了")

    monkeypatch.setattr("app.agent.execute_tool", boom)

    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return tool_round(make_tool_call("calculator", '{"expression":"1 + 1"}'))
        return answer("失败了。")

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=fake_llm, max_messages=10, tracer=tracer)
    agent.ask("算一下")

    tool_event = next(event for event in tracer.events if event["kind"] == "tool_call")

    assert tool_event["payload"]["ok"] is False
    assert tool_event["payload"]["error"] == "RuntimeError"


def test_agent_trace_records_usage_and_latency():
    def fake_llm(messages, tools):
        return answer(
            "收到。",
            usage=SimpleNamespace(
                prompt_tokens=812,
                completion_tokens=24,
                total_tokens=836,
                prompt_cache_hit_tokens=768,
                prompt_cache_miss_tokens=44,
            ),
        )

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=fake_llm, max_messages=10, tracer=tracer)
    agent.ask("你好")

    response_event = next(event for event in tracer.events if event["kind"] == "response")

    assert response_event["payload"]["usage"]["total_tokens"] == 836
    assert response_event["payload"]["latency_ms"] >= 0

    answer_event = next(event for event in tracer.events if event["kind"] == "answer")
    assert answer_event["payload"]["turn_latency_ms"] >= 0


def test_agent_trace_records_error_then_leaves_memory_clean():
    """Day 6 的「失败不污染记忆」在 trace 这边也要看得见。"""
    def boom(messages, tools):
        raise RuntimeError("模型服务 502")

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=boom, max_messages=10, tracer=tracer)

    with pytest.raises(RuntimeError):
        agent.ask("我叫小明")

    assert [event["kind"] for event in tracer.events] == ["request", "error"]
    assert tracer.events[-1]["payload"]["phase"] == "llm_call"
    assert tracer.events[-1]["payload"]["type"] == "RuntimeError"
    assert agent.get_memory() == []


def test_agent_trace_marks_timeout_reason():
    def fake_llm(messages, tools):
        return tool_round(make_tool_call("calculator", '{"expression":"1 + 1"}'))

    tracer = TraceRecorder()
    agent = AgentSession(llm_call=fake_llm, max_messages=10, max_tool_rounds=2, tracer=tracer)
    agent.ask("一直算")

    answer_event = tracer.events[-1]

    assert answer_event["kind"] == "answer"
    assert answer_event["payload"]["stopped_by"] == "max_tool_rounds"


def test_agent_trace_numbers_turns():
    tracer = TraceRecorder()
    agent = AgentSession(
        llm_call=lambda messages, tools: answer("收到。"),
        max_messages=10,
        tracer=tracer,
    )

    agent.ask("第一个问题")
    agent.ask("第二个问题")

    # 每轮 3 个事件：request / response / answer，轮号连续递增
    assert [event["turn"] for event in tracer.events] == [1, 1, 1, 2, 2, 2]
    assert [event["step"] for event in tracer.events] == [1, 2, 3, 1, 2, 3]


def test_agent_blank_input_does_not_start_a_turn():
    tracer = TraceRecorder()
    agent = AgentSession(
        llm_call=lambda messages, tools: answer("收到。"),
        max_messages=10,
        tracer=tracer,
    )

    assert agent.ask("   ") == "请输入问题。"
    assert tracer.current_turn == 0
    assert tracer.events == []


# ---- 端到端：agent → jsonl → 回放 ----


def test_trace_round_trips_through_disk_and_replays(tmp_path):
    """验收标准：一次任务能完整回放，能定位到具体哪一轮出错。"""
    calls = []

    def flaky_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("第一次就挂")
        if len(calls) == 2:
            return tool_round(make_tool_call("calculator", '{"expression":"25 * 4 + 10"}'))
        return answer("计算结果是 110。")

    path = tmp_path / "nested" / "trace.jsonl"
    tracer = TraceRecorder(trace_id="t-e2e", path=path)
    agent = AgentSession(llm_call=flaky_llm, max_messages=10, tracer=tracer)

    with pytest.raises(RuntimeError):
        agent.ask("算一下 25 * 4 + 10")

    assert agent.ask("再试一次") == "计算结果是 110。"

    events = load_events(path)

    # 每一行都是合法 JSON —— 这正是统一消息形状换来的
    assert len(events) == len(tracer.events)
    assert {event["trace_id"] for event in events} == {"t-e2e"}

    report = render_replay(events, show_messages=True)

    assert "第 1 轮" in report and "第 2 轮" in report
    assert "出错      1 次" in report
    assert "第 1 轮 llm_call：RuntimeError 第一次就挂" in report
    assert "25 * 4 + 10" in report
    assert "计算结果是 110。" in report
    assert "[system]" in report, "show_messages 要把真正发出去的上下文打出来"
