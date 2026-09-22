from types import SimpleNamespace

from app.agent import run_agent


def response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_calls_tool_then_returns_final_answer():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="calculator", arguments='{"expression":"2 + 3"}'
                ),
            )
            return response(SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call]))
        return response(SimpleNamespace(role="assistant", content="计算结果是 5。", tool_calls=None))

    assert run_agent("2 + 3 等于多少？", llm_call=fake_llm) == "计算结果是 5。"
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "tool"
    assert calls[1][-1]["content"] == "5"


def test_agent_stops_at_max_rounds():
    def fake_llm(messages, tools):
        tool_call = SimpleNamespace(
            id="call-loop",
            function=SimpleNamespace(name="calculator", arguments='{"expression":"1 + 1"}'),
        )
        return response(SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call]))

    assert "超过限制" in run_agent("循环调用", llm_call=fake_llm, max_rounds=2)



from types import SimpleNamespace
from app.agent import AgentSession
def make_response(message):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def test_agent_remembers_previous_turn():
    calls = []

    def fake_llm(messages, tools):
        calls.append(messages)

        if len(calls) == 1:
            return make_response(
                SimpleNamespace(
                    content="你好，小明。",
                    tool_calls=None,
                )
            )

        previous_user_messages = [
            message
            for message in messages
            if message.get("role") == "user"
        ]

        assert any(
            "我叫小明" in message["content"]
            for message in previous_user_messages
        )

        return make_response(
            SimpleNamespace(
                content="你叫小明。",
                tool_calls=None,
            )
        )

    agent = AgentSession(
        llm_call=fake_llm,
        max_messages=10,
    )

    assert agent.ask("我叫小明") == "你好，小明。"
    assert agent.ask("我叫什么？") == "你叫小明。"


def test_agent_clear_memory():
    def fake_llm(messages, tools):
        return make_response(
            SimpleNamespace(
                content="收到。",
                tool_calls=None,
            )
        )

    agent = AgentSession(
        llm_call=fake_llm,
        max_messages=10,
    )

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

        return make_response(SimpleNamespace(content="收到。", tool_calls=None))

    agent = AgentSession(llm_call=fake_llm, max_messages=10)

    # 必须跑到超过窗口容量（10 条 = 5 轮）才能触发裁剪
    for index in range(20):
        agent.ask(f"第 {index} 个问题")

    assert len(seen) == 20
    assert all(roles[1] == "user" for roles in seen)


def test_agent_memory_stays_paired_after_many_turns():
    """回归测试：多轮之后记忆里不能出现孤儿的 assistant 开头。"""
    def fake_llm(messages, tools):
        return make_response(SimpleNamespace(content="收到。", tool_calls=None))

    agent = AgentSession(llm_call=fake_llm, max_messages=10)

    for index in range(20):
        agent.ask(f"第 {index} 个问题")
        memory = agent.get_memory()
        assert memory[0]["role"] == "user"