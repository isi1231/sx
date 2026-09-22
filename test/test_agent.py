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