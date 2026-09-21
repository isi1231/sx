from app.tools import calculator, execute_tool, search_notes


def test_calculator_returns_result():
    assert calculator("25 * 4 + 10") == "110"


def test_calculator_rejects_code():
    result = execute_tool("calculator", {"expression": "__import__('os').getcwd()"})
    assert "工具执行失败" in result or "只允许" in result


def test_search_notes_finds_agent():
    assert "Tool Calling" in search_notes("Tool Calling")


def test_unknown_tool_is_rejected():
    assert "未知工具" in execute_tool("delete_file", {"path": ".env"})
