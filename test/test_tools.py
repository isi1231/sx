import pytest

from app.tools import TOOLS, ToolResult, calculator, execute_tool, search_notes


def test_calculator_returns_result():
    assert calculator("25 * 4 + 10") == "110"


def test_calculator_rejects_code():
    result = execute_tool("calculator", {"expression": "__import__('os').getcwd()"})
    assert result.ok is False
    assert "只允许" in result.text


def test_search_notes_finds_agent():
    assert "Tool Calling" in search_notes("Tool Calling")


def test_unknown_tool_is_rejected():
    result = execute_tool("delete_file", {"path": ".env"})

    assert result.ok is False
    assert "未知工具" in result.text


# ============ Day 8：异常契约收口 ============


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1/0", "除数不能为 0"),
        ("10 % 0", "除数不能为 0"),
        ("1 +", "表达式语法错误"),
        ("(1 + 2", "表达式语法错误"),
        # 浮点溢出：Python 返回 inf，由 isfinite 拦下
        ("1e300 * 1e300", "不是有限数字"),
        # 大整数转 float 溢出：Python 抛 OverflowError，由 except 拦下
        ("((10**8)**8)**8", "超出可表示范围"),
    ],
)
def test_calculator_only_raises_value_error(expression, expected):
    """契约：calculator 只抛 ValueError，Python 内部异常不许漏出去。"""
    with pytest.raises(ValueError) as info:
        calculator(expression)

    assert expected in str(info.value)
    assert not isinstance(info.value, (ZeroDivisionError, OverflowError, SyntaxError))


def test_calculator_zero_division_reaches_the_model_as_a_sentence():
    """以前这里漏的是 `division by zero`，模型看不懂，也没法据此纠正。"""
    result = execute_tool("calculator", {"expression": "1/0"})

    assert result.text == "工具拒绝执行：除数不能为 0"
    assert result.ok is False
    assert result.reason == "rejected"


def test_execute_tool_separates_the_three_failure_classes():
    wrong_field = execute_tool("calculator", {"expr": "1 + 1"})       # TypeError
    rejected = execute_tool("calculator", {"expression": "1 + 1 +"})  # ValueError
    unknown = execute_tool("delete_file", {})                        # 白名单外

    assert (wrong_field.ok, wrong_field.reason) == (False, "bad_arguments")
    assert wrong_field.text.startswith("工具参数错误：")

    assert (rejected.ok, rejected.reason) == (False, "rejected")
    assert rejected.text.startswith("工具拒绝执行：")

    assert (unknown.ok, unknown.reason) == (False, "unknown_tool")


def test_execute_tool_reports_success_structurally():
    """成功路径：ok=True，reason 为空 —— 下游不需要去读 text 才知道成功了。"""
    result = execute_tool("calculator", {"expression": "25 * 4 + 10"})

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.reason is None
    assert result.text == "110"


def test_execute_tool_never_raises(monkeypatch):
    """契约：execute_tool 永不抛异常，任何失败都变成 ok=False 的结果。"""
    def boom(**kwargs):
        raise KeyError("意料之外的炸法")

    monkeypatch.setitem(TOOLS, "boom", boom)

    result = execute_tool("boom", {})

    assert result.ok is False
    assert result.reason == "crashed"
    assert result.text.startswith("工具执行失败：")
    assert "意料之外的炸法" in result.text
