"""Local tools that the Agent is allowed to call."""

import ast
import math
import operator
from pathlib import Path
from typing import Any, NamedTuple


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "notes.txt"


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_number(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("不允许布尔值")
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_number(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_number(node.left)
        right = _evaluate_number(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("幂运算指数不能超过 8")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    raise ValueError("只允许数字、括号和 + - * / % ** 运算")


def calculator(expression: str) -> str:
    """Safely calculate a small arithmetic expression without eval().

    异常契约（Day 8 收口）：**只抛 `ValueError`**。

    以前这里会把 Python 内部的异常原样漏出去：

    | 输入 | 泄漏的异常 | 用户看到 |
    | --- | --- | --- |
    | `1/0` | `ZeroDivisionError` | 工具执行失败：division by zero |
    | `"9"*30 + "**8"` | `OverflowError` | 工具执行失败：(34, 'Result too large') |
    | `1 +` | `SyntaxError` | 工具执行失败：invalid syntax |

    三条都是「这个表达式不能用」，属于**工具拒绝执行**，不是工具坏了。
    现在统一转成带中文说明的 `ValueError`，让 `execute_tool` 能按类型分流。
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression 必须是非空字符串")
    if len(expression) > 100:
        raise ValueError("表达式过长")

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        raise ValueError("表达式语法错误") from None

    try:
        result = _evaluate_number(tree.body)
        if not math.isfinite(float(result)):
            raise ValueError("计算结果不是有限数字")
    except ZeroDivisionError:
        raise ValueError("除数不能为 0") from None
    except OverflowError:
        raise ValueError("计算结果超出可表示范围") from None

    return str(result)


def search_notes(query: str) -> str:
    """Search only the bundled notes file and return matching lines."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not DATA_FILE.exists():
        return "知识库为空。"

    query_text = query.strip().casefold()
    lines = DATA_FILE.read_text(encoding="utf-8").splitlines()
    matches = [line for line in lines if query_text in line.casefold()]
    if not matches:
        return f"没有找到与“{query.strip()}”相关的笔记。"
    return "\n".join(matches[:5])


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算简单的数学表达式，只用于四则运算等安全计算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "例如：25 * 4 + 10",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "搜索项目 data/notes.txt 中的学习笔记。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


TOOLS = {
    "calculator": calculator,
    "search_notes": search_notes,
}


class ToolResult(NamedTuple):
    """工具执行结果：**文本给模型看，`ok` 给程序看**。

    为什么非要拆开（Day 8 实测才发现的问题）
    ----------------------------------------
    以前 `execute_tool` 只返回一个字符串，失败信息编码在文本里：

        "工具拒绝执行：除数不能为 0"

    而调用方判断成败只能靠「有没有抛异常」。可 `execute_tool` 的契约恰恰是
    **永不抛异常** —— 于是 `calculator("1/0")` 这条路径上，
    trace 里 `ok=True`、回放里写着 `ok`、合计里「工具失败 0 次」。

    真跑一遍就露馅了：明明被拒绝了，统计却一片祥和。
    根因就是**把布尔信息编码进了人类可读的字符串**。

    所以现在结构化：`text` 进 messages，`ok` / `reason` 进 trace。

    `reason` 的取值：`unknown_tool` / `bad_arguments` / `rejected` / `crashed`
    （再加上调用方自己判定的 `bad_json`）。
    """

    text: str
    ok: bool = True
    reason: str | None = None


def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute a registered tool; unknown tools are never executed.

    失败分三档，因为「模型该怎么纠」完全不同：

    | 档位 | 触发 | `reason` | 文本前缀 | 模型该做的 |
    | --- | --- | --- | --- | --- |
    | 参数错误 | `TypeError`（字段名错、多传少传） | `bad_arguments` | 工具参数错误 | 改参数名 / 补齐字段 |
    | 拒绝执行 | `ValueError`（表达式非法、除零） | `rejected` | 工具拒绝执行 | 换个表达式 / 换个问法 |
    | 执行失败 | 其他 `Exception` | `crashed` | 工具执行失败 | 大概率纠不了，属于 bug |

    **契约：本函数永不抛异常。** 任何失败都变成 `ToolResult(text=..., ok=False)`。
    """
    if name not in TOOLS:
        return ToolResult(f"工具拒绝执行：未知工具 {name}", False, "unknown_tool")
    if not isinstance(arguments, dict):
        return ToolResult("工具拒绝执行：参数必须是 JSON 对象", False, "bad_arguments")

    try:
        return ToolResult(TOOLS[name](**arguments))
    except TypeError as exc:
        return ToolResult(f"工具参数错误：{exc}", False, "bad_arguments")
    except ValueError as exc:
        return ToolResult(f"工具拒绝执行：{exc}", False, "rejected")
    except Exception as exc:
        return ToolResult(f"工具执行失败：{exc}", False, "crashed")
