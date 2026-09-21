"""Local tools that the Agent is allowed to call."""

import ast
import math
import operator
from pathlib import Path
from typing import Any


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
    """Safely calculate a small arithmetic expression without eval()."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression 必须是非空字符串")
    if len(expression) > 100:
        raise ValueError("表达式过长")

    tree = ast.parse(expression, mode="eval")
    result = _evaluate_number(tree.body)
    if not math.isfinite(float(result)):
        raise ValueError("计算结果不是有限数字")
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


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a registered tool; unknown tools are never executed."""
    if name not in TOOLS:
        return f"工具拒绝执行：未知工具 {name}"
    if not isinstance(arguments, dict):
        return "工具拒绝执行：参数必须是 JSON 对象"

    try:
        return TOOLS[name](**arguments)
    except TypeError as exc:
        return f"工具参数错误：{exc}"
    except Exception as exc:
        return f"工具执行失败：{exc}"
