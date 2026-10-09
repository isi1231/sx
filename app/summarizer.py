"""把被裁掉的旧对话压缩成一段摘要。

分工：
- `memory.py` 决定「什么时候压缩」，是纯逻辑，可脱离 LLM 单测
- 本模块负责「怎么压缩」，是唯一碰网络的一层

摘要器是可注入的，所以测试时可以塞一个假函数，不花一分钱 API 费用。
"""

from typing import Any, Callable


# 提示词里说了「不超过 max_chars 字」，但**那是软约束** —— 模型不保证遵守。
# 代码里再兜一道硬上限，防止「摘要」反而把上下文撑大。
# 留 2 倍余量：既给模型正常发挥的空间，又兜住失控的情况。
HARD_LIMIT_FACTOR = 2

TRUNCATED_MARK = "…（摘要已截断）"


SUMMARY_PROMPT = """你在维护一段多轮对话的长期记忆。

下面有「已有摘要」和「新增对话」两部分。请把两者合并成一段不超过 {max_chars} 字的摘要。

保留这些（按重要性排序）：
1. 用户的身份、偏好、称呼
2. 用户的目标和正在做的事
3. 已经确认的事实、已经完成的操作及其结果
4. 尚未解决的问题

丢弃这些：
- 寒暄、客套
- 重复出现的内容
- 助手自己的思考过程和解释

严格要求：
1. 只输出摘要正文，不要标题、不要前言、不要解释
2. 用第三人称陈述，例如「用户名叫小明」
3. 不要编造对话里没出现过的信息
4. 已有摘要里的信息如果仍然有效，必须保留

已有摘要：
{old_summary}

新增对话：
{new_text}
"""


def enforce_limit(text: str, limit: int) -> str:
    """超长就砍掉尾巴。

    砍尾巴是有讲究的：`SUMMARY_PROMPT` 要求模型**按重要性排序**输出
    （用户身份 → 目标 → 已确认事实 → 未解决问题），
    所以被砍掉的恰好是最不重要的那部分。
    """
    if limit <= 0 or len(text) <= limit:
        return text

    return text[:limit] + TRUNCATED_MARK


def build_summarizer(
    llm_call: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
    max_chars: int = 400,
    hard_limit: int | None = None,
) -> Callable[[str, str], str]:
    """返回一个 (旧摘要, 新增对话) -> 新摘要 的函数。

    传入的 llm_call 与 Agent 用的是同一个签名 `(messages, tools)`，
    这样测试里可以直接复用同一个假模型。

    `hard_limit` 不给时按 `max_chars * HARD_LIMIT_FACTOR` 兜底。
    """
    if hard_limit is None:
        hard_limit = max_chars * HARD_LIMIT_FACTOR

    def summarize(old_summary: str, new_text: str) -> str:
        prompt = SUMMARY_PROMPT.format(
            max_chars=max_chars,
            old_summary=old_summary.strip() or "（还没有摘要）",
            new_text=new_text,
        )

        response = llm_call([{"role": "user", "content": prompt}], [])
        content = getattr(response.choices[0].message, "content", None) or ""

        return enforce_limit(content.strip(), hard_limit)

    return summarize
