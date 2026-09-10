from app.llm_client import stream_chat
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """
你是一个专业、简洁的 AI 助手。
回答问题时要清晰、准确。
如果不确定答案，不要编造事实。
""".strip()


def main():
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("DeepSeek 多轮对话已经启动。")
    print("输入 quit 或 exit 退出。")
    print("输入 clear 清空当前对话。")

    while True:
        question = input("\n你：").strip()

        if not question:
            print("问题不能为空。")
            continue

        if question.lower() in {"quit", "exit"}:
            print("对话结束。")
            break

        if question.lower() == "clear":
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]
            print("对话历史已清空。")
            continue

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        print("DeepSeek：", end="", flush=True)

        answer_parts = []

        for content in stream_chat(messages):
            print(content, end="", flush=True)
            answer_parts.append(content)

        answer = "".join(answer_parts)

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )
        if len(messages) > MAX_HISTORY_MESSAGES + 1:
            messages = [messages[0]] + messages[-MAX_HISTORY_MESSAGES:]
        print()


if __name__ == "__main__":
    main()