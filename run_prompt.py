from app.llm_client import chat
from app.prompting import (
    build_chat_messages,
    build_json_messages,
    build_task_messages,
    parse_json_response,
)


def print_json_result(result: dict) -> None:
    print(f"\n意图：{result['intent']}")
    print(f"回答：{result['answer']}")
    print("步骤：")
    for index, step in enumerate(result["steps"], start=1):
        print(f"{index}. {step}")
    if result["uncertainties"]:
        print("不确定点：")
        for item in result["uncertainties"]:
            print(f"- {item}")


def main() -> None:
    history: list[dict[str, str]] = []

    print("第三天 Prompt 工程实验程序已启动。")
    print("普通输入：System Prompt + Few-shot")
    print("输入 /plan：任务拆解")
    print("输入 /json：结构化 JSON 输出")
    print("输入 clear 清空上下文，输入 quit 或 exit 退出。")

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            print("问题不能为空。")
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("对话结束。")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("对话历史已清空。")
            continue

        if user_input.startswith("/plan "):
            messages = build_task_messages(user_input[6:].strip())
            print("\n模型：")
            print(chat(messages))
            continue

        if user_input.startswith("/json "):
            messages = build_json_messages(user_input[6:].strip())
            raw_answer = chat(messages)
            try:
                print_json_result(parse_json_response(raw_answer))
            except (ValueError, TypeError) as exc:
                print(f"\nJSON 解析失败：{exc}")
                print(f"模型原始输出：{raw_answer}")
            continue

        messages = build_chat_messages(user_input, history)
        answer = chat(messages)
        print(f"\n模型：{answer}")
        history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]
        )


if __name__ == "__main__":
    main()
