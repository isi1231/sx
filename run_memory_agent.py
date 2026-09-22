from app.agent import create_real_agent


def main() -> None:
    agent = create_real_agent(
        max_messages=10,
        max_tool_rounds=3,
    )

    print("第 5 天多轮记忆 Agent 已启动。")
    print("输入 clear 清空记忆。")
    print("输入 memory 查看当前记忆。")
    print("输入 quit 或 exit 退出。")

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("对话结束。")
            break

        if user_input.lower() == "clear":
            agent.clear_memory()
            print("记忆已清空。")
            continue

        if user_input.lower() == "memory":
            memory = agent.get_memory()

            if not memory:
                print("当前没有记忆。")
                continue

            print("\n当前记忆：")
            for index, message in enumerate(memory, start=1):
                role = message.get("role")
                content = message.get("content", "")
                print(f"{index}. [{role}] {content}")

            continue

        if not user_input:
            print("问题不能为空。")
            continue

        try:
            answer = agent.ask(user_input)
            print(f"\nAgent：{answer}")
        except Exception as exc:
            print(f"\nAgent 运行失败：{exc}")


if __name__ == "__main__":
    main()