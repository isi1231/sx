from app.agent import run_agent


def main() -> None:
    print("第 4 天 Agent 工具调用实验已启动。")
    print("可尝试：25 * 4 + 10 等于多少？")
    print("输入 quit 或 exit 退出。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("对话结束。")
            return
        if not user_input:
            print("问题不能为空。")
            continue

        try:
            print(f"\nAgent：{run_agent(user_input)}")
        except Exception as exc:
            print(f"\nAgent 运行失败：{exc}")


if __name__ == "__main__":
    main()
