from app.llm_client import chat


if __name__ == "__main__":
    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空")
    else:
        print("\nDeepSeek 回答：")
        print(chat(question))