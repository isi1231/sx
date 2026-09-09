import time

from openai import OpenAI

from app.config import  DEEPSEEK_API_KEY, DEEPSEEK_MODEL


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def chat(user_message: str) -> str:
    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业、简洁的 AI 助手。",
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=0.2,
            timeout=30,
        )

        answer = response.choices[0].message.content or ""

        elapsed = time.perf_counter() - start_time
        print(f"[DeepSeek] elapsed={elapsed:.2f}s")

        return answer

    except Exception as exc:
        print(f"[DeepSeek ERROR] {exc}")
        return "模型服务暂时不可用，请稍后重试。"