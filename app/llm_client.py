import time

from openai import OpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def chat_completion(messages, tools=None):
    """Return the raw OpenAI-compatible completion for Agent tool calling."""
    start_time = time.perf_counter()
    request = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "timeout": 30,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    response = client.chat.completions.create(**request)
    elapsed = time.perf_counter() - start_time
    print(f"[DeepSeek] elapsed={elapsed:.2f}s")
    return response


def chat(messages: list[dict[str, str]]) -> str:
    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
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


def stream_chat(messages: list[dict[str, str]]):
    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.2,
            timeout=30,
            stream=True,
        )

        for chunk in response:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content or ""
            if content:
                yield content

        elapsed = time.perf_counter() - start_time
        print(f"\n[DeepSeek] elapsed={elapsed:.2f}s")

    except Exception as exc:
        print(f"\n[DeepSeek ERROR] {exc}")
        yield "模型服务暂时不可用，请稍后重试。"
