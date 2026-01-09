from openai import OpenAI

__all__ = ['request_llm']

client = OpenAI()


def request_llm(messages: list[dict], model: str) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content, response.usage.model_dump()
