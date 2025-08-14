from openai import OpenAI

__all__ = ['request_llm']

client = OpenAI()


def request_llm(messages: list[dict], model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
