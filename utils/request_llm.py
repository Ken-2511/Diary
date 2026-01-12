from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

__all__ = ['request_llm']

client = OpenAI()


def request_llm(messages: list[ChatCompletionMessageParam], model: str) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    content = response.choices[0].message.content
    usage = response.usage
    
    if content is None:
        content = ""
    if usage is None:
        usage_dict = {}
    else:
        usage_dict = usage.model_dump()
    
    return content, usage_dict
