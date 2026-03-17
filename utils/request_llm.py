from google import genai
from google.genai import types

__all__ = ['request_llm']

def request_llm(messages: list[dict], model: str) -> tuple[str, dict]:
    # Extract system instructions
    system_instruction = ""
    history = []

    for msg in messages:
        if msg['role'] == 'system':
            system_instruction += str(msg['content']) + "\n\n"
        elif msg['role'] in ['user', 'assistant', 'model']:
            role = 'model' if msg['role'] == 'assistant' else msg['role']
            content_val = msg['content']
            if isinstance(content_val, list):
                parts = []
                for p in content_val:
                    if isinstance(p, str):
                        parts.append(types.Part.from_text(text=p))
                    elif isinstance(p, dict) and 'data' in p:
                        parts.append(types.Part.from_bytes(data=p['data'], mime_type=p['mime_type']))
                history.append(types.Content(role=role, parts=parts))
            else:
                history.append(types.Content(role=role, parts=[types.Part.from_text(text=str(content_val))]))

    client = genai.Client()
    
    # Prepare chat configuration
    config = types.GenerateContentConfig(
        system_instruction=system_instruction.strip() if system_instruction else None
    )

    # Pop the last user message to send it
    last_user_message = history.pop() if history and history[-1].role == 'user' else None
    
    # Create the chat session
    chat = client.chats.create(model=model, config=config, history=history)
    
    # Ensure there's a last user message to send
    if last_user_message:
        response = chat.send_message(last_user_message.parts)
    else:
        # If no user message to send, just an empty text block (this shouldn't happen based on original logic)
        response = chat.send_message("")
    content = response.text
    usage = {
        'prompt_tokens': response.usage_metadata.prompt_token_count,
        'completion_tokens': response.usage_metadata.candidates_token_count,
        'total_tokens': response.usage_metadata.total_token_count
    } if hasattr(response, 'usage_metadata') else {}
    
    return content, usage
