import google.generativeai as genai

__all__ = ['request_llm']

genai.configure()

def request_llm(messages: list[dict], model: str) -> tuple[str, dict]:
    # Extract system instructions
    system_instruction = ""
    history = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_instruction += str(msg['content']) + "\n\n"
        elif msg['role'] == 'user':
            parts = msg['content'] if isinstance(msg['content'], list) else [msg['content']]
            history.append({'role': 'user', 'parts': parts})
        elif msg['role'] == 'assistant':
            parts = msg['content'] if isinstance(msg['content'], list) else [msg['content']]
            history.append({'role': 'model', 'parts': parts})

    # Prepare chat model
    generative_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction.strip() if system_instruction else None
    )
    
    # Send the last message
    last_user_message_parts = history.pop()['parts'] if history and history[-1]['role'] == 'user' else []
    chat = generative_model.start_chat(history=history)
    response = chat.send_message(last_user_message_parts)
    
    content = response.text
    usage = {
        'prompt_tokens': response.usage_metadata.prompt_token_count,
        'completion_tokens': response.usage_metadata.candidates_token_count,
        'total_tokens': response.usage_metadata.total_token_count
    } if hasattr(response, 'usage_metadata') else {}
    
    return content, usage
