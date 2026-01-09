# from openai import OpenAI

# __all__ = ['request_llm']

# client = OpenAI()


# def request_llm(messages: list[dict], model: str) -> tuple[str, dict]:
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         reasoning_effort="high"
#     )
#     return response.choices[0].message.content, response.usage.model_dump()

import os
from google import genai
from google.genai import types

__all__ = ['request_llm']

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key is not None, "GOOGLE_API_KEY environment variable not set"

def request_llm(messages: list[dict], model: str) -> tuple[str, dict]:
	client = genai.Client(api_key=api_key)
	
	# Build history from messages (excluding system messages and the last user message)
	history = []
	for msg in messages[:-1]:  # Exclude last message, it will be sent separately
		if msg["role"] == "system":
			continue
		role = "user" if msg["role"] == "user" else "model"
		history.append(types.Content(
			role=role,
			parts=[types.Part(text=msg["content"])]
		))
	
	# Get system instructions
	system_instructions = [
		msg["content"] for msg in messages if msg["role"] == "system"
	]
	
	chat = client.chats.create(
		model=model,
		history=history,
		config=types.GenerateContentConfig(
			system_instruction=system_instructions if system_instructions else None
		)
	)
	
	# Send the last user message
	last_message = messages[-1]["content"] if messages else ""
	response = chat.send_message(last_message)
	
	return response.text, response.usage_metadata.model_dump() if response.usage_metadata else {}