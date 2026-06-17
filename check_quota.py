import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ZHIPU_API_KEY")
base_urls = ["https://api.z.ai/api/coding/paas/v4", "https://api.z.ai/api/paas/v4"]

for base_url in base_urls:
    print(f"\n--- Testing Base URL: {base_url} ---")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Test 1: Simple Prompt
    try:
        response = client.chat.completions.create(
            model="GLM-5.1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(f"  SUCCESS (Simple): {response.choices[0].message.content}")
    except Exception as e:
        print(f"  FAILED (Simple): {e}")

    # Test 2: Prompt with Tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model="GLM-5.1",
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=tools,
            temperature=0.2,
        )
        print(f"  SUCCESS (Tools): {response.choices[0].message}")
    except Exception as e:
        print(f"  FAILED (Tools): {e}")
