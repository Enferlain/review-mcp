import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ZHIPU_API_KEY")
base_url = "https://api.z.ai/api/coding/paas/v4"
client = OpenAI(api_key=api_key, base_url=base_url)

print(f"Testing 200k Payload on: {base_url}")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_uncommitted_changes",
            "description": "Get git diff output for uncommitted changes.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "default": "staged"}},
            },
        },
    }
]

# Generate a very large block of text
large_text = "Data block for payload testing. " * 6000  # ~186k chars

user_message = f"""Please review this code.
        
CONTEXT:
{large_text}

I have staged changes. PLEASE USE get_uncommitted_changes(target='staged') TO SEE THEM."""

print(f"Payload size (User Message): {len(user_message)} chars")

try:
    response = client.chat.completions.create(
        model="GLM-5.1",
        messages=[
            {
                "role": "system",
                "content": "You are a senior reviewer. Always use tools.",
            },
            {"role": "user", "content": user_message},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )
    print("SUCCESS!")
    print(f"Response: {response.choices[0].message}")
except Exception as e:
    print(f"FAILED: {e}")
