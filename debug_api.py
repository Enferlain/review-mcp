import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ZHIPU_API_KEY")
base_url = os.getenv("ZHIPU_BASE_URL", "https://api.z.ai/api/coding/paas/v4")

print(f"Testing API...")
print(f"Base URL: {base_url}")
print("Model: glm-5.2")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

try:
    response = client.chat.completions.create(
        model="glm-5.2",
        messages=[{"role": "user", "content": "Hello, are you still functional?"}],
        temperature=0.2,
    )
    print("SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"FAILED: {e}")
