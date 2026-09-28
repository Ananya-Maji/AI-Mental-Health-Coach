import os
from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    print("ERROR: COHERE_API_KEY not found in .env")
    exit()

client = cohere.ClientV2(api_key=api_key)

user_text = "I am very stressed about my exams"
sentiment = -0.125
emotion = "Stress"

prompt = f"""
You are an empathetic AI mental health support assistant.

User message: {user_text}
Detected emotion: {emotion}
Sentiment score: {sentiment}

Give a short, supportive and non-judgmental response.
Do not diagnose the user.
Give a practical suggestion when appropriate.
"""

response = client.chat(
    model="command-a-03-2025",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n===== COHERE AI RESPONSE =====")
print(response.message.content[0].text)