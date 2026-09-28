import csv
import os
from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
client = cohere.ClientV2(api_key=api_key)

input_file = "output/analyzed_conversations.csv"
output_file = "output/final_responses.csv"

rows = []

with open(input_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:

        user_text = row["user_input"]
        sentiment = row["sentiment_score"]
        emotion = row["emotion"]

        prompt = f"""
You are an empathetic AI mental health support assistant.

User message: {user_text}
Detected emotion: {emotion}
Sentiment score: {sentiment}

Give a short, supportive and non-judgmental response.
Do not diagnose the user.
Give one practical suggestion when appropriate.
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

        ai_response = response.message.content[0].text

        row["ai_response"] = ai_response
        rows.append(row)

with open(output_file, "w", newline="", encoding="utf-8") as file:

    fieldnames = rows[0].keys()

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

print("===== COHERE BATCH PROCESSING COMPLETED =====")
print("Output saved to:", output_file)