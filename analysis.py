import csv
from textblob import TextBlob

input_file = "output/processed_conversations/part-00000-0c0c7209-398d-433e-917f-dd0d631c3b2f-c000.csv"
output_file = "output/analyzed_conversations.csv"

def detect_emotion(text):
    text = text.lower()

    if any(word in text for word in ["stress", "stressed", "pressure"]):
        return "Stress"

    elif any(word in text for word in ["lonely", "loneliness", "nobody"]):
        return "Loneliness"

    elif any(word in text for word in ["fail", "failure", "scared"]):
        return "Fear of Failure"

    elif any(word in text for word in ["sleep", "sleeping", "tired"]):
        return "Sleep Problems"

    elif any(word in text for word in ["job", "career"]):
        return "Career Anxiety"

    else:
        return "General Concern"


rows = []

with open(input_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        user_text = row["user_input"]

        sentiment = TextBlob(user_text).sentiment.polarity
        emotion = detect_emotion(user_text)

        row["sentiment_score"] = round(sentiment, 3)
        row["emotion"] = emotion

        rows.append(row)


with open(output_file, "w", newline="", encoding="utf-8") as file:
    fieldnames = rows[0].keys()

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)


print("===== SENTIMENT AND EMOTION ANALYSIS =====")

for row in rows:
    print(
        row["user_input"],
        "| Sentiment:",
        row["sentiment_score"],
        "| Emotion:",
        row["emotion"]
    )

print("===== ANALYSIS COMPLETED SUCCESSFULLY =====")
print("Output saved to:", output_file)