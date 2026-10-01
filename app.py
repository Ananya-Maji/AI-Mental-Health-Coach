import os
import csv
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from textblob import TextBlob
import cohere

load_dotenv()

client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load historical data ──────────────────────────────────────────────────────

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

analyzed_data   = load_csv("output/analyzed_conversations.csv")
final_responses = load_csv("output/final_responses.csv")

# ── Emotion detection (mirrors analysis.py) ───────────────────────────────────

def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["stress", "stressed", "pressure"]):  return "Stress"
    if any(w in t for w in ["lonely", "loneliness", "nobody"]):  return "Loneliness"
    if any(w in t for w in ["fail", "failure", "scared"]):       return "Fear of Failure"
    if any(w in t for w in ["sleep", "sleeping", "tired"]):      return "Sleep Problems"
    if any(w in t for w in ["job", "career"]):                   return "Career Anxiety"
    return "General Concern"

def get_examples(emotion, n=2):
    return [r for r in final_responses if r.get("emotion") == emotion][:n]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    message   = request.json.get("message", "")
    sentiment = round(TextBlob(message).sentiment.polarity, 3)
    emotion   = detect_emotion(message)
    examples  = get_examples(emotion)

    example_block = "".join(
        f"User: {ex['user_input']}\nAssistant: {ex['ai_response']}\n\n"
        for ex in examples
    )

    prompt = f"""You are an empathetic AI mental health support assistant.

Here are some example conversations for context:
{example_block}
Now respond to the following:
User message: {message}
Detected emotion: {emotion}
Sentiment score: {sentiment}

Respond in a supportive, calm and non-judgmental way.
Do not diagnose the user. Keep the response concise.
Give one practical suggestion when appropriate."""

    response = client.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({
        "response":  response.message.content[0].text,
        "emotion":   emotion,
        "sentiment": sentiment
    })


@app.route("/sessions")
def sessions():
    """Return list of unique session IDs with their turns for the review panel."""
    sessions_map = {}
    for r in final_responses:
        sid = r["session_id"]
        sessions_map.setdefault(sid, []).append({
            "turn_id":    r["turn_id"],
            "user_input": r["user_input"],
            "bot_response": r.get("bot_response", ""),
            "ai_response":  r.get("ai_response", ""),
            "emotion":      r.get("emotion", ""),
            "sentiment":    r.get("sentiment_score", "0"),
            "length_of_response": r.get("length_of_response", "")
        })
    return jsonify(sessions_map)


@app.route("/review", methods=["POST"])
def review():
    """Save professional empathy rating for a turn."""
    data = request.json
    ratings_file = "output/professional_ratings.csv"
    file_exists  = os.path.exists(ratings_file)

    with open(ratings_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "turn_id", "rating", "notes"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "session_id": data.get("session_id"),
            "turn_id":    data.get("turn_id"),
            "rating":     data.get("rating"),
            "notes":      data.get("notes", "")
        })
    return jsonify({"status": "saved"})


@app.route("/dashboard")
def dashboard():
    """Grafana-style aggregated metrics — no raw user text exposed."""

    # Emotion distribution
    emotion_counts = {}
    sentiment_by_session = {}
    response_lengths = []
    turns_per_session = {}

    for row in analyzed_data:
        em  = row.get("emotion", "Unknown")
        sen = float(row.get("sentiment_score", 0))
        sid = row.get("session_id", "")
        rl  = int(row.get("length_of_response", 0) or 0)

        emotion_counts[em] = emotion_counts.get(em, 0) + 1
        sentiment_by_session.setdefault(sid, []).append(sen)
        response_lengths.append(rl)
        turns_per_session[sid] = turns_per_session.get(sid, 0) + 1

    # Sentiment per session (time-series proxy)
    sentiment_series = [
        {"session": sid, "avg_sentiment": round(sum(v) / len(v), 3)}
        for sid, v in sentiment_by_session.items()
    ]

    avg_turns = round(sum(turns_per_session.values()) / len(turns_per_session), 2) if turns_per_session else 0
    avg_resp_len = round(sum(response_lengths) / len(response_lengths), 1) if response_lengths else 0
    overall_sentiment = round(
        sum(float(r.get("sentiment_score", 0)) for r in analyzed_data) / len(analyzed_data), 3
    ) if analyzed_data else 0

    # Avg sentiment per emotion
    sent_by_emotion = {}
    for row in analyzed_data:
        em  = row.get("emotion", "Unknown")
        sen = float(row.get("sentiment_score", 0))
        sent_by_emotion.setdefault(em, []).append(sen)
    avg_sentiment_by_emotion = {
        em: round(sum(v) / len(v), 3) for em, v in sent_by_emotion.items()
    }

    return jsonify({
        "kpi": {
            "total_sessions":    len(turns_per_session),
            "total_turns":       len(analyzed_data),
            "avg_turns_session": avg_turns,
            "avg_response_len":  avg_resp_len,
            "overall_sentiment": overall_sentiment
        },
        "emotion_distribution":    emotion_counts,
        "sentiment_series":        sentiment_series,
        "avg_sentiment_by_emotion": avg_sentiment_by_emotion
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
