import os
import gradio as gr
from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
client = cohere.ClientV2(api_key=api_key)


def get_response(message):

    prompt = f"""
You are an empathetic AI mental health support assistant.

User message:
{message}

Respond in a supportive, calm and non-judgmental way.
Do not diagnose the user.
Keep the response concise.
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

    return response.message.content[0].text


with gr.Blocks(title="AI Mental Health Coach") as app:

    gr.Markdown(
        "# 🧠 AI Mental Health Coach"
    )

    gr.Markdown(
        "A supportive AI assistant for emotional well-being."
    )

    chatbot = gr.Chatbot(
        label="Conversation"
    )

    message = gr.Textbox(
        label="Your Message",
        placeholder="How are you feeling today?"
    )

    send = gr.Button("Send")

    send.click(
        get_response,
        inputs=message,
        outputs=chatbot
    )


app.launch()