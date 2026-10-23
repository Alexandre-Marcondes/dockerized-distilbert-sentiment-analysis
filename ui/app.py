import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://api:8000/predict")


def predict(text):
    try:
        response = requests.post(
            API_URL, json={"text": text}, timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["label"], result["score"]
    except requests.RequestException as exc:
        raise gr.Error("The API request failed. Check the API server and input.") from exc


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="English text", lines=2),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Number(label="Confidence"),
    ],
    title=" AI Sentiment Analysis",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
