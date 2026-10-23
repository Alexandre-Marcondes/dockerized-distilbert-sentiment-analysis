from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI()

sentiment_model = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt",
    device=-1,
)

@app.get("/")
def root():
    return {"message": "DistilBERT API server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.post("/predict")
def predict(request: PredictionRequest):
    result = sentiment_model(request.text, truncation=True)[0]
    return {"label": result["label"], "score": result["score"]}
