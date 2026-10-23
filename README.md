# Dockerized DistilBERT Sentiment Analysis

A university AI infrastructure project demonstrating pretrained deep learning inference with two Docker containers orchestrated by Docker Compose.

The application classifies English text as **POSITIVE** or **NEGATIVE** and displays a confidence score.

## Architecture

Gradio UI → HTTP POST `/predict` → FastAPI → pretrained DistilBERT

- **UI container:** Gradio uses Python Requests to send text to `http://api:8000/predict`.
- **API container:** FastAPI runs DistilBERT inference using PyTorch on CPU.
- **Docker Compose:** Builds and starts both services on a shared network, where `api` identifies the API service.

## Technologies

Python 3.11, FastAPI, Uvicorn, Hugging Face Transformers, PyTorch, Gradio, Requests, Docker, and Docker Compose.

Model: `distilbert-base-uncased-finetuned-sst-2-english`.

## Requirements

- Docker Desktop running, with Docker Compose available.
- Internet access to download base images, Python packages, and model files when they are not cached.

## Start

From the project root:

```bash
docker compose up --build
```

Wait for the API to finish loading the model before submitting predictions. Inference runs on CPU; no GPU is required.

## Access

- Gradio interface: http://localhost:7860
- FastAPI interactive documentation: http://localhost:8000/docs

Enter English text in Gradio and submit it to display the sentiment and confidence score.

## Stop

From the project root, in another terminal if necessary:

```bash
docker compose down
```

This removes the application containers and the default Compose network.
