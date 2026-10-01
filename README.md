# sps_genai: Applied Generative AI, Assignment 1

FastAPI service with two models:

- Bigram text generation (Module 3 class activity): `POST /generate`
- spaCy word embeddings (Assignment 1): `POST /embedding` and `POST /similarity`, using the 300-dimensional `en_core_web_lg` vectors from the Module 2 demo

## Run with Docker
```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
Interactive docs: http://127.0.0.1:8000/docs

## Run locally without Docker
```bash
uv sync
uv run fastapi dev main.py
```

## Example requests
```bash
# Word embedding
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" -d '{"word": "king"}'

# Similarity between two words
curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" -d '{"word1": "king", "word2": "queen"}'

# Bigram text generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" -d '{"start_word": "the", "length": 10}'
```

Words without a vector return 404. Multi-word input returns the average of the word vectors.
