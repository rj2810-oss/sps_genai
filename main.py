from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.word_embeddings import WordEmbeddingModel

app = FastAPI(title="SPS GenAI API")

# Sample corpus for the bigram model (Module 3 class activity)
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = WordEmbeddingModel()  # loaded once at startup


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["king"])


class EmbeddingResponse(BaseModel):
    word: str
    dimension: int
    embedding: list[float]


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, examples=["king"])
    word2: str = Field(..., min_length=1, examples=["queen"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    vector = embedding_model.get_embedding(request.word)
    if vector is None:
        raise HTTPException(status_code=404, detail=f"No embedding found for '{request.word}'")
    return EmbeddingResponse(word=request.word, dimension=len(vector), embedding=vector)


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = embedding_model.similarity(request.word1, request.word2)
    if score is None:
        raise HTTPException(status_code=404, detail="One or both words have no embedding")
    return {"word1": request.word1, "word2": request.word2, "similarity": score}
