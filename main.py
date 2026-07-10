from fastapi import FastAPI
from pydantic import BaseModel
from rag.generator import generate_answer

app = FastAPI()


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(question: Question):
    answer = generate_answer(question.query)
    return {"answer": answer}