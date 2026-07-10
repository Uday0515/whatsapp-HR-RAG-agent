from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from rag.generator import generate_answer

app = FastAPI()


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(question: Question):
    answer = generate_answer(question.query)
    return {"answer": answer}


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    incoming_msg = form.get("Body")
    answer = generate_answer(incoming_msg)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{answer}</Message>
</Response>"""

    return Response(content=twiml, media_type="application/xml")