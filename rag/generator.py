import os
from dotenv import load_dotenv
import google.generativeai as genai
from rag.retriever import retrieve

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """You are an HR assistant. Answer the employee's question using ONLY the context provided below.
If the answer is not present in the context, say you don't have that information.
Do not guess or make up details. Keep answers short and clear, suitable for a WhatsApp message.

Example 1:
Question: How many casual leaves am I entitled to?
Answer: You are entitled to 12 casual leaves per year, as per company policy.

Example 2:
Question: What is the office dress code on weekends?
Answer: I don't have that information in the provided policy documents.

Now answer the real question in the same style, using only the context below."""


def generate_answer(query):
    chunks = retrieve(query)
    context = "\n\n".join(chunks)

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    query = input("Ask a question: ")
    answer = generate_answer(query)
    print("\n" + answer)