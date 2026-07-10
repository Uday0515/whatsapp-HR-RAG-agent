import os
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "hr_policy.pdf")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "hr_policies"


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def build_index():
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"Indexed {collection.count()} chunks into '{CHROMA_DIR}'")


if __name__ == "__main__":
    build_index()