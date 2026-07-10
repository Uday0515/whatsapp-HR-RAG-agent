import chromadb
from chromadb.utils import embedding_functions
import os

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "hr_policies"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(COLLECTION_NAME, embedding_function=ef)


def retrieve(query, n_results=3):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


if __name__ == "__main__":
    query = input("Ask a question: ")
    chunks = retrieve(query)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n- Chunk {i} -")
        print(chunk)