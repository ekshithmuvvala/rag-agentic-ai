import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


def run_ingestion(pdf_path: str):

    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    print("Splitting document...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    print("Connecting to Pinecone...")

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    print("Uploading document chunks to Pinecone...")

    vector_store.add_documents(chunks)

    print("Ingestion completed successfully.")


if __name__ == "__main__":

    pdf_path = os.path.join(
        "data",
        "Ebook-Agentic-AI.pdf"
    )

    run_ingestion(pdf_path)