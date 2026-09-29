from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_rag_graph


app = FastAPI(
    title="Agentic AI RAG API",
    description="Document-grounded RAG chatbot using LangGraph and Pinecone"
)


graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


@app.get("/")
def root():

    return {
        "message": "Agentic AI RAG API is running"
    }


@app.post(
    "/chat",
    response_model=QueryResponse
)
def chat_endpoint(request: QueryRequest):

    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"]
    )