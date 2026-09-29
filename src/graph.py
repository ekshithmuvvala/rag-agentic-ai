from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


class AgentState(TypedDict):
    question: str
    context: list[str]
    answer: str
    score: float


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


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


retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def retrieve_node(state: AgentState):

    documents = retriever.invoke(
        state["question"]
    )

    context = [
        document.page_content
        for document in documents
    ]

    return {
        "context": context
    }


def generate_node(state: AgentState):

    context_text = "\n\n".join(
        state["context"]
    )

    prompt = f"""
You are a strict document-grounded AI assistant.

Answer the question ONLY using the provided
document context.

Do not use outside knowledge.

If the context does not contain enough information
to answer the question, respond exactly:

I cannot answer based on the provided document.

Context:
{context_text}

Question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    if state["context"]:
        confidence = 0.95
    else:
        confidence = 0.0

    return {
        "answer": response.content,
        "score": confidence
    }


workflow = StateGraph(AgentState)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "generate"
)

workflow.add_edge(
    "generate",
    END
)

graph = workflow.compile()


def build_rag_graph():

    return graph