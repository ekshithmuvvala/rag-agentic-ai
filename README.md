**# Agentic AI RAG Chatbot**



**A document-grounded Retrieval-Augmented Generation chatbot built using Python, LangChain, LangGraph, Pinecone, OpenAI embeddings and FastAPI.**



**## Objective**



**The objective of this project is to build a RAG-based AI chatbot that answers questions using only information retrieved from the provided Agentic AI eBook.**



**## Architecture**



**```text**

**Agentic AI PDF**

&#x20;     **|**

&#x20;     **v**

**PyPDFLoader**

&#x20;     **|**

&#x20;     **v**

**Text Chunking**

&#x20;     **|**

&#x20;     **v**

**OpenAI Embeddings**

&#x20;     **|**

&#x20;     **v**

**Pinecone Vector Database**

&#x20;     **|**

&#x20;     **v**

**User Query**

&#x20;     **|**

&#x20;     **v**

**LangGraph**

&#x20;     **|**

&#x20;     **+----------+**

&#x20;     **|          |**

&#x20;     **v          v**

&#x20;  **Retrieve   Generate**

&#x20;     **|          |**

&#x20;     **+-----+----+**

&#x20;           **|**

&#x20;           **v**

&#x20;        **FastAPI**

&#x20;           **|**

&#x20;           **v**

**Answer + Context + Confidence**

