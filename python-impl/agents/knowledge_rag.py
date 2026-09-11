"""
Knowledge Retrieval Agent — RAG Knowledge Base Q&A

Responsible for retrieving relevant documents from a vector database,
combining context to generate accurate answers.
Implements the complete RAG pipeline: Query Rewrite → Vector Retrieval → Reranking → Context Injection → Answer Generation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.long_term import LongTermMemory
from tracing.otel_config import trace_agent_call


RAG_SYSTEM_PROMPT = """You are a professional knowledge base Q&A Agent, responsible for answering user questions based on retrieved documents.

Answer rules:
1. Strictly answer based on retrieved document content, do not fabricate information
2. If the documents contain no relevant information, clearly inform the user and suggest transferring to a human agent
3. Answers should be concise and professional, suitable for customer service scenarios
4. For financial product information, must include the note "The above information is for reference only, subject to contract terms"
5. Cite the referenced document source at the end of the answer

Answer format:
- First directly answer the user's question
- Supplement with related information if necessary
- Add risk disclaimer for financial scenarios
"""

QUERY_REWRITE_PROMPT = """Please rewrite the user's colloquial question into a query statement more suitable for vector retrieval.
Preserve the core semantics, remove colloquial expressions, and add professional terminology.
Only return the rewritten query, nothing else.

User's original question: {query}
"""


class KnowledgeRAGAgent:
    """Knowledge Retrieval Agent - Implements complete RAG pipeline"""

    def __init__(self, llm: ChatOpenAI, long_term_memory: LongTermMemory | None = None):
        self.llm = llm
        self.long_term_memory = long_term_memory or LongTermMemory()

    @trace_agent_call("rag_query_rewrite")
    async def rewrite_query(self, original_query: str) -> str:
        """Query rewrite: Transform colloquial questions into retrieval-friendly queries"""
        messages = [
            HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=original_query)),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("rag_retrieve")
    async def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant documents from the vector database"""
        docs = self.long_term_memory.search(query, top_k=top_k)
        return docs

    @trace_agent_call("rag_rerank")
    async def rerank_documents(
        self, query: str, documents: list[dict], top_k: int = 3
    ) -> list[dict]:
        """Rerank retrieval results to improve relevance"""
        if not documents:
            return []

        doc_summaries = "\n".join(
            f"[{i}] {doc.get('content', '')[:200]}"
            for i, doc in enumerate(documents)
        )

        messages = [
            SystemMessage(content="You are a document relevance ranking expert."),
            HumanMessage(content=(
                f"User query: {query}\n\n"
                f"Candidate documents:\n{doc_summaries}\n\n"
                f"Please return the indices of the {top_k} most relevant documents, separated by commas, e.g.: 0,2,4"
            )),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            indices = [int(i.strip()) for i in response.content.split(",")]
            reranked = [documents[i] for i in indices if i < len(documents)]
        except (ValueError, IndexError):
            reranked = documents[:top_k]

        return reranked

    @trace_agent_call("rag_generate")
    async def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        """Generate answer based on retrieved documents"""
        if not context_docs:
            return "Sorry, no information related to your question was found in the knowledge base. We suggest you contact a human agent for assistance."

        context = "\n\n---\n\n".join(
            f"Source: {doc.get('source', 'Unknown')}\nContent: {doc.get('content', '')}"
            for doc in context_docs
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"User question: {query}\n\n"
                f"Retrieved reference documents:\n{context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("knowledge_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Complete RAG pipeline (as a Graph node):
        1. Query rewrite
        2. Vector retrieval
        3. Reranking
        4. Answer generation
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        original_query = messages[-1].content

        rewritten_query = await self.rewrite_query(original_query)

        raw_docs = await self.retrieve_documents(rewritten_query, top_k=5)

        reranked_docs = await self.rerank_documents(rewritten_query, raw_docs, top_k=3)

        answer = await self.generate_answer(original_query, reranked_docs)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }
