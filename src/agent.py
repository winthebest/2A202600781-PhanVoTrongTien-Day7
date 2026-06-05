from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> str:
        if metadata_filter:
            retrieved = self.store.search_with_filter(
                question, top_k=top_k, metadata_filter=metadata_filter
            )
        else:
            retrieved = self.store.search(question, top_k=top_k)
        context_blocks = [item.get("content", "") for item in retrieved if item.get("content")]
        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        prompt = (
            "You are a helpful assistant. Answer the user's question using only the context below.\n"
            "If the answer is not in the context, say you do not have enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
