import os
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional

from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...config import settings
from ...helpers.AgentTools.ollama_client import OllamaClient


def _configure_hf_hub_token() -> str | None:
    token = (settings.HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        return None
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return token


class RAGAgent(BaseAgent):
    def __init__(
        self,
        embeddings_factory: Optional[Callable[[], Any]] = None,
        vector_store_factory: Optional[Callable[[Any, str], Any]] = None,
        ollama_client: OllamaClient | None = None,
    ):
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="RAGAgent",
            ollama_client=ollama_client,
        )
        base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.docs_path = base_path / "docs"
        self.chroma_path = self.docs_path / ".chroma"
        self.docs_path.mkdir(exist_ok=True)

        self.embeddings = (
            embeddings_factory() if embeddings_factory else self._build_embeddings()
        )
        self.vector_store = (
            vector_store_factory(self.embeddings, str(self.chroma_path))
            if vector_store_factory
            else self._build_vector_store(self.embeddings, str(self.chroma_path))
        )

    @staticmethod
    def _build_embeddings() -> Any:
        from langchain_huggingface import HuggingFaceEmbeddings

        hf_token = _configure_hf_hub_token()
        model_kwargs: dict = {}
        if hf_token:
            model_kwargs["token"] = hf_token
        return HuggingFaceEmbeddings(
            model_name=settings.RAG_EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
        )

    @staticmethod
    def _build_vector_store(
        embeddings: Any,
        persist_directory: str,
    ) -> Any:
        from langchain_chroma import Chroma

        return Chroma(
            collection_name="neurina_docs",
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _collection_count(self) -> int:
        try:
            return int(self.vector_store._collection.count())
        except Exception:
            return 0

    def _sync_documents(self):
        """Read documents, chunk them, and add them to Chroma if empty."""
        try:
            if not self.docs_path.exists():
                return

            if self._collection_count() > 0:
                return

            documents = []
            for file_path in self.docs_path.glob("*.*"):
                if file_path.name.startswith("."):
                    continue

                if file_path.suffix in [".txt", ".md", ".csv", ".json"]:
                    from langchain_community.document_loaders import TextLoader

                    loader = TextLoader(str(file_path), encoding="utf-8")
                    documents.extend(loader.load())

            if documents:
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_documents(documents)
                self.vector_store.add_documents(chunks)

                self.logger.log_step("RAG_DOCS_SYNCED", {"chunks_added": len(chunks)}, level="INFO")

        except Exception as e:
            self.logger.log_step("RAG_DOC_SYNC_ERROR", {"error": str(e)}, level="ERROR")

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "")

        self._sync_documents()

        try:
            if self._collection_count() == 0:
                context = ""
            else:
                docs = self.vector_store.similarity_search(user_input, k=3)
                context = "\n\n".join(
                    [f"--- Context Segment ---\n{doc.page_content}" for doc in docs]
                )
        except Exception as e:
            self.logger.log_step("RAG_SEARCH_ERROR", {"error": str(e)}, level="ERROR")
            context = ""

        prompt = f"""You are a helpful AI assistant for the Neurina app. 
The user asked a general question: "{user_input}"

Here is the relevant documentation context:
{context if context else "No documentation available."}

Instructions:
1. Answer the user's question ONLY using the provided documentation context.
2. If the answer is not in the documentation, or the documentation is empty, simply reply with "I don't know" or "لا أعرف" (in the same language as the user's question).
3. Be concise and polite.
4. Do not invent information.

Answer:"""

        try:
            response = self.query_llm(prompt)
            state["rag_response"] = response.strip()
            state["status"] = "ANSWERED_QUESTION"

            self.logger.log_tool_call(
                "RAG_Answer",
                {"question": user_input, "context_found": bool(context)},
                output_summary={"answer": state["rag_response"]},
            )
        except Exception as e:
            state["rag_response"] = "Sorry, I encountered an error while trying to answer your question."
            self.logger.log_step("RAG_FAILED", {"error": str(e)}, level="ERROR")

        return state


class LazyRAGAgentProvider:
    """Thread-safe async lazy provider for the heavy RAG stack."""

    def __init__(self, agent_factory: Optional[Callable[[], RAGAgent]] = None):
        self._agent_factory = agent_factory or RAGAgent
        self._agent: Optional[RAGAgent] = None
        self._lock = asyncio.Lock()

    async def get(self) -> RAGAgent:
        if self._agent is not None:
            return self._agent

        async with self._lock:
            if self._agent is None:
                self._agent = await asyncio.to_thread(self._agent_factory)
            return self._agent
