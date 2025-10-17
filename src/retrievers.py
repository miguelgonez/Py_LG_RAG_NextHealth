"""Retriever utilities for the RAG application."""

from __future__ import annotations

import os
from typing import List, Tuple

from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure tokenizers for deterministic behaviour
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGRetriever:
    """Semantic retriever with optional reranking."""

    def __init__(
        self,
        persist_dir: str = "./db/chroma",
        collection_name: str = "rag_collection",
        top_k: int = 10,
        use_reranker: bool = True,
        reranker_top_n: int = 5,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker_top_n = reranker_top_n

        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.retriever = self._create_retriever()

    @staticmethod
    def _load_embeddings() -> HuggingFaceEmbeddings:
        """Load the embedding model used for retrieval."""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ Modelo de embeddings cargado")
        return embeddings

    def _load_vectorstore(self) -> Chroma:
        """Load an existing Chroma vector store."""
        if not os.path.exists(self.persist_dir):
            raise ValueError(f"Vector store no encontrado en {self.persist_dir}")

        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        print("✅ Vector store cargado exitosamente")
        return vectorstore

    def _create_retriever(self):  # noqa: ANN201 - dynamic type depending on reranker usage
        """Create the retriever, wrapping with Flashrank if requested."""
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        if not self.use_reranker:
            return base_retriever

        try:
            compressor = FlashrankRerank(
                top_n=self.reranker_top_n,
                model="ms-marco-MiniLM-L-12-v2",
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
            print(f"✅ Flashrank Reranker cargado (top_n={self.reranker_top_n})")
            return retriever
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Error cargando reranker: {exc}")
            print("   Usando retriever sin reranking")
            return base_retriever

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        return self.retriever.get_relevant_documents(query)

    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        """Run a plain similarity search against the vector store."""
        k = k or self.top_k
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int | None = None) -> List[Tuple[Document, float]]:
        """Return documents with similarity scores."""
        k = k or self.top_k
        return self.vectorstore.similarity_search_with_score(query, k=k)


class HybridRetriever(RAGRetriever):
    """Retriever that mixes semantic and keyword search."""

    def __init__(self, *args, alpha: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """Placeholder keyword search (falls back to semantic search)."""
        return self.vectorstore.similarity_search(query, k=k)

    def hybrid_search(self, query: str, k: int | None = None) -> List[Document]:
        """Combine semantic retrieval with keyword-style retrieval."""
        k = k or self.top_k
        semantic_docs = self.vectorstore.similarity_search(query, k=k)
        keyword_docs = self._keyword_search(query, k=k)

        seen_ids = set()
        combined_docs: List[Document] = []
        for doc in semantic_docs + keyword_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_docs.append(doc)

        return combined_docs[:k]


def create_retriever(
    persist_dir: str = "./db/chroma",
    use_reranker: bool = True,
    reranker_top_n: int = 5,
    top_k: int = 10,
    hybrid: bool = False,
) -> RAGRetriever:
    """Factory helper for creating retrievers."""
    if hybrid:
        return HybridRetriever(
            persist_dir=persist_dir,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_top_n=reranker_top_n,
        )
    return RAGRetriever(
        persist_dir=persist_dir,
        top_k=top_k,
        use_reranker=use_reranker,
        reranker_top_n=reranker_top_n,
    )


if __name__ == "__main__":
    retriever = create_retriever(
        persist_dir="./db/chroma",
        use_reranker=True,
        reranker_top_n=5,
        top_k=10,
    )

    query = "¿Qué es la medicina del estilo de vida?"
    docs = retriever.retrieve(query)

    print(f"\n🔍 Query: {query}")
    print(f"📊 Documentos recuperados: {len(docs)}\n")
    for idx, doc in enumerate(docs, start=1):
        print(f"{idx}. {doc.metadata.get('file_name', 'Unknown')}")
        print(f"   {doc.page_content[:200]}...\n")


def build_vectorstore(persist_dir: str, collection_name: str = "rag_collection") -> Chroma:
    """Convenience helper that returns a loaded Chroma vector store."""
    retriever = RAGRetriever(
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_k=1,
        use_reranker=False,
    )
    return retriever.vectorstore

def build_advanced_rag_retriever(persist_dir: str, mode: str | None = None, **kwargs) -> RAGRetriever:
    """Backward-compatible helper that falls back to the base retriever."""
    _ = mode  # Mode handling is collapsed into the simplified retriever.
    return create_retriever(persist_dir=persist_dir, **kwargs)

