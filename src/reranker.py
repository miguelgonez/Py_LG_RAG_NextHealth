"""
BGE Re-ranker implementation for improving retrieval quality
"""

import os
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.schema import Document


class BGEReranker:
    """
    BGE (BAAI General Embedding) Reranker for cross-encoder re-ranking
    Improves relevance of retrieved documents
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize BGE reranker
        
        Args:
            model_name: Name of the BGE reranker model
        """
        if model_name is None:
            model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"BGE Reranker loaded: {model_name} on {self.device}")
            
        except Exception as e:
            print(f"Error loading BGE reranker {model_name}: {e}")
            print("Falling back to simple similarity-based reranking")
            self.model = None
            self.tokenizer = None
    
    def compute_score(self, query: str, passage: str) -> float:
        """
        Compute relevance score between query and passage
        
        Args:
            query: Search query
            passage: Document passage
            
        Returns:
            Relevance score (higher = more relevant)
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to simple text overlap scoring
            return self._simple_similarity_score(query, passage)
        
        try:
            # Prepare input for BGE reranker
            inputs = self.tokenizer(
                query, 
                passage,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                # BGE reranker outputs logits, take the positive class score
                score = outputs.logits[:, 1].item() if outputs.logits.size(1) > 1 else outputs.logits.item()
            
            return score
            
        except Exception as e:
            print(f"Error computing BGE score: {e}")
            return self._simple_similarity_score(query, passage)
    
    def _simple_similarity_score(self, query: str, passage: str) -> float:
        """
        Fallback simple similarity scoring based on term overlap
        """
        query_terms = set(query.lower().split())
        passage_terms = set(passage.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_terms.intersection(passage_terms))
        union = len(query_terms.union(passage_terms))
        
        return intersection / union if union > 0 else 0.0
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        # Compute scores for all documents
        scored_docs = []
        
        for doc in documents:
            score = self.compute_score(query, doc.page_content)
            scored_docs.append((doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, score in scored_docs[:top_k]]
    
    def batch_rerank(self, queries: List[str], documents_list: List[List[Document]], top_k: int = 4) -> List[List[Document]]:
        """
        Batch rerank multiple query-document pairs
        
        Args:
            queries: List of search queries
            documents_list: List of document lists (one per query)
            top_k: Number of top documents to return per query
            
        Returns:
            List of reranked document lists
        """
        results = []
        
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        
        return results


class CrossEncoderReranker:
    """
    Alternative cross-encoder reranker using sentence-transformers
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            print(f"CrossEncoder loaded: {model_name}")
        except ImportError:
            print("sentence-transformers not available, using BGE fallback")
            self.model = None
        except Exception as e:
            print(f"Error loading CrossEncoder: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """
        Rerank documents using cross-encoder
        """
        if self.model is None or not documents:
            return documents[:top_k]
        
        if len(documents) <= top_k:
            return documents
        
        # Prepare pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort documents by scores
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]


def compare_rerankers(query: str, documents: List[Document], top_k: int = 4) -> dict:
    """
    Compare different reranking methods
    """
    results = {}
    
    # BGE Reranker
    try:
        bge_reranker = BGEReranker()
        bge_results = bge_reranker.rerank(query, documents, top_k)
        results['bge'] = bge_results
    except Exception as e:
        results['bge_error'] = str(e)
    
    # CrossEncoder Reranker
    try:
        ce_reranker = CrossEncoderReranker()
        ce_results = ce_reranker.rerank(query, documents, top_k)
        results['cross_encoder'] = ce_results
    except Exception as e:
        results['cross_encoder_error'] = str(e)
    
    # Simple similarity (baseline)
    def simple_rerank(docs: List[Document]) -> List[Document]:
        scored = []
        query_terms = set(query.lower().split())
        
        for doc in docs:
            doc_terms = set(doc.page_content.lower().split())
            score = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
            scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_k]]
    
    results['simple'] = simple_rerank(documents)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test the reranker
    sample_query = "¿Cuáles son los síntomas de la diabetes?"
    
    sample_docs = [
        Document(page_content="La diabetes es una enfermedad crónica caracterizada por niveles altos de glucosa en sangre.", 
                metadata={"source": "diabetes_guide.pdf"}),
        Document(page_content="Los síntomas comunes incluyen sed excesiva, micción frecuente y fatiga.", 
                metadata={"source": "symptoms_manual.pdf"}),
        Document(page_content="El tratamiento puede incluir cambios en la dieta y medicación.", 
                metadata={"source": "treatment_guide.pdf"}),
        Document(page_content="La hipertensión arterial es otro factor de riesgo cardiovascular importante.", 
                metadata={"source": "cardio_risks.pdf"})
    ]
    
    # Test BGE reranker
    reranker = BGEReranker()
    reranked = reranker.rerank(sample_query, sample_docs, top_k=2)
    
    print(f"Query: {sample_query}")
    print(f"Original docs: {len(sample_docs)}")
    print(f"Reranked top-2:")
    for i, doc in enumerate(reranked):
        print(f"{i+1}. {doc.page_content[:100]}...")
