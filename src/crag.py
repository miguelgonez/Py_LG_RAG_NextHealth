"""
Módulo CRAG (Corrective Retrieval Augmented Generation)
Evalúa relevancia de resultados y re-busca automáticamente si la calidad es baja
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class RelevanceEvaluator:
    """
    Evaluador de relevancia que determina si los documentos recuperados
    son útiles para responder la consulta
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None, threshold: float = 0.6):
        """
        Args:
            llm: Modelo LLM para evaluación
            threshold: Umbral de relevancia (0-1)
        """
        if llm is None:
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5"),
                api_key=os.getenv("OPENAI_API_KEY")
                # Note: gpt-5 only supports temperature=1 (default)
            )
        else:
            self.llm = llm
        
        self.threshold = threshold
        
        # Prompt para evaluar relevancia
        self.relevance_prompt = PromptTemplate.from_template(
            """Eres un experto evaluador de relevancia para búsqueda médica.

            PREGUNTA DEL USUARIO:
            {question}

            DOCUMENTO A EVALUAR:
            {document}

            INSTRUCCIONES:
            Evalúa si el documento anterior contiene información RELEVANTE para responder la pregunta.
            Un documento es relevante si:
            - Aborda directamente el tema de la pregunta
            - Proporciona información médica útil para responder
            - Contiene datos, criterios o protocolos relacionados con la consulta
            
            Responde SOLO con un número del 0 al 10:
            - 0-3: No relevante (información no relacionada)
            - 4-6: Parcialmente relevante (relacionado pero insuficiente)
            - 7-10: Muy relevante (información útil y directa)

            SCORE DE RELEVANCIA (solo el número):"""
        )
    
    def evaluate_document(self, query: str, document: Document) -> float:
        """
        Evalúa la relevancia de un documento único
        
        Returns:
            Score de relevancia normalizado (0-1)
        """
        try:
            prompt = self.relevance_prompt.format(
                question=query,
                document=document.page_content[:1000]  # Limitar longitud
            )
            
            response = self.llm.invoke(prompt)
            
            # Parsear score
            score_text = response.content.strip()
            # Extraer primer número encontrado
            import re
            numbers = re.findall(r'\d+', score_text)
            
            if numbers:
                score = float(numbers[0]) / 10.0  # Normalizar a 0-1
                return min(1.0, max(0.0, score))
            else:
                # Si no hay número, evaluar por presencia de palabras clave
                return self._fallback_scoring(query, document)
                
        except Exception as e:
            print(f"⚠️ Error evaluando relevancia: {str(e)}")
            return self._fallback_scoring(query, document)
    
    def _fallback_scoring(self, query: str, document: Document) -> float:
        """
        Sistema de puntuación de fallback basado en heurísticas
        """
        query_terms = set(query.lower().split())
        doc_text = document.page_content.lower()
        
        # Contar términos que aparecen
        matches = sum(1 for term in query_terms if term in doc_text)
        score = matches / len(query_terms) if query_terms else 0.0
        
        return min(1.0, score)
    
    def evaluate_batch(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Evalúa relevancia de un batch de documentos
        
        Returns:
            Lista de tuplas (documento, score_relevancia)
        """
        results = []
        
        for doc in documents:
            score = self.evaluate_document(query, doc)
            results.append((doc, score))
        
        return results
    
    def is_relevant(self, query: str, documents: List[Document]) -> bool:
        """
        Determina si el conjunto de documentos es suficientemente relevante
        
        Returns:
            True si la relevancia promedio supera el umbral
        """
        if not documents:
            return False
        
        scores = [self.evaluate_document(query, doc) for doc in documents]
        avg_score = sum(scores) / len(scores)
        
        return avg_score >= self.threshold


class CRAGRetriever:
    """
    CRAG Retriever que evalúa y corrige automáticamente resultados de baja relevancia
    """
    
    def __init__(
        self,
        base_retriever,
        evaluator: Optional[RelevanceEvaluator] = None,
        web_search_fallback: bool = False,
        max_iterations: int = 2
    ):
        """
        Args:
            base_retriever: Retriever base a usar
            evaluator: Evaluador de relevancia
            web_search_fallback: Si True, busca en web como último recurso
            max_iterations: Máximo número de re-búsquedas
        """
        self.base_retriever = base_retriever
        self.evaluator = evaluator or RelevanceEvaluator()
        self.web_search_fallback = web_search_fallback
        self.max_iterations = max_iterations
    
    def retrieve_with_correction(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Recupera documentos con corrección automática si la relevancia es baja
        
        Returns:
            Tupla de (documentos, metadata_crag)
        """
        metadata = {
            "iterations": 0,
            "relevance_scores": [],
            "correction_applied": False,
            "final_relevance": 0.0
        }
        
        # Primera búsqueda
        docs = self._retrieve_from_base(query, k)
        metadata["iterations"] = 1
        
        # Evaluar relevancia
        doc_scores = self.evaluator.evaluate_batch(query, docs)
        scores = [score for _, score in doc_scores]
        avg_relevance = sum(scores) / len(scores) if scores else 0.0
        
        metadata["relevance_scores"].append(avg_relevance)
        metadata["final_relevance"] = avg_relevance
        
        print(f"CRAG - Relevancia inicial: {avg_relevance:.2f}")
        
        # Si la relevancia es baja, aplicar correcciones
        if avg_relevance < self.evaluator.threshold and metadata["iterations"] < self.max_iterations:
            print("⚠️ CRAG - Relevancia baja, aplicando corrección...")
            metadata["correction_applied"] = True
            
            # Estrategia de corrección 1: Reformular y re-buscar
            refined_query = self._refine_query(query, docs)
            print(f"   Consulta refinada: {refined_query[:100]}...")
            
            docs = self._retrieve_from_base(refined_query, k)
            metadata["iterations"] += 1
            
            # Re-evaluar
            doc_scores = self.evaluator.evaluate_batch(refined_query, docs)
            scores = [score for _, score in doc_scores]
            avg_relevance = sum(scores) / len(scores) if scores else 0.0
            
            metadata["relevance_scores"].append(avg_relevance)
            metadata["final_relevance"] = avg_relevance
            
            print(f"   Relevancia tras corrección: {avg_relevance:.2f}")
        
        # Filtrar documentos por relevancia mínima
        filtered_docs = [
            doc for doc, score in doc_scores
            if score >= self.evaluator.threshold * 0.7  # 70% del umbral
        ]
        
        if not filtered_docs:
            # Si todo fue filtrado, mantener los mejores originales
            filtered_docs = docs[:k]
        
        # Agregar metadata de CRAG a documentos
        for i, (doc, score) in enumerate(doc_scores[:len(filtered_docs)]):
            doc.metadata['crag_relevance'] = float(score)
            doc.metadata['crag_rank'] = i + 1
            doc.metadata['crag_corrected'] = metadata["correction_applied"]
        
        return filtered_docs, metadata
    
    def _retrieve_from_base(self, query: str, k: int) -> List[Document]:
        """
        Recupera documentos del retriever base
        """
        try:
            # Detectar tipo de retriever y llamar método apropiado
            if hasattr(self.base_retriever, 'retrieve_with_fusion'):
                return self.base_retriever.retrieve_with_fusion(query, final_k=k)
            elif hasattr(self.base_retriever, 'retrieve_with_hyde'):
                return self.base_retriever.retrieve_with_hyde(query, k=k)
            elif hasattr(self.base_retriever, 'retrieve'):
                try:
                    return self.base_retriever.retrieve(query, total_k=k)
                except TypeError:
                    return self.base_retriever.retrieve(query, k=k)
            elif hasattr(self.base_retriever, 'retrieve_and_fuse'):
                return self.base_retriever.retrieve_and_fuse(query, k=k*2, final_k=k)
            else:
                # Fallback
                return self.base_retriever.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Error en retrieve_from_base: {str(e)}")
            return []
    
    def _refine_query(self, original_query: str, low_relevance_docs: List[Document]) -> str:
        """
        Refina la consulta basándose en los documentos de baja relevancia
        para mejorar la siguiente búsqueda
        """
        try:
            # Extraer temas de los documentos recuperados
            doc_snippets = "\n".join([
                doc.page_content[:200] for doc in low_relevance_docs[:3]
            ])
            
            refine_prompt = PromptTemplate.from_template(
                """La búsqueda anterior no encontró información muy relevante.

                PREGUNTA ORIGINAL: {original_query}

                DOCUMENTOS RECUPERADOS (poco relevantes):
                {doc_snippets}

                TAREA: Reformula la pregunta original para mejorar la búsqueda.
                La nueva consulta debe:
                - Usar terminología médica más específica
                - Ser más clara y directa
                - Enfocarse en aspectos clave de la pregunta
                - Evitar ambigüedades

                CONSULTA REFINADA (solo la pregunta, sin explicaciones):"""
            )
            
            prompt = refine_prompt.format(
                original_query=original_query,
                doc_snippets=doc_snippets
            )
            
            response = self.evaluator.llm.invoke(prompt)
            refined = response.content.strip()
            
            # Si la reformulación es muy corta o similar, usar original
            if len(refined) < 10 or refined.lower() == original_query.lower():
                return original_query
            
            return refined
            
        except Exception as e:
            print(f"⚠️ Error refinando consulta: {str(e)}")
            return original_query


def build_crag_retriever(base_retriever, threshold: float = 0.6) -> CRAGRetriever:
    """
    Construye un CRAG retriever wrapper
    
    Args:
        base_retriever: Retriever base a envolver
        threshold: Umbral de relevancia para corrección
    """
    evaluator = RelevanceEvaluator(threshold=threshold)
    return CRAGRetriever(
        base_retriever=base_retriever,
        evaluator=evaluator,
        max_iterations=2
    )
