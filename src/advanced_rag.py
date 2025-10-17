"""
Módulo de técnicas avanzadas de RAG: RAG-Fusion y HYDE
Implementa estrategias para mejorar el recall en casos difíciles
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class HYDEGenerator:
    """
    HYDE (Hypothetical Document Embeddings)
    Genera documentos hipotéticos que responden la pregunta,
    luego busca documentos similares a estos hipotéticos
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Args:
            llm: Modelo LLM para generar documentos hipotéticos
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
        
        # Prompt para generar documento hipotético en español clínico
        self.hyde_prompt = PromptTemplate.from_template(
            """Eres un experto médico escribiendo un fragmento de documentación clínica.

            PREGUNTA: {question}

            Escribe un párrafo técnico de 150-200 palabras que respondería perfectamente esta pregunta.
            El texto debe:
            - Usar terminología médica precisa en español
            - Incluir datos específicos, criterios diagnósticos, o protocolos relevantes
            - Escribirse como si fuera parte de una guía clínica o manual médico
            - Ser informativo y estructurado
            
            No uses frases como "La respuesta es..." o "Para responder...". Escribe directamente el contenido médico.

            DOCUMENTO HIPOTÉTICO:"""
        )
    
    def generate_hypothetical_docs(self, query: str, num_docs: int = 3) -> List[str]:
        """
        Genera documentos hipotéticos para la consulta
        
        Args:
            query: Consulta del usuario
            num_docs: Número de documentos hipotéticos a generar
            
        Returns:
            Lista de strings con documentos hipotéticos
        """
        hypothetical_docs = []
        
        try:
            for i in range(num_docs):
                prompt = self.hyde_prompt.format(question=query)
                response = self.llm.invoke(prompt)
                hypothetical_docs.append(response.content)
                print(f"✅ Documento hipotético {i+1} generado")
            
            return hypothetical_docs
            
        except Exception as e:
            print(f"❌ Error generando documentos hipotéticos: {str(e)}")
            # Fallback: retornar la consulta original
            return [query]


class RAGFusionRetriever:
    """
    RAG-Fusion: Genera múltiples reformulaciones de la consulta,
    recupera documentos para cada una, y fusiona resultados usando RRF
    (Reciprocal Rank Fusion)
    """
    
    def __init__(
        self,
        vectorstore,
        llm: Optional[ChatOpenAI] = None,
        num_queries: int = 4,
        k_per_query: int = 6
    ):
        """
        Args:
            vectorstore: Vector store para búsqueda
            llm: Modelo LLM para generar consultas
            num_queries: Número de reformulaciones a generar
            k_per_query: Documentos a recuperar por consulta
        """
        self.vectorstore = vectorstore
        self.num_queries = num_queries
        self.k_per_query = k_per_query
        
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
        
        # Prompt para generar reformulaciones diversas
        self.fusion_prompt = PromptTemplate.from_template(
            """Eres un experto en búsqueda de información médica. Dada una pregunta clínica,
            genera {num_queries} reformulaciones DIFERENTES que:
            - Mantengan el mismo significado médico
            - Usen sinónimos y terminología alternativa
            - Enfaticen diferentes aspectos de la pregunta
            - Ayuden a maximizar la recuperación de información relevante
            
            PREGUNTA ORIGINAL: {question}
            
            Genera exactamente {num_queries} reformulaciones, una por línea, numeradas:
            1.
            2.
            3.
            {extra_line}"""
        )
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """
        Genera reformulaciones de la consulta original
        """
        try:
            extra_line = "4." if self.num_queries >= 4 else ""
            prompt = self.fusion_prompt.format(
                question=query,
                num_queries=self.num_queries,
                extra_line=extra_line
            )
            
            response = self.llm.invoke(prompt)
            
            # Parsear las reformulaciones
            lines = response.content.strip().split('\n')
            variants = []
            
            for line in lines:
                # Remover numeración
                cleaned = line.strip()
                if cleaned and len(cleaned) > 3:
                    # Remover prefijos como "1.", "2.", etc.
                    if cleaned[0].isdigit() and '.' in cleaned[:4]:
                        cleaned = cleaned.split('.', 1)[1].strip()
                    variants.append(cleaned)
            
            # Incluir consulta original
            variants.insert(0, query)
            
            # Limitar al número solicitado
            variants = variants[:self.num_queries]
            
            print(f"✅ Generadas {len(variants)} variantes de consulta")
            return variants
            
        except Exception as e:
            print(f"❌ Error generando variantes: {str(e)}")
            return [query]  # Fallback a consulta original
    
    def _reciprocal_rank_fusion(
        self,
        doc_lists: List[List[Document]],
        k: int = 60
    ) -> List[Document]:
        """
        Fusiona múltiples listas de documentos usando Reciprocal Rank Fusion
        
        RRF Score = Σ (1 / (k + rank_i)) para cada lista donde aparece el documento
        
        Args:
            doc_lists: Lista de listas de documentos
            k: Constante para RRF (típicamente 60)
            
        Returns:
            Lista fusionada y rankeada de documentos
        """
        # Calcular scores RRF
        rrf_scores = defaultdict(float)
        doc_map = {}  # Mapeo de ID a documento
        
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                # Crear ID único basado en contenido
                doc_id = self._get_doc_id(doc)
                
                # Calcular RRF score
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                
                # Guardar documento
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # Ordenar por score RRF
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Construir lista final con metadata
        fused_docs = []
        for doc_id, score in sorted_docs:
            doc = doc_map[doc_id]
            doc.metadata['rrf_score'] = float(score)
            doc.metadata['fusion_method'] = 'RAG-Fusion'
            fused_docs.append(doc)
        
        return fused_docs
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        Genera ID único para un documento
        """
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 0)
        content_hash = hash(doc.page_content[:100])
        return f"{source}_{page}_{content_hash}"
    
    def retrieve_with_fusion(self, query: str, final_k: int = 5) -> List[Document]:
        """
        Recupera documentos usando RAG-Fusion
        
        Args:
            query: Consulta original
            final_k: Número final de documentos a retornar
            
        Returns:
            Lista fusionada de documentos relevantes
        """
        try:
            # 1. Generar variantes de la consulta
            query_variants = self._generate_query_variants(query)
            print(f"📝 Variantes: {len(query_variants)}")
            
            # 2. Recuperar documentos para cada variante
            all_doc_lists = []
            for i, variant in enumerate(query_variants):
                docs = self.vectorstore.similarity_search(
                    variant,
                    k=self.k_per_query
                )
                all_doc_lists.append(docs)
                print(f"   Variante {i+1}: {len(docs)} docs")
            
            # 3. Aplicar RRF fusion
            fused_docs = self._reciprocal_rank_fusion(all_doc_lists)
            
            print(f"✅ RAG-Fusion: {len(fused_docs)} docs únicos fusionados")
            
            # 4. Retornar top-k
            return fused_docs[:final_k]
            
        except Exception as e:
            print(f"❌ Error en RAG-Fusion: {str(e)}")
            # Fallback a búsqueda simple
            return self.vectorstore.similarity_search(query, k=final_k)


class HYDERetriever:
    """
    Retriever que combina HYDE con búsqueda vectorial
    """
    
    def __init__(
        self,
        vectorstore,
        embeddings_model,
        llm: Optional[ChatOpenAI] = None,
        num_hypothetical: int = 2
    ):
        """
        Args:
            vectorstore: Vector store para búsqueda
            embeddings_model: Modelo de embeddings
            llm: Modelo LLM para generar hipótesis
            num_hypothetical: Número de documentos hipotéticos a generar
        """
        self.vectorstore = vectorstore
        self.embeddings_model = embeddings_model
        self.hyde_generator = HYDEGenerator(llm)
        self.num_hypothetical = num_hypothetical
    
    def retrieve_with_hyde(self, query: str, k: int = 5) -> List[Document]:
        """
        Recupera documentos usando HYDE
        
        Args:
            query: Consulta del usuario
            k: Número de documentos a retornar
            
        Returns:
            Documentos relevantes encontrados vía HYDE
        """
        try:
            # 1. Generar documentos hipotéticos
            hypothetical_docs = self.hyde_generator.generate_hypothetical_docs(
                query,
                num_docs=self.num_hypothetical
            )
            
            # 2. Buscar con cada documento hipotético
            all_results = []
            seen_ids = set()
            
            for i, hyp_doc in enumerate(hypothetical_docs):
                # Búsqueda de similitud con documento hipotético
                docs = self.vectorstore.similarity_search(
                    hyp_doc,
                    k=k
                )
                
                # Evitar duplicados
                for doc in docs:
                    doc_id = hash(doc.page_content[:100])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc.metadata['hyde_source'] = i + 1
                        doc.metadata['retrieval_method'] = 'HYDE'
                        all_results.append(doc)
            
            print(f"✅ HYDE: {len(all_results)} documentos únicos recuperados")
            
            # 3. Retornar top-k
            return all_results[:k]
            
        except Exception as e:
            print(f"❌ Error en HYDE: {str(e)}")
            # Fallback a búsqueda directa con la consulta
            return self.vectorstore.similarity_search(query, k=k)


def build_fusion_retriever(vectorstore, llm=None) -> RAGFusionRetriever:
    """
    Construye un RAG-Fusion retriever
    """
    return RAGFusionRetriever(
        vectorstore=vectorstore,
        llm=llm,
        num_queries=4,
        k_per_query=6
    )


def build_hyde_retriever(vectorstore, embeddings_model, llm=None) -> HYDERetriever:
    """
    Construye un HYDE retriever
    """
    return HYDERetriever(
        vectorstore=vectorstore,
        embeddings_model=embeddings_model,
        llm=llm,
        num_hypothetical=2
    )
