"""
Módulo RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
Implementa sumarios jerárquicos para mejorar la recuperación en consultas largas
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


class RAPTORNode:
    """
    Nodo en el árbol RAPTOR que puede ser un documento original o un sumario
    """
    def __init__(
        self, 
        content: str, 
        level: int, 
        node_id: str,
        children_ids: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.content = content
        self.level = level  # 0 = hojas (docs originales), >0 = sumarios
        self.node_id = node_id
        self.children_ids = children_ids or []
        self.metadata = metadata or {}
        self.embedding = None
    
    def to_document(self) -> Document:
        """Convierte el nodo a un Document de LangChain"""
        metadata = {
            **self.metadata,
            'raptor_level': self.level,
            'raptor_node_id': self.node_id,
            'raptor_children': self.children_ids,
            'is_summary': self.level > 0
        }
        return Document(page_content=self.content, metadata=metadata)


class RAPTORRetriever:
    """
    Retriever RAPTOR que construye y consulta una jerarquía de sumarios
    """
    
    def __init__(
        self,
        documents: List[Document],
        embeddings_model: HuggingFaceEmbeddings,
        llm: Optional[ChatOpenAI] = None,
        max_levels: int = 3,
        cluster_size: int = 10,
        top_k_per_level: int = 2
    ):
        """
        Args:
            documents: Documentos base (nivel 0)
            embeddings_model: Modelo para generar embeddings
            llm: Modelo LLM para generar sumarios
            max_levels: Niveles máximos de la jerarquía
            cluster_size: Tamaño promedio de clusters para sumarizar
            top_k_per_level: Número de nodos a recuperar por nivel
        """
        self.embeddings_model = embeddings_model
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.top_k_per_level = top_k_per_level
        
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
        
        # Prompt para sumarización clínica en español
        self.summary_prompt = PromptTemplate.from_template(
            """Eres un experto médico. Resume el siguiente conjunto de textos clínicos en español 
            en un único sumario coherente que capture los conceptos médicos clave, diagnósticos, 
            tratamientos y hallazgos relevantes. El resumen debe ser comprensivo pero conciso.

            TEXTOS A RESUMIR:
            {texts}

            RESUMEN CLÍNICO:"""
        )
        
        # Construir árbol RAPTOR
        self.tree = self._build_raptor_tree(documents)
        print(f"✅ Árbol RAPTOR construido con {len(self.tree)} nodos en {self.max_levels} niveles")
    
    def _build_raptor_tree(self, documents: List[Document]) -> Dict[str, RAPTORNode]:
        """
        Construye el árbol jerárquico de sumarios
        """
        tree = {}
        
        # Nivel 0: Documentos originales
        level_0_nodes = []
        for i, doc in enumerate(documents):
            node_id = f"L0_N{i}"
            node = RAPTORNode(
                content=doc.page_content,
                level=0,
                node_id=node_id,
                metadata=doc.metadata
            )
            tree[node_id] = node
            level_0_nodes.append(node)
        
        print(f"Nivel 0: {len(level_0_nodes)} documentos originales")
        
        # Generar embeddings para nivel 0
        self._compute_embeddings(level_0_nodes)
        
        # Construir niveles superiores recursivamente
        current_level_nodes = level_0_nodes
        
        for level in range(1, self.max_levels):
            # Si hay muy pocos nodos, no es necesario otro nivel
            if len(current_level_nodes) <= 2:
                break
            
            print(f"Construyendo nivel {level}...")
            next_level_nodes = self._create_summary_level(
                current_level_nodes, 
                level, 
                tree
            )
            
            if not next_level_nodes:
                break
            
            current_level_nodes = next_level_nodes
            print(f"Nivel {level}: {len(next_level_nodes)} sumarios creados")
        
        return tree
    
    def _compute_embeddings(self, nodes: List[RAPTORNode]):
        """
        Calcula embeddings para una lista de nodos
        """
        if not nodes:
            return
        
        texts = [node.content for node in nodes]
        embeddings = self.embeddings_model.embed_documents(texts)
        
        for node, embedding in zip(nodes, embeddings):
            node.embedding = np.array(embedding)
    
    def _create_summary_level(
        self, 
        nodes: List[RAPTORNode], 
        level: int, 
        tree: Dict[str, RAPTORNode]
    ) -> List[RAPTORNode]:
        """
        Crea un nivel de sumarios agrupando y resumiendo nodos
        """
        if len(nodes) <= 1:
            return []
        
        # Agrupar nodos usando clustering basado en embeddings
        clusters = self._cluster_nodes(nodes)
        
        summary_nodes = []
        
        for cluster_id, cluster_nodes in enumerate(clusters):
            if not cluster_nodes:
                continue
            
            # Crear sumario del cluster
            try:
                summary_content = self._summarize_nodes(cluster_nodes)
                
                node_id = f"L{level}_N{cluster_id}"
                children_ids = [node.node_id for node in cluster_nodes]
                
                summary_node = RAPTORNode(
                    content=summary_content,
                    level=level,
                    node_id=node_id,
                    children_ids=children_ids,
                    metadata={
                        'cluster_size': len(cluster_nodes),
                        'source_levels': [node.level for node in cluster_nodes]
                    }
                )
                
                tree[node_id] = summary_node
                summary_nodes.append(summary_node)
                
            except Exception as e:
                print(f"⚠️ Error creando sumario para cluster {cluster_id}: {str(e)}")
                continue
        
        # Calcular embeddings para los nuevos sumarios
        self._compute_embeddings(summary_nodes)
        
        return summary_nodes
    
    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """
        Agrupa nodos usando K-means clustering
        """
        if len(nodes) <= self.cluster_size:
            return [nodes]
        
        # Extraer embeddings
        embeddings = np.array([node.embedding for node in nodes])
        
        # Calcular número óptimo de clusters
        n_clusters = max(2, len(nodes) // self.cluster_size)
        n_clusters = min(n_clusters, len(nodes))
        
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Organizar nodos por cluster
            clusters = defaultdict(list)
            for node, label in zip(nodes, labels):
                clusters[label].append(node)
            
            return list(clusters.values())
            
        except Exception as e:
            print(f"⚠️ Error en clustering: {str(e)}")
            # Fallback: dividir en grupos de tamaño fijo
            return [nodes[i:i + self.cluster_size] for i in range(0, len(nodes), self.cluster_size)]
    
    def _summarize_nodes(self, nodes: List[RAPTORNode]) -> str:
        """
        Genera un sumario de un grupo de nodos usando el LLM
        """
        # Concatenar contenidos
        texts = "\n\n---\n\n".join([
            f"Texto {i+1}:\n{node.content[:1500]}"  # Limitar longitud
            for i, node in enumerate(nodes)
        ])
        
        # Generar sumario
        prompt = self.summary_prompt.format(texts=texts)
        response = self.llm.invoke(prompt)
        
        return response.content
    
    def retrieve(self, query: str, total_k: int = 8) -> List[Document]:
        """
        Recupera documentos relevantes consultando todos los niveles del árbol
        
        Args:
            query: Consulta del usuario
            total_k: Número total de documentos a retornar
            
        Returns:
            Lista de documentos rankeados por relevancia
        """
        # Generar embedding de la consulta
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        
        # Recuperar top-k nodos de cada nivel
        all_retrieved = []
        
        for level in range(self.max_levels):
            level_nodes = [node for node in self.tree.values() if node.level == level]
            
            if not level_nodes:
                continue
            
            # Calcular similitudes
            similarities = []
            for node in level_nodes:
                if node.embedding is not None:
                    sim = np.dot(query_embedding, node.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
                    )
                    similarities.append((node, sim))
            
            # Ordenar por similitud y tomar top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_nodes = similarities[:self.top_k_per_level]
            
            # Convertir a documentos
            for node, sim_score in top_nodes:
                doc = node.to_document()
                doc.metadata['raptor_similarity'] = float(sim_score)
                all_retrieved.append((doc, sim_score))
        
        # Re-rankear todos los documentos recuperados
        all_retrieved.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top-k documentos únicos
        seen_contents = set()
        unique_docs = []
        
        for doc, score in all_retrieved:
            content_hash = hash(doc.page_content[:200])  # Hash basado en inicio del contenido
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
            
            if len(unique_docs) >= total_k:
                break
        
        return unique_docs
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del árbol RAPTOR
        """
        level_counts = defaultdict(int)
        for node in self.tree.values():
            level_counts[node.level] += 1
        
        return {
            'total_nodes': len(self.tree),
            'levels': dict(level_counts),
            'max_level': max(level_counts.keys()) if level_counts else 0
        }


def build_raptor_retriever(
    vectorstore,
    embeddings_model: HuggingFaceEmbeddings,
    max_documents: int = 100,
    **raptor_kwargs
) -> Optional[RAPTORRetriever]:
    """
    Construye un RAPTORRetriever desde un vectorstore existente
    
    Args:
        vectorstore: Vector store de ChromaDB
        embeddings_model: Modelo de embeddings
        max_documents: Máximo número de documentos a usar para construir RAPTOR
        **raptor_kwargs: Argumentos adicionales para RAPTORRetriever
    """
    try:
        # Recuperar documentos del vectorstore
        # Hacemos una búsqueda genérica para obtener documentos variados
        sample_queries = [
            "información médica clínica",
            "diagnóstico tratamiento",
            "síntomas enfermedad"
        ]
        
        documents = []
        seen_contents = set()
        
        for query in sample_queries:
            results = vectorstore.similarity_search(
                query, 
                k=max_documents // len(sample_queries)
            )
            
            for doc in results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    documents.append(doc)
        
        if not documents:
            print("⚠️ No se encontraron documentos para construir RAPTOR")
            return None
        
        print(f"Construyendo RAPTOR con {len(documents)} documentos...")
        
        raptor = RAPTORRetriever(
            documents=documents,
            embeddings_model=embeddings_model,
            **raptor_kwargs
        )
        
        stats = raptor.get_tree_stats()
        print(f"✅ RAPTOR construido: {stats}")
        
        return raptor
        
    except Exception as e:
        print(f"❌ Error construyendo RAPTOR: {str(e)}")
        return None
