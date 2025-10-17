"""
Módulo de evaluación para RAG NextHealth
Implementa métricas nDCG, Recall@k y análisis de calidad
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime

from sklearn.metrics import ndcg_score
from .retrievers import create_retriever
from .graph import build_rag_chain


def load_test_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carga el dataset de test desde CSV
    Formato esperado: question, relevant_docs (separados por ;)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validar columnas requeridas
        required_columns = ['question', 'relevant_docs']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Columnas faltantes en CSV: {missing_cols}")
        
        # Limpiar datos
        df = df.dropna(subset=['question'])
        df['relevant_docs'] = df['relevant_docs'].fillna('')
        
        print(f"✅ Dataset cargado: {len(df)} preguntas")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {str(e)}")
        raise


def create_sample_test_dataset(output_path: str = "./test_dataset.csv"):
    """
    Crea un dataset de test de ejemplo para demostración
    """
    sample_data = [
        {
            "question": "¿Cuáles son los criterios diagnósticos para diabetes mellitus tipo 2?",
            "relevant_docs": "diabetes_guide.pdf;clinical_protocols.pdf;endocrine_manual.pdf",
            "expected_answer_type": "clinical_criteria",
            "difficulty": "medium"
        },
        {
            "question": "¿Qué código ICPC-3 corresponde a hipertensión arterial?",
            "relevant_docs": "icpc3_codes.csv;hypertension_classification.pdf",
            "expected_answer_type": "code_lookup",
            "difficulty": "easy"
        },
        {
            "question": "Explica el tratamiento farmacológico de primera línea para la hipertensión",
            "relevant_docs": "hypertension_treatment.pdf;pharmacology_guide.pdf;clinical_guidelines.pdf",
            "expected_answer_type": "treatment_explanation",
            "difficulty": "hard"
        },
        {
            "question": "¿Cómo se mapea T90 de ICPC-3 a CIE-10?",
            "relevant_docs": "icpc3_icd10_mapping.csv;code_equivalences.pdf",
            "expected_answer_type": "code_mapping",
            "difficulty": "easy"
        },
        {
            "question": "¿Cuáles son las complicaciones microvasculares de la diabetes?",
            "relevant_docs": "diabetes_complications.pdf;diabetic_retinopathy.pdf;diabetic_nephropathy.pdf",
            "expected_answer_type": "medical_complications",
            "difficulty": "medium"
        },
        {
            "question": "Lista todos los códigos ICPC-3 de la categoría cardiovascular",
            "relevant_docs": "icpc3_cardiovascular.csv;cardiac_codes.pdf",
            "expected_answer_type": "code_list",
            "difficulty": "medium"
        },
        {
            "question": "¿Qué es la retinopatía diabética y cómo se clasifica?",
            "relevant_docs": "diabetic_retinopathy.pdf;ophthalmology_guide.pdf;diabetes_complications.pdf",
            "expected_answer_type": "medical_definition",
            "difficulty": "hard"
        },
        {
            "question": "Criterios para derivación a especialista en caso de hipertensión resistente",
            "relevant_docs": "referral_criteria.pdf;hypertension_management.pdf;specialist_guidelines.pdf",
            "expected_answer_type": "referral_criteria",
            "difficulty": "hard"
        },
        {
            "question": "¿Cuántos códigos ICPC-3 existen en total en la base de datos?",
            "relevant_docs": "icpc3_statistics.csv;database_summary.pdf",
            "expected_answer_type": "statistical_query",
            "difficulty": "easy"
        },
        {
            "question": "Protocolo de seguimiento de pacientes con diabetes tipo 2",
            "relevant_docs": "diabetes_followup.pdf;clinical_protocols.pdf;monitoring_guidelines.pdf",
            "expected_answer_type": "clinical_protocol",
            "difficulty": "hard"
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset de ejemplo creado: {output_path}")
    print(f"   - {len(df)} preguntas de test")
    print(f"   - Tipos: {df['expected_answer_type'].value_counts().to_dict()}")
    print(f"   - Dificultades: {df['difficulty'].value_counts().to_dict()}")
    
    return df


def calculate_ndcg(relevant_docs: List[str], retrieved_docs: List[str], k: int = 5) -> float:
    """
    Calcula nDCG@k (Normalized Discounted Cumulative Gain)
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # Crear vector de relevancia
    relevance_scores = []
    
    for doc in retrieved_docs[:k]:
        if doc in relevant_docs:
            relevance_scores.append(1.0)
        else:
            relevance_scores.append(0.0)
    
    # Si no hay documentos relevantes, nDCG es 0
    if sum(relevance_scores) == 0:
        return 0.0
    
    # Crear scores ideales (todos los relevantes primero)
    num_relevant = min(len(relevant_docs), k)
    ideal_scores = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    
    try:
        # Calcular nDCG usando sklearn
        # Reshape para sklearn (necesita array 2D)
        true_relevance = np.array([ideal_scores])
        predicted_scores = np.array([relevance_scores])
        
        ndcg = ndcg_score(true_relevance, predicted_scores, k=k)
        return float(ndcg)
        
    except Exception as e:
        print(f"⚠️ Error calculando nDCG: {str(e)}")
        return 0.0


def calculate_recall(relevant_docs: List[str], retrieved_docs: List[str], k: int = 5) -> float:
    """
    Calcula Recall@k
    """
    if not relevant_docs:
        return 0.0
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    intersection = relevant_set.intersection(retrieved_set)
    
    recall = len(intersection) / len(relevant_set)
    return recall


def calculate_precision(relevant_docs: List[str], retrieved_docs: List[str], k: int = 5) -> float:
    """
    Calcula Precision@k
    """
    if not retrieved_docs:
        return 0.0
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    intersection = relevant_set.intersection(retrieved_set)
    
    precision = len(intersection) / min(len(retrieved_set), k)
    return precision


def evaluate_ndcg(relevant_docs: List[str], retrieved_docs: List[str], k: int = 5) -> float:
    """
    Wrapper para cálculo de nDCG (compatibilidad)
    """
    return calculate_ndcg(relevant_docs, retrieved_docs, k)


def extract_doc_ids_from_results(documents) -> List[str]:
    """
    Extrae IDs de documentos desde los resultados de retrieval
    """
    doc_ids = []
    
    for doc in documents:
        # Intentar extraer ID del documento de diferentes fuentes
        doc_id = None
        
        # Desde metadata
        if hasattr(doc, 'metadata'):
            doc_id = doc.metadata.get('source', None)
            if not doc_id:
                doc_id = doc.metadata.get('doc_id', None)
        
        # Desde el contenido (buscar patrones de archivo)
        if not doc_id and hasattr(doc, 'page_content'):
            content = doc.page_content[:200].lower()
            # Buscar nombres de archivo comunes
            import re
            file_pattern = r'(\w+\.(pdf|html|txt|md|csv))'
            match = re.search(file_pattern, content)
            if match:
                doc_id = match.group(1)
        
        # ID por defecto
        if not doc_id:
            doc_id = f"doc_{len(doc_ids)}"
        
        doc_ids.append(doc_id)
    
    return doc_ids


class RAGEvaluator:
    """
    Evaluador completo para sistema RAG
    """
    
    def __init__(self, persist_dir: str, sqlite_path: str = None):
        self.persist_dir = persist_dir
        self.sqlite_path = sqlite_path
        self.retriever = None
        self.rag_chain = None
        
        # Cargar componentes
        try:
            self.retriever = create_retriever(persist_dir=persist_dir, use_reranker=False)
            self.rag_chain = build_rag_chain(persist_dir, sqlite_path)
            print("✅ Evaluador inicializado correctamente")
        except Exception as e:
            print(f"❌ Error inicializando evaluador: {str(e)}")
    
    def evaluate_retrieval(self, test_df: pd.DataFrame, k_values: List[int] = [3, 5, 10]) -> Dict[str, Any]:
        """
        Evalúa la calidad del retrieval
        """
        results = {
            'questions_evaluated': 0,
            'average_metrics': {},
            'per_question_results': [],
            'k_values': k_values
        }
        
        if self.retriever is None:
            return {'error': 'Retriever no disponible'}
        
        for idx, row in test_df.iterrows():
            question = row['question']
            expected_docs = row['relevant_docs'].split(';') if pd.notna(row['relevant_docs']) else []
            
            if not expected_docs:
                continue
            
            try:
                start_time = time.time()
                
                # Recuperar documentos
                retrieved_docs = self.retriever.retrieve_and_fuse(question, k=max(k_values), final_k=max(k_values))
                retrieved_ids = extract_doc_ids_from_results(retrieved_docs)
                
                retrieval_time = time.time() - start_time
                
                # Calcular métricas para diferentes valores de k
                question_metrics = {
                    'question': question,
                    'expected_docs_count': len(expected_docs),
                    'retrieved_docs_count': len(retrieved_ids),
                    'retrieval_time': retrieval_time
                }
                
                for k in k_values:
                    ndcg = calculate_ndcg(expected_docs, retrieved_ids, k)
                    recall = calculate_recall(expected_docs, retrieved_ids, k)
                    precision = calculate_precision(expected_docs, retrieved_ids, k)
                    
                    question_metrics[f'ndcg@{k}'] = ndcg
                    question_metrics[f'recall@{k}'] = recall
                    question_metrics[f'precision@{k}'] = precision
                
                results['per_question_results'].append(question_metrics)
                results['questions_evaluated'] += 1
                
            except Exception as e:
                print(f"❌ Error evaluando pregunta '{question}': {str(e)}")
                continue
        
        # Calcular promedios
        if results['per_question_results']:
            df_results = pd.DataFrame(results['per_question_results'])
            
            for k in k_values:
                results['average_metrics'][f'ndcg@{k}'] = df_results[f'ndcg@{k}'].mean()
                results['average_metrics'][f'recall@{k}'] = df_results[f'recall@{k}'].mean()
                results['average_metrics'][f'precision@{k}'] = df_results[f'precision@{k}'].mean()
            
            results['average_metrics']['average_retrieval_time'] = df_results['retrieval_time'].mean()
        
        return results
    
    def evaluate_generation(self, test_df: pd.DataFrame, sample_size: int = None) -> Dict[str, Any]:
        """
        Evalúa la calidad de generación de respuestas
        """
        if self.rag_chain is None:
            return {'error': 'RAG chain no disponible'}
        
        # Limitar el tamaño de muestra si se especifica
        if sample_size and len(test_df) > sample_size:
            test_df = test_df.sample(n=sample_size, random_state=42)
        
        results = {
            'questions_evaluated': 0,
            'generation_results': [],
            'average_response_time': 0,
            'policy_violations': 0
        }
        
        total_time = 0
        
        for idx, row in test_df.iterrows():
            question = row['question']
            
            try:
                start_time = time.time()
                response = self.rag_chain.invoke({"question": question})
                generation_time = time.time() - start_time
                
                total_time += generation_time
                
                # Análisis básico de la respuesta
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                question_result = {
                    'question': question,
                    'response_length': len(response_text),
                    'generation_time': generation_time,
                    'has_disclaimer': 'aviso' in response_text.lower() or 'importante' in response_text.lower(),
                    'has_sources': '[' in response_text and ']' in response_text,
                    'response_preview': response_text[:200] + '...' if len(response_text) > 200 else response_text
                }
                
                results['generation_results'].append(question_result)
                results['questions_evaluated'] += 1
                
            except Exception as e:
                print(f"❌ Error generando respuesta para '{question}': {str(e)}")
                continue
        
        if results['questions_evaluated'] > 0:
            results['average_response_time'] = total_time / results['questions_evaluated']
        
        return results
    
    def run_full_evaluation(self, test_df: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta evaluación completa (retrieval + generation)
        """
        print("🚀 Iniciando evaluación completa...")
        
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'test_questions': len(test_df),
            'retrieval_evaluation': {},
            'generation_evaluation': {},
            'system_info': {
                'persist_dir': self.persist_dir,
                'sqlite_path': self.sqlite_path,
                'retriever_available': self.retriever is not None,
                'rag_chain_available': self.rag_chain is not None
            }
        }
        
        # Evaluación de retrieval
        print("1️⃣ Evaluando retrieval...")
        full_results['retrieval_evaluation'] = self.evaluate_retrieval(test_df)
        
        # Evaluación de generación (muestra más pequeña)
        print("2️⃣ Evaluando generación...")
        sample_size = min(5, len(test_df))  # Máximo 5 para no sobrecargar
        full_results['generation_evaluation'] = self.evaluate_generation(test_df, sample_size)
        
        # Guardar resultados si se especifica path
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados guardados en: {output_path}")
        
        print("✅ Evaluación completa terminada")
        return full_results


def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """
    Genera un reporte legible de los resultados de evaluación
    """
    report_lines = [
        "# 📊 Reporte de Evaluación RAG NextHealth",
        f"**Fecha:** {results.get('timestamp', 'N/A')}",
        f"**Preguntas evaluadas:** {results.get('test_questions', 'N/A')}",
        ""
    ]
    
    # Reporte de retrieval
    retrieval = results.get('retrieval_evaluation', {})
    if retrieval and 'average_metrics' in retrieval:
        report_lines.extend([
            "## 🔍 Evaluación de Retrieval",
            f"- Preguntas procesadas: {retrieval.get('questions_evaluated', 0)}",
            f"- Tiempo promedio: {retrieval['average_metrics'].get('average_retrieval_time', 0):.3f}s",
            ""
        ])
        
        # Métricas por k
        for k in [3, 5, 10]:
            ndcg = retrieval['average_metrics'].get(f'ndcg@{k}', 0)
            recall = retrieval['average_metrics'].get(f'recall@{k}', 0)
            precision = retrieval['average_metrics'].get(f'precision@{k}', 0)
            
            if ndcg > 0 or recall > 0:
                report_lines.extend([
                    f"### Métricas @{k}",
                    f"- **nDCG@{k}:** {ndcg:.3f}",
                    f"- **Recall@{k}:** {recall:.3f}",
                    f"- **Precision@{k}:** {precision:.3f}",
                    ""
                ])
    
    # Reporte de generación
    generation = results.get('generation_evaluation', {})
    if generation and 'generation_results' in generation:
        report_lines.extend([
            "## ✍️ Evaluación de Generación",
            f"- Respuestas generadas: {generation.get('questions_evaluated', 0)}",
            f"- Tiempo promedio: {generation.get('average_response_time', 0):.3f}s",
            ""
        ])
        
        # Análisis de calidad
        results_list = generation['generation_results']
        if results_list:
            avg_length = np.mean([r['response_length'] for r in results_list])
            has_disclaimer_pct = np.mean([r['has_disclaimer'] for r in results_list]) * 100
            has_sources_pct = np.mean([r['has_sources'] for r in results_list]) * 100
            
            report_lines.extend([
                f"- **Longitud promedio:** {avg_length:.0f} caracteres",
                f"- **Con disclaimer:** {has_disclaimer_pct:.1f}%",
                f"- **Con fuentes:** {has_sources_pct:.1f}%",
                ""
            ])
    
    return "\n".join(report_lines)
