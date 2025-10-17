"""
Módulo LangGraph RAG con memoria conversacional, Self-RAG y CRAG mejorado
Implementa un grafo de estado con ciclos para corrección automática
"""

import os
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
import time

from langchain.schema import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from .retrievers import (
    build_advanced_rag_retriever,
    build_vectorstore
)
from .reranker import BGEReranker
from .crag import RelevanceEvaluator
from .tools import analyze_sql_intent, get_sql_chain, run_sql
from .policies import DEFAULT_POLICY, format_response_with_policy


# ============================================================================
# ESTADO DEL GRAFO
# ============================================================================

class RAGState(TypedDict):
    """Estado compartido entre todos los nodos del grafo"""

    # Input del usuario
    question: str
    original_question: str  # Guardamos la pregunta original
    conversation_id: str

    # Memoria conversacional
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chat_history: list[tuple[str, str]]  # (user, assistant) pairs

    # Routing
    route: str  # "sql" | "vector" | "hybrid"
    route_confidence: float
    route_reasoning: str

    # Retrieval
    documents: list[Document]
    retrieval_method: str
    retrieval_iteration: int  # Para CRAG con límite de intentos

    # Reranking
    reranked_docs: list[Document]

    # Evaluation (Self-RAG)
    relevance_score: float
    can_answer: bool
    needs_refinement: bool

    # Generation
    response: str

    # Metadata y tracking
    error: str | None
    latency_ms: dict[str, float]
    node_sequence: list[str]  # Para debugging


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def get_llm():
    """Obtiene el LLM configurado"""
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5"),
        api_key=os.getenv("OPENAI_API_KEY")
        # Note: gpt-5 only supports temperature=1 (default)
    )


# ============================================================================
# NODOS DEL GRAFO
# ============================================================================

def contextualize_question(state: RAGState) -> RAGState:
    """
    Nodo 1: Contextualiza la pregunta con el historial de conversación
    """
    start_time = time.time()

    state["node_sequence"].append("contextualize")

    # Si no hay historial, usar pregunta original
    if not state.get("chat_history") or len(state["chat_history"]) == 0:
        state["original_question"] = state["question"]
        state["latency_ms"]["contextualize"] = (time.time() - start_time) * 1000
        return state

    # Contextualizar con historial
    llm = get_llm()

    # Formatear historial
    history_text = "\n".join([
        f"Usuario: {q}\nAsistente: {a}"
        for q, a in state["chat_history"][-3:]  # Últimas 3 interacciones
    ])

    contextualize_prompt = PromptTemplate.from_template("""
Dada una conversación médica y una pregunta de seguimiento, reformula la pregunta
para que sea autocontenida (standalone) sin perder el contexto médico.

HISTORIAL DE CONVERSACIÓN:
{history}

PREGUNTA DE SEGUIMIENTO: {question}

PREGUNTA REFORMULADA (solo la pregunta, sin explicaciones):""")

    try:
        prompt = contextualize_prompt.format(
            history=history_text,
            question=state["question"]
        )

        response = llm.invoke(prompt)
        contextualized = response.content.strip()

        state["original_question"] = state["question"]
        state["question"] = contextualized

        print(f"📝 Pregunta contextualizada: {contextualized[:100]}...")

    except Exception as e:
        print(f"⚠️ Error contextualizando: {str(e)}")
        state["original_question"] = state["question"]

    state["latency_ms"]["contextualize"] = (time.time() - start_time) * 1000
    return state


def route_query(state: RAGState) -> RAGState:
    """
    Nodo 2: Determina la ruta (SQL vs Vector)
    """
    start_time = time.time()
    state["node_sequence"].append("route")

    try:
        intent_analysis = analyze_sql_intent(state["question"])

        state["route"] = intent_analysis["intent"]
        state["route_confidence"] = intent_analysis["confidence"]
        state["route_reasoning"] = intent_analysis["reasoning"]

        print(f"🧭 Ruta: {state['route']} (confianza: {state['route_confidence']:.2f})")

    except Exception as e:
        print(f"⚠️ Error en routing: {str(e)}")
        state["route"] = "vector"
        state["route_confidence"] = 0.5
        state["error"] = f"Routing error: {str(e)}"

    state["latency_ms"]["route"] = (time.time() - start_time) * 1000
    return state


def retrieve_documents(state: RAGState) -> RAGState:
    """
    Nodo 3: Recupera documentos usando la estrategia configurada
    """
    start_time = time.time()
    state["node_sequence"].append("retrieve")

    persist_dir = os.getenv("PERSIST_DIR", "./db/chroma")
    sqlite_path = os.getenv("SQLITE_PATH", "./db/icpc3.db")

    try:
        if state["route"] == "sql":
            # Ruta SQL
            sql_chain, db = get_sql_chain(sqlite_path)
            generated_sql = sql_chain.invoke({"question": state["question"]})
            df = run_sql(sqlite_path, generated_sql)

            if not df.empty:
                doc_content = f"Resultados SQL:\n{df.to_string(index=False)}"
                state["documents"] = [
                    Document(
                        page_content=doc_content,
                        metadata={
                            "source": "SQL Database",
                            "sql_query": generated_sql,
                            "route": "sql"
                        }
                    )
                ]
                state["retrieval_method"] = "sql"
            else:
                state["documents"] = []
                state["retrieval_method"] = "sql_empty"

        else:
            # Ruta Vector - usar retriever configurado
            retrieval_mode = os.getenv("RETRIEVAL_MODE", "standard")
            retriever = build_advanced_rag_retriever(persist_dir, mode=retrieval_mode)

            if retriever is None:
                raise ValueError("No se pudo cargar el retriever")

            # Detectar tipo de retriever y llamar método apropiado
            if hasattr(retriever, 'retrieve_with_correction'):
                docs, crag_meta = retriever.retrieve_with_correction(state["question"], k=8)
                state["retrieval_method"] = "crag"
            elif hasattr(retriever, 'retrieve_with_fusion'):
                docs = retriever.retrieve_with_fusion(state["question"], final_k=8)
                state["retrieval_method"] = "rag-fusion"
            elif hasattr(retriever, 'retrieve_with_hyde'):
                docs = retriever.retrieve_with_hyde(state["question"], k=8)
                state["retrieval_method"] = "hyde"
            elif hasattr(retriever, 'retrieve'):
                try:
                    docs = retriever.retrieve(state["question"], total_k=8)
                except TypeError:
                    docs = retriever.retrieve(state["question"], k=8)
                state["retrieval_method"] = "raptor"
            elif hasattr(retriever, 'retrieve_and_fuse'):
                docs = retriever.retrieve_and_fuse(state["question"], k=10, final_k=8)
                state["retrieval_method"] = "fusion"
            else:
                docs = retriever.vectorstore.similarity_search(state["question"], k=8)
                state["retrieval_method"] = "standard"

            state["documents"] = docs

        print(f"📚 Recuperados {len(state['documents'])} documentos ({state['retrieval_method']})")

    except Exception as e:
        print(f"❌ Error en retrieval: {str(e)}")
        state["documents"] = []
        state["error"] = f"Retrieval error: {str(e)}"

    state["latency_ms"]["retrieve"] = (time.time() - start_time) * 1000
    return state


def rerank_documents(state: RAGState) -> RAGState:
    """
    Nodo 4: Re-rankea documentos con BGE
    """
    start_time = time.time()
    state["node_sequence"].append("rerank")

    # Skip reranking para SQL
    if state["route"] == "sql":
        state["reranked_docs"] = state["documents"]
        state["latency_ms"]["rerank"] = (time.time() - start_time) * 1000
        return state

    try:
        if not state["documents"]:
            state["reranked_docs"] = []
        else:
            reranker = BGEReranker()
            reranked = reranker.rerank(
                state["question"],
                state["documents"],
                top_k=4
            )
            state["reranked_docs"] = reranked

            print(f"🔄 Reranking: {len(state['documents'])} → {len(reranked)} docs")

    except Exception as e:
        print(f"⚠️ Error en reranking: {str(e)}")
        state["reranked_docs"] = state["documents"][:4]

    state["latency_ms"]["rerank"] = (time.time() - start_time) * 1000
    return state


def evaluate_relevance(state: RAGState) -> RAGState:
    """
    Nodo 5: Evalúa la relevancia de los documentos (Self-RAG)
    """
    start_time = time.time()
    state["node_sequence"].append("evaluate")

    # Skip evaluación para SQL
    if state["route"] == "sql":
        state["relevance_score"] = 1.0 if state["reranked_docs"] else 0.0
        state["can_answer"] = bool(state["reranked_docs"])
        state["needs_refinement"] = False
        state["latency_ms"]["evaluate"] = (time.time() - start_time) * 1000
        return state

    try:
        if not state["reranked_docs"]:
            state["relevance_score"] = 0.0
            state["can_answer"] = False
            state["needs_refinement"] = True
        else:
            evaluator = RelevanceEvaluator(threshold=0.6)

            # Evaluar documentos
            doc_scores = evaluator.evaluate_batch(
                state["question"],
                state["reranked_docs"]
            )

            scores = [score for _, score in doc_scores]
            avg_relevance = sum(scores) / len(scores) if scores else 0.0

            state["relevance_score"] = avg_relevance
            state["can_answer"] = avg_relevance >= 0.6
            state["needs_refinement"] = avg_relevance < 0.4

            print(f"📊 Relevancia promedio: {avg_relevance:.2f}")

    except Exception as e:
        print(f"⚠️ Error evaluando relevancia: {str(e)}")
        state["relevance_score"] = 0.5
        state["can_answer"] = bool(state["reranked_docs"])
        state["needs_refinement"] = False

    state["latency_ms"]["evaluate"] = (time.time() - start_time) * 1000
    return state


def refine_query(state: RAGState) -> RAGState:
    """
    Nodo 6: Refina la consulta si la relevancia es muy baja
    """
    start_time = time.time()
    state["node_sequence"].append("refine")

    llm = get_llm()

    refine_prompt = PromptTemplate.from_template("""
La búsqueda médica anterior no dio buenos resultados (relevancia baja).

PREGUNTA ORIGINAL: {question}

DOCUMENTOS RECUPERADOS (poco relevantes):
{doc_snippets}

TAREA: Reformula la pregunta usando terminología médica más específica y precisa.
La nueva consulta debe:
- Ser más clara y directa
- Usar términos clínicos estándar
- Enfocarse en aspectos clave
- Evitar ambigüedades

CONSULTA REFINADA (solo la pregunta médica reformulada):""")

    try:
        # Tomar snippets de documentos
        doc_snippets = "\n".join([
            doc.page_content[:200]
            for doc in state["reranked_docs"][:2]
        ]) if state["reranked_docs"] else "No se encontraron documentos relevantes"

        prompt = refine_prompt.format(
            question=state["question"],
            doc_snippets=doc_snippets
        )

        response = llm.invoke(prompt)
        refined_question = response.content.strip()

        # Actualizar pregunta solo si el refinamiento es sustancial
        if len(refined_question) >= 10 and refined_question.lower() != state["question"].lower():
            state["question"] = refined_question
            print(f"🔧 Consulta refinada: {refined_question[:100]}...")
        else:
            print(f"⚠️ Refinamiento no sustancial, manteniendo pregunta original")

    except Exception as e:
        print(f"❌ Error refinando consulta: {str(e)}")

    # Incrementar contador de iteración
    state["retrieval_iteration"] += 1

    state["latency_ms"]["refine"] = (time.time() - start_time) * 1000
    return state


def generate_response(state: RAGState) -> RAGState:
    """
    Nodo 7: Genera la respuesta final con contexto conversacional
    """
    start_time = time.time()
    state["node_sequence"].append("generate")

    llm = get_llm()

    # Formatear documentos
    def format_docs(docs):
        if not docs:
            return "No se encontró información relevante."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Desconocido')
            route = doc.metadata.get('route', 'unknown')

            if route == 'sql':
                formatted.append(f"[FUENTE SQL {i+1}] Base de datos:\n{doc.page_content}")
            else:
                page = doc.metadata.get('page', '')
                page_info = f" (p. {page})" if page else ""
                formatted.append(f"[FUENTE {i+1}] {source}{page_info}:\n{doc.page_content[:800]}")

        return "\n\n".join(formatted)

    # Formatear historial
    history_text = ""
    if state.get("chat_history") and len(state["chat_history"]) > 0:
        history_text = "CONVERSACIÓN PREVIA:\n" + "\n".join([
            f"Usuario: {q}\nAsistente: {a[:200]}..."
            for q, a in state["chat_history"][-2:]
        ]) + "\n\n"

    response_prompt = PromptTemplate.from_template("""
Eres un asistente médico especializado en español que responde preguntas clínicas.

{history}INSTRUCCIONES IMPORTANTES:
{policy_instructions}

CONTEXTO RECUPERADO:
Método: {retrieval_method}
Confianza: {relevance_score:.2f}

{documents}

PREGUNTA ACTUAL: {question}

INSTRUCCIONES:
- Responde en español claro y profesional
- Usa SOLO la información de las fuentes proporcionadas
- Estructura la respuesta en puntos claros
- Cita las fuentes específicas
- Si la información es insuficiente, indícalo claramente
- Para resultados SQL, presenta los datos estructuradamente
- Añade el disclaimer médico al final

RESPUESTA:""")

    try:
        prompt = response_prompt.format(
            history=history_text,
            policy_instructions=DEFAULT_POLICY.get_instructions(),
            retrieval_method=state.get("retrieval_method", "unknown"),
            relevance_score=state.get("relevance_score", 0.0),
            documents=format_docs(state["reranked_docs"]),
            question=state["original_question"]  # Usar pregunta original para el usuario
        )

        response = llm.invoke(prompt)

        # Aplicar políticas
        final_response = format_response_with_policy(
            response.content,
            state["reranked_docs"],
            state["route"],
            DEFAULT_POLICY
        )

        state["response"] = final_response

        # Actualizar memoria conversacional
        if "chat_history" not in state or state["chat_history"] is None:
            state["chat_history"] = []

        state["chat_history"].append((
            state["original_question"],
            final_response
        ))

        print(f"✅ Respuesta generada ({len(final_response)} caracteres)")

    except Exception as e:
        print(f"❌ Error generando respuesta: {str(e)}")
        state["response"] = f"Error generando respuesta: {str(e)}\n\n{DEFAULT_POLICY.disclaimer}"
        state["error"] = f"Generation error: {str(e)}"

    state["latency_ms"]["generate"] = (time.time() - start_time) * 1000
    return state


# ============================================================================
# DECISIONES CONDICIONALES
# ============================================================================

def should_refine_query(state: RAGState) -> Literal["refine", "generate"]:
    """
    Decide si refinar la query o proceder a generar
    """
    # Límite de 2 iteraciones
    if state.get("retrieval_iteration", 0) >= 2:
        print("⚠️ Máximo de iteraciones alcanzado, procediendo a generar")
        return "generate"

    # Si necesita refinamiento y relevancia muy baja
    if state.get("needs_refinement", False) and state.get("relevance_score", 0) < 0.4:
        print("🔧 Relevancia muy baja, refinando consulta...")
        return "refine"

    # Si tiene documentos suficientes, generar
    if state.get("can_answer", False):
        return "generate"

    # Fallback: generar con lo que hay
    return "generate"


# ============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================================

def create_rag_graph():
    """
    Crea y compila el grafo de LangGraph
    """
    # Crear grafo
    workflow = StateGraph(RAGState)

    # Añadir nodos
    workflow.add_node("contextualize", contextualize_question)
    workflow.add_node("route", route_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("evaluate", evaluate_relevance)
    workflow.add_node("refine", refine_query)
    workflow.add_node("generate", generate_response)

    # Definir flujo
    workflow.set_entry_point("contextualize")
    workflow.add_edge("contextualize", "route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "evaluate")

    # Edge condicional: ¿refinar o generar?
    workflow.add_conditional_edges(
        "evaluate",
        should_refine_query,
        {
            "refine": "refine",
            "generate": "generate"
        }
    )

    # Ciclo: refinamiento → nueva búsqueda
    workflow.add_edge("refine", "retrieve")

    # Fin
    workflow.add_edge("generate", END)

    # Configurar checkpointing
    # Crear directorio si no existe
    from pathlib import Path
    Path("./db").mkdir(parents=True, exist_ok=True)

    # Inicializar checkpointer correctamente
    import sqlite3
    conn = sqlite3.connect("./db/langgraph_checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Compilar
    app = workflow.compile(checkpointer=checkpointer)

    return app


# ============================================================================
# INSTANCIA GLOBAL
# ============================================================================

# Crear la aplicación de grafo
rag_app = create_rag_graph()


# ============================================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================================

def invoke_rag(
    question: str,
    conversation_id: str = None,
    chat_history: list[tuple[str, str]] = None,
    retrieval_mode: str = "standard"
) -> dict:
    """
    Función de conveniencia para invocar el RAG

    Args:
        question: Pregunta del usuario
        conversation_id: ID de conversación (para checkpointing)
        chat_history: Historial de conversación [(user, assistant), ...]
        retrieval_mode: Modo de recuperación

    Returns:
        Estado final con la respuesta
    """
    import uuid

    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    if chat_history is None:
        chat_history = []

    # Estado inicial
    initial_state = {
        "question": question,
        "original_question": question,
        "conversation_id": conversation_id,
        "messages": [],
        "chat_history": chat_history,
        "route": "",
        "route_confidence": 0.0,
        "route_reasoning": "",
        "documents": [],
        "retrieval_method": retrieval_mode,
        "retrieval_iteration": 0,
        "reranked_docs": [],
        "relevance_score": 0.0,
        "can_answer": False,
        "needs_refinement": False,
        "response": "",
        "error": None,
        "latency_ms": {},
        "node_sequence": []
    }

    # Configuración con thread_id para checkpointing
    config = {"configurable": {"thread_id": conversation_id}}

    # Ejecutar grafo
    final_state = rag_app.invoke(initial_state, config)

    return final_state
