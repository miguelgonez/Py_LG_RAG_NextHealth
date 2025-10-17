"""
Aplicación Streamlit con LangGraph RAG + Memoria Conversacional
Versión mejorada con grafo de estado, Self-RAG y CRAG con ciclos
"""

import streamlit as st
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import time
import traceback
import uuid

from src.langgraph_rag import invoke_rag
from src.tools import create_sample_icpc_db, run_sql
from src.policies import DEFAULT_POLICY
from src.evaluation import evaluate_ndcg, load_test_dataset, calculate_recall
from src.ingest import main as ingest_main

st.set_page_config(
    page_title="RAG NextHealth (LangGraph)",
    layout="wide",
    page_icon="🧠"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .node-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        background-color: #e0e0e0;
        font-size: 0.75rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧠 RAG NextHealth - LangGraph Edition")
st.caption("Sistema RAG avanzado con memoria conversacional, Self-RAG y CRAG con ciclos")
st.markdown('</div>', unsafe_allow_html=True)

# Inicializar session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuración")

    # Directories and paths
    persist_dir = st.text_input("ChromaDB Directory", "./db/chroma")
    sqlite_path = st.text_input("SQLite Path (ICPC-3)", "./db/icpc3.db")
    data_dir = st.text_input("Data Directory", "./docs")

    st.divider()

    # API Configuration
    openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Requerido para LLM y generación")
    emb_model = st.selectbox(
        "Modelo de Embeddings",
        ["intfloat/multilingual-e5-base", "sentence-transformers/all-MiniLM-L6-v2"],
        help="Modelo para generar embeddings"
    )

    st.divider()

    # Session management
    st.subheader("💬 Sesión Conversacional")
    st.info(f"ID: {st.session_state.conversation_id[:8]}...")
    st.write(f"Mensajes: {len(st.session_state.messages)}")

    if st.button("🔄 Nueva Conversación"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # System status
    st.subheader("📊 Estado del Sistema")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("✅ OpenAI API configurada")
    else:
        st.warning("⚠️ OpenAI API no configurada")

    os.environ["EMBEDDINGS_MODEL"] = emb_model
    os.environ["PERSIST_DIR"] = persist_dir
    os.environ["SQLITE_PATH"] = sqlite_path

    # Check if vector store exists
    if Path(persist_dir).exists():
        st.success("✅ Vector store disponible")
    else:
        st.warning("⚠️ Vector store no encontrado")

    # Check if SQLite DB exists
    if Path(sqlite_path).exists():
        st.success("✅ Base de datos SQL disponible")
    else:
        st.warning("⚠️ Base de datos SQL no encontrada")

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat RAG", "🗃️ SQL Console", "📈 Evaluación"])

with tab1:
    st.header("Chat Conversacional con Memoria")

    col_main, col_info = st.columns([3, 1])

    with col_main:
        # Ingestion section
        with st.expander("📥 Ingesta de Documentos", expanded=False):
            st.write("Indexar documentos en el vector store")

            ingestion_mode = st.radio(
                "Modo de ingesta:",
                options=["incremental", "full"],
                format_func=lambda x: {
                    "incremental": "🔄 Incremental (solo nuevos/modificados)",
                    "full": "♻️ Reindexar Todo (borrar y reprocesar)"
                }[x],
                horizontal=True
            )

            if st.button("🚀 Ejecutar Ingesta"):
                if not Path(data_dir).exists():
                    st.error(f"Directory {data_dir} no existe")
                else:
                    with st.spinner(f"Procesando documentos ({ingestion_mode})..."):
                        try:
                            progress_bar = st.progress(0)
                            progress_bar.progress(25)

                            ingest_main(persist_dir, data_dir, mode=ingestion_mode)

                            progress_bar.progress(100)
                            st.success(f"✅ Ingesta {ingestion_mode} completada")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.code(traceback.format_exc())

        # Display chat history
        st.subheader("Conversación")

        # Chat container
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 Usuario:</strong><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Asistente:</strong><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)

                    # Mostrar metadata si está disponible
                    if "metadata" in msg:
                        with st.expander("🔍 Detalles técnicos", expanded=False):
                            meta = msg["metadata"]

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Ruta", meta.get("route", "N/A"))
                            with col2:
                                st.metric("Relevancia", f"{meta.get('relevance_score', 0):.2f}")
                            with col3:
                                st.metric("Iteraciones", meta.get("retrieval_iteration", 0))

                            # Secuencia de nodos
                            if "node_sequence" in meta:
                                st.write("**Flujo de nodos:**")
                                nodes_html = " → ".join([
                                    f'<span class="node-badge">{node}</span>'
                                    for node in meta["node_sequence"]
                                ])
                                st.markdown(nodes_html, unsafe_allow_html=True)

                            # Latencias
                            if "latency_ms" in meta:
                                st.write("**Latencias por nodo (ms):**")
                                latency_df = pd.DataFrame([
                                    {"Nodo": k, "Tiempo (ms)": f"{v:.1f}"}
                                    for k, v in meta["latency_ms"].items()
                                ])
                                st.dataframe(latency_df, use_container_width=True)

        # Query input
        st.divider()

        col_input, col_mode = st.columns([3, 1])

        with col_mode:
            retrieval_mode = st.selectbox(
                "Estrategia:",
                options=["standard", "raptor", "rag-fusion", "hyde", "crag"],
                format_func=lambda x: {
                    "standard": "🔹 Estándar",
                    "raptor": "🌳 RAPTOR",
                    "rag-fusion": "🔀 RAG-Fusion",
                    "hyde": "💡 HYDE",
                    "crag": "🔄 CRAG"
                }[x]
            )

        with col_input:
            query = st.text_input(
                "Escribe tu pregunta:",
                placeholder="Ej: ¿Cuáles son los síntomas de la diabetes tipo 2?",
                key="query_input"
            )

        if st.button("📤 Enviar", type="primary", use_container_width=True) and query:
            if not openai_key:
                st.error("⚠️ Se requiere OpenAI API Key")
            elif not Path(persist_dir).exists():
                st.error("⚠️ Vector store no encontrado. Ejecuta la ingesta primero.")
            else:
                # Agregar mensaje del usuario
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })

                # Mostrar progreso
                with st.spinner("🤔 Procesando con LangGraph..."):
                    try:
                        # Configurar modo de recuperación
                        os.environ["RETRIEVAL_MODE"] = retrieval_mode

                        # Ejecutar RAG con LangGraph
                        start_time = time.time()

                        final_state = invoke_rag(
                            question=query,
                            conversation_id=st.session_state.conversation_id,
                            chat_history=st.session_state.chat_history,
                            retrieval_mode=retrieval_mode
                        )

                        end_time = time.time()
                        total_latency = (end_time - start_time) * 1000

                        # Extraer respuesta
                        response = final_state.get("response", "Error: No se generó respuesta")

                        # Actualizar historial de sesión
                        st.session_state.chat_history = final_state.get("chat_history", st.session_state.chat_history)

                        # Agregar mensaje del asistente
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "metadata": {
                                "route": final_state.get("route", "unknown"),
                                "relevance_score": final_state.get("relevance_score", 0.0),
                                "retrieval_iteration": final_state.get("retrieval_iteration", 0),
                                "retrieval_method": final_state.get("retrieval_method", "unknown"),
                                "node_sequence": final_state.get("node_sequence", []),
                                "latency_ms": final_state.get("latency_ms", {}),
                                "total_latency_ms": total_latency,
                                "error": final_state.get("error")
                            }
                        })

                        # Recargar para mostrar el mensaje
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.code(traceback.format_exc())

    with col_info:
        st.subheader("ℹ️ Información")

        st.info("""
        **LangGraph Features:**
        - ✅ Memoria conversacional
        - ✅ Self-RAG (auto-evaluación)
        - ✅ CRAG con ciclos (hasta 2 intentos)
        - ✅ Checkpointing automático
        - ✅ Flujo visual de nodos
        """)

        st.subheader("💡 Tips")
        st.write("""
        - Puedes hacer preguntas de seguimiento
        - El sistema recuerda el contexto
        - Si la relevancia es baja, refinará automáticamente
        - Máximo 2 iteraciones de refinamiento
        """)

        st.subheader("🔧 Nodos del Grafo")
        nodes = [
            "1. Contextualizar",
            "2. Routing",
            "3. Retrieve",
            "4. Rerank",
            "5. Evaluate",
            "6. Refine (condicional)",
            "7. Generate"
        ]
        for node in nodes:
            st.write(f"• {node}")

with tab2:
    st.header("SQL Console")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create sample database if not exists
        if not Path(sqlite_path).exists():
            st.warning("⚠️ Base de datos no encontrada.")
            if st.button("🗃️ Crear Base de Datos ICPC-3"):
                try:
                    create_sample_icpc_db(sqlite_path)
                    st.success("✅ Base de datos creada")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # SQL Query interface
        st.subheader("Consulta SQL Directa")

        direct_sql = st.text_area(
            "Ejecuta SQL:",
            height=100,
            placeholder="SELECT * FROM icpc_codes LIMIT 10;"
        )

        if st.button("▶️ Ejecutar SQL"):
            if not Path(sqlite_path).exists():
                st.error("⚠️ Base de datos no encontrada")
            else:
                try:
                    df = run_sql(sqlite_path, direct_sql)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        st.subheader("🗃️ Esquema DB")
        if Path(sqlite_path).exists():
            try:
                tables_df = run_sql(sqlite_path, "SELECT name FROM sqlite_master WHERE type='table';")
                st.write("**Tablas:**")
                for table in tables_df['name']:
                    st.write(f"- {table}")
            except:
                st.write("No se pudo cargar")

with tab3:
    st.header("Evaluación de Calidad RAG")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Métricas")

        uploaded_file = st.file_uploader(
            "Subir dataset de test (CSV)",
            type=['csv'],
            help="CSV con columnas: question, relevant_docs"
        )

        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset: {len(test_df)} preguntas")
                st.dataframe(test_df.head(), use_container_width=True)

                if st.button("🚀 Ejecutar Evaluación"):
                    st.info("La evaluación con LangGraph está en desarrollo...")

            except Exception as e:
                st.error(f"Error: {str(e)}")

        else:
            st.info("📁 Sube un CSV con: question, relevant_docs")

    with col2:
        st.subheader("📚 Métricas")

        with st.expander("nDCG@k"):
            st.write("""
            **Normalized Discounted Cumulative Gain**
            - Calidad del ranking
            - 0-1 (mayor = mejor)
            """)

        with st.expander("Recall@k"):
            st.write("""
            **Recall at k**
            - % docs relevantes encontrados
            - 0-1 (mayor = mejor)
            """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    RAG NextHealth v2.0 - LangGraph Edition<br>
    Memoria conversacional • Self-RAG • CRAG con ciclos • Checkpointing
</div>
""", unsafe_allow_html=True)
