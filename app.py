import streamlit as st
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import time
import traceback

from src.graph import build_rag_chain
from src.retrievers import build_vectorstore
from src.tools import get_sql_chain, run_sql, create_sample_icpc_db
from src.policies import DEFAULT_POLICY
from src.evaluation import evaluate_ndcg, load_test_dataset, calculate_recall
from src.ingest import main as ingest_main

st.set_page_config(
    page_title="RAG NextHealth", 
    layout="wide",
    page_icon="🧭"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧭 RAG NextHealth")
st.caption("Sistema RAG avanzado para búsqueda clínica en español con routing inteligente")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuración")

    # Directories and paths
    persist_dir = st.text_input("ChromaDB Directory", "./db/chroma")
    sqlite_path = st.text_input("SQLite Path (ICPC-3)", "./db/icpc3.db")
    data_dir = st.text_input("Data Directory", "./docs")

    st.divider()

    # API Configuration
    st.subheader("🔑 API Key")

    # Load API key from .env
    from dotenv import load_dotenv

    # Check if button was pressed
    if 'load_env_key' not in st.session_state:
        st.session_state.load_env_key = False

    col_api1, col_api2 = st.columns([4, 1])
    with col_api2:
        if st.button("API", help="Cargar API key desde .env", use_container_width=True):
            st.session_state.load_env_key = True

    # Load from .env if button pressed
    default_key = ""
    if st.session_state.load_env_key:
        load_dotenv(override=True)
        default_key = os.getenv("OPENAI_API_KEY", "")
        if default_key:
            st.session_state.load_env_key = False  # Reset flag

    with col_api1:
        openai_key = st.text_input(
            "OPENAI_API_KEY",
            value=default_key,
            type="password",
            help="Ingresa tu API key o pulsa 'API' para cargar desde .env"
        )

    emb_model = st.selectbox(
        "Modelo de Embeddings",
        ["intfloat/multilingual-e5-base", "sentence-transformers/all-MiniLM-L6-v2"],
        help="Modelo para generar embeddings"
    )
    
    st.divider()
    
    # System status
    st.subheader("📊 Estado del Sistema")
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("✅ OpenAI API configurada")
    else:
        st.warning("⚠️ OpenAI API no configurada")
    
    os.environ["EMBEDDINGS_MODEL"] = emb_model
    
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
tab1, tab2, tab3 = st.tabs(["🔍 Búsqueda RAG", "🗃️ Consulta SQL", "📈 Evaluación"])

with tab1:
    st.header("Búsqueda RAG con Routing Inteligente")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
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
                help="Incremental procesa solo documentos nuevos o modificados. Reindexar Todo borra el índice completo y procesa todos los documentos.",
                horizontal=True
            )
            
            if st.button("🚀 Ejecutar Ingesta", help=f"Procesa documentos en modo {ingestion_mode}"):
                if not Path(data_dir).exists():
                    st.error(f"Directory {data_dir} no existe")
                else:
                    with st.spinner(f"Procesando documentos ({ingestion_mode})..."):
                        try:
                            progress_bar = st.progress(0)
                            progress_bar.progress(25)
                            
                            # Run ingestion with selected mode
                            ingest_main(persist_dir, data_dir, mode=ingestion_mode)
                            
                            progress_bar.progress(100)
                            st.success(f"✅ Ingesta {ingestion_mode} completada exitosamente")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error durante la ingesta: {str(e)}")
                            st.code(traceback.format_exc())
        
        # Query section
        st.subheader("Consulta")
        query = st.text_area(
            "Escribe tu pregunta clínica:",
            height=100,
            placeholder="Ej: ¿Cuáles son los criterios para el diagnóstico de diabetes tipo 2 según las guías clínicas?"
        )
        
        col_search1, col_search2 = st.columns([2, 1])
        with col_search1:
            retrieval_mode = st.selectbox(
                "Modo de recuperación:",
                options=["standard", "raptor", "rag-fusion", "hyde", "crag", "hybrid"],
                format_func=lambda x: {
                    "standard": "🔹 Estándar (Multi-query + Reranking)",
                    "raptor": "🌳 RAPTOR (Jerárquico)",
                    "rag-fusion": "🔀 RAG-Fusion (RRF)",
                    "hyde": "💡 HYDE (Hipotético)",
                    "crag": "🔄 CRAG (Corrección automática)",
                    "hybrid": "⚡ Híbrido (Fusion + HYDE)"
                }[x],
                help="Selecciona la estrategia de recuperación de documentos"
            )
        with col_search2:
            search_button = st.button("🔍 Buscar", type="primary", use_container_width=True)
        
        if search_button and query:
            if not openai_key:
                st.error("⚠️ Se requiere OpenAI API Key para realizar búsquedas")
            elif not Path(persist_dir).exists():
                st.error("⚠️ Vector store no encontrado. Ejecuta la ingesta primero.")
            else:
                try:
                    with st.spinner(f"Buscando información relevante ({retrieval_mode})..."):
                        start_time = time.time()
                        
                        # Build RAG chain with selected retrieval mode
                        rag_chain = build_rag_chain(persist_dir, sqlite_path, retrieval_mode=retrieval_mode)
                        
                        # Execute query
                        response = rag_chain.invoke({"question": query})
                        
                        end_time = time.time()
                        latency = end_time - start_time
                        
                        # Display results
                        st.subheader("📋 Respuesta")
                        st.write(response.content)
                        
                        # Display disclaimer
                        st.markdown(f"""
                        <div class="disclaimer">
                            <strong>⚠️ Aviso:</strong> {DEFAULT_POLICY.disclaimer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Performance metrics
                        with st.expander("📊 Métricas de Rendimiento"):
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Latencia Total", f"{latency:.2f}s")
                            with col_m2:
                                st.metric("Modelo", "gpt-5")
                            with col_m3:
                                st.metric("Estado", "✅ Completado")
                
                except Exception as e:
                    st.error(f"Error durante la búsqueda: {str(e)}")
                    with st.expander("🔧 Detalles del Error"):
                        st.code(traceback.format_exc())
    
    with col2:
        st.subheader("📋 Información")
        st.info("""
        **Técnicas RAG Avanzadas:**
        - RAPTOR: sumarios jerárquicos
        - RAG-Fusion: fusión RRF
        - HYDE: docs hipotéticos
        - CRAG: corrección automática
        - Multi-query retrieval
        - Re-ranking con BGE
        - Routing SQL automático
        """)
        
        st.subheader("🔧 Tips")
        st.write("""
        - Usa preguntas específicas
        - Menciona códigos ICPC-3 para routing SQL
        - Las respuestas incluyen fuentes
        """)

with tab2:
    st.header("Consultas SQL Estructuradas")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create sample database if not exists
        if not Path(sqlite_path).exists():
            st.warning("⚠️ Base de datos no encontrada. Creando base de ejemplo...")
            if st.button("🗃️ Crear Base de Datos ICPC-3 de Ejemplo"):
                try:
                    create_sample_icpc_db(sqlite_path)
                    st.success("✅ Base de datos creada exitosamente")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creando la base de datos: {str(e)}")
        
        # SQL Query interface
        st.subheader("Consulta Natural a SQL")
        
        sql_query = st.text_area(
            "Pregunta sobre códigos ICPC-3 o mapeos:",
            height=80,
            placeholder="Ej: ¿Qué código ICPC-3 corresponde a diabetes mellitus tipo 2?"
        )
        
        if st.button("🔄 Convertir a SQL", type="primary"):
            if not openai_key:
                st.error("⚠️ Se requiere OpenAI API Key")
            elif not Path(sqlite_path).exists():
                st.error("⚠️ Base de datos SQL no encontrada")
            else:
                try:
                    with st.spinner("Generando consulta SQL..."):
                        sql_chain, db = get_sql_chain(sqlite_path)
                        generated_sql = sql_chain.invoke({"question": sql_query})
                        
                        st.subheader("🔍 SQL Generado")
                        st.code(generated_sql, language="sql")
                        
                        # Execute SQL
                        if st.button("▶️ Ejecutar SQL"):
                            try:
                                df = run_sql(sqlite_path, generated_sql)
                                st.subheader("📊 Resultados")
                                st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error ejecutando SQL: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error generando SQL: {str(e)}")
        
        # Direct SQL interface
        st.divider()
        st.subheader("SQL Directo")
        direct_sql = st.text_area(
            "Ejecuta SQL directamente:",
            height=100,
            placeholder="SELECT * FROM icpc_codes LIMIT 10;"
        )
        
        if st.button("▶️ Ejecutar SQL Directo"):
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
                st.write("**Tablas disponibles:**")
                for table in tables_df['name']:
                    st.write(f"- {table}")
            except:
                st.write("No se pudo cargar el esquema")
        else:
            st.write("Base de datos no disponible")

with tab3:
    st.header("Evaluación de Calidad RAG")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Métricas de Evaluación")
        
        # Upload test dataset
        uploaded_file = st.file_uploader(
            "Subir dataset de test (CSV)",
            type=['csv'],
            help="CSV con columnas: question, relevant_docs (separados por ;)"
        )
        
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset cargado: {len(test_df)} preguntas")
                st.dataframe(test_df.head(), use_container_width=True)
                
                if st.button("🚀 Ejecutar Evaluación", type="primary"):
                    if not openai_key or not Path(persist_dir).exists():
                        st.error("⚠️ Requiere OpenAI API y vector store")
                    else:
                        try:
                            with st.spinner("Ejecutando evaluación..."):
                                progress_bar = st.progress(0)
                                
                                # Build RAG chain for evaluation
                                rag_chain = build_rag_chain(persist_dir)
                                vectorstore = build_vectorstore(persist_dir)
                                
                                results = []
                                total_questions = len(test_df)
                                
                                for idx, row in test_df.iterrows():
                                    question = row['question']
                                    expected_docs = row['relevant_docs'].split(';') if pd.notna(row['relevant_docs']) else []
                                    
                                    # Retrieve documents
                                    retrieved_docs = vectorstore.similarity_search(question, k=10)
                                    retrieved_doc_ids = [f"doc_{i}" for i in range(len(retrieved_docs))]  # Placeholder
                                    
                                    # Calculate metrics
                                    ndcg = evaluate_ndcg(expected_docs, retrieved_doc_ids, k=5)
                                    recall = calculate_recall(expected_docs, retrieved_doc_ids, k=5)
                                    
                                    results.append({
                                        'question': question,
                                        'ndcg@5': ndcg,
                                        'recall@5': recall,
                                        'retrieved_docs': len(retrieved_docs)
                                    })
                                    
                                    progress_bar.progress((idx + 1) / total_questions)
                                
                                # Display results
                                results_df = pd.DataFrame(results)
                                
                                st.subheader("📈 Resultados de Evaluación")
                                
                                # Summary metrics
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                with col_m1:
                                    st.metric("nDCG@5 Promedio", f"{results_df['ndcg@5'].mean():.3f}")
                                with col_m2:
                                    st.metric("Recall@5 Promedio", f"{results_df['recall@5'].mean():.3f}")
                                with col_m3:
                                    st.metric("Total Preguntas", len(results_df))
                                with col_m4:
                                    st.metric("Docs Promedio", f"{results_df['retrieved_docs'].mean():.1f}")
                                
                                # Detailed results
                                st.subheader("📋 Resultados Detallados")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Charts
                                st.subheader("📊 Distribución de Métricas")
                                col_chart1, col_chart2 = st.columns(2)
                                with col_chart1:
                                    st.bar_chart(results_df.set_index('question')['ndcg@5'])
                                with col_chart2:
                                    st.bar_chart(results_df.set_index('question')['recall@5'])
                        
                        except Exception as e:
                            st.error(f"Error durante la evaluación: {str(e)}")
                            st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Error cargando el dataset: {str(e)}")
        
        else:
            st.info("📁 Sube un archivo CSV con el formato: question, relevant_docs")
            
            # Sample dataset format
            st.subheader("📋 Formato de Dataset")
            sample_data = pd.DataFrame({
                'question': [
                    '¿Cuáles son los criterios de diagnóstico para diabetes?',
                    '¿Qué código ICPC-3 corresponde a hipertensión?'
                ],
                'relevant_docs': [
                    'diabetes_criteria.pdf;clinical_guide.pdf',
                    'icpc3_codes.csv;hypertension_guide.pdf'
                ]
            })
            st.dataframe(sample_data, use_container_width=True)
    
    with col2:
        st.subheader("📚 Métricas Explicadas")
        
        with st.expander("nDCG@k"):
            st.write("""
            **Normalized Discounted Cumulative Gain**
            - Mide la calidad del ranking
            - Valores entre 0 y 1
            - Más alto = mejor ranking
            """)
        
        with st.expander("Recall@k"):
            st.write("""
            **Recall at k**
            - % de documentos relevantes encontrados
            - Entre 0 y 1
            - Más alto = mejor cobertura
            """)
        
        with st.expander("Latencia"):
            st.write("""
            **Tiempo de respuesta**
            - Tiempo total de procesamiento
            - Incluye retrieval + generación
            - Meta: < 2 segundos
            """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    RAG NextHealth v1.0 | Sistema de búsqueda clínica inteligente<br>
    Cumple con regulaciones AI Act/MDR - Solo información, no diagnósticos
</div>
""", unsafe_allow_html=True)
