# RAG NextHealth

Sistema avanzado de búsqueda clínica en español con técnicas RAG avanzadas, re-ranking BGE, y guardrails de cumplimiento médico.

## 🚀 Características Principales

### 🔍 Sistema RAG Avanzado
- **Técnicas de Recuperación Múltiples**:
  - RAG-Fusion: Combina múltiples variaciones de consultas con Reciprocal Rank Fusion
  - HYDE: Genera documentos hipotéticos para búsqueda semántica mejorada
  - RAPTOR: Recuperación jerárquica con clustering KMeans
  - CRAG: RAG correctivo con evaluación de relevancia automática
  
- **Re-ranking Inteligente**:
  - Modelo BGE cross-encoder (`bge-reranker-base`)
  - Recuperación inicial: k=8 documentos
  - Selección final: Top k=4 documentos más relevantes

### 📚 Ingesta Incremental de Documentos
- **Detección Automática de Cambios**: Sistema basado en SQLite con hashing SHA256
- **Prevención de Duplicados**: Elimina chunks antiguos automáticamente al actualizar archivos
- **Limpieza Inteligente**: Remueve embeddings cuando se eliminan archivos fuente
- **Modos de Ingesta**:
  - Incremental (por defecto): Solo procesa archivos nuevos/modificados
  - Full Reindex: Reprocesa todo el corpus desde cero
- **Formatos Soportados**: PDF, HTML, TXT, MD

### 🧭 Routing Inteligente
- **Clasificación Automática de Consultas**:
  - Vectorstore: Preguntas semánticas/contenido
  - SQL: Consultas estructuradas (códigos ICPC-3, mapeos ICD-10)
  - Hybrid: Requieren ambos enfoques
- **Text-to-SQL**: Conversión de lenguaje natural a SQL para códigos clínicos

### 🛡️ Guardrails Clínicos (MDR/AI Act)
- **Políticas de Seguridad Estrictas**:
  - ❌ No diagnósticos
  - ❌ No dosis de medicamentos
  - ❌ No consejos médicos
  - ✅ Solo información educativa
- **Citación Obligatoria**: Mínimo 2 fuentes por respuesta
- **Disclaimer Automático**: En todas las respuestas
- **Límites de Longitud**: Máximo 2000 caracteres

### 📊 Framework de Evaluación
- **Métricas Implementadas**:
  - nDCG@k (Normalized Discounted Cumulative Gain)
  - Recall@k
  - Precision@k
  - Latency tracking por etapa
- **Análisis de Rendimiento**: Identificación de consultas de bajo rendimiento

### 🌐 Interfaz Streamlit (3 Pestañas)
1. **RAG Search**: Búsqueda con configuración de k, umbral de similitud, modo de recuperación
2. **SQL Console**: Editor de consultas, templates, paginación de resultados
3. **Evaluation Dashboard**: Carga de CSV, visualización de métricas, análisis de desempeño

## 📋 Requisitos del Sistema

- Python 3.8+
- 4GB+ RAM (recomendado para modelos de embeddings)
- 2GB+ espacio en disco (vectorstore y modelos)
- OpenAI API Key

## 🛠️ Instalación

### Dependencias Python
```bash
pip install streamlit langchain langchain-community openai chromadb \
    sentence-transformers flagembedding pandas numpy scikit-learn \
    pypdf unstructured tiktoken sqlalchemy pydantic
```

### Variables de Entorno
```bash
# Requerido
export OPENAI_API_KEY="sk-..."

# Opcional
export OPENAI_CHAT_MODEL="gpt-5"  # Default
export RERANKER_MODEL="BAAI/bge-reranker-base"  # Default
```

## 🚀 Uso

### Iniciar la Aplicación
```bash
streamlit run app.py --server.port 5000
```

### Cargar Documentos
1. Coloca tus documentos clínicos en la carpeta `./docs`
2. En la pestaña "RAG Search", haz clic en "Procesar Documentos"
3. Selecciona el modo:
   - **Incremental**: Solo procesa cambios (recomendado)
   - **Full Reindex**: Reprocesa todo

### Realizar Búsquedas
1. Selecciona el modo de recuperación (Standard, Multi-query, RAG-Fusion, HYDE, RAPTOR, CRAG)
2. Ajusta k (número de documentos) y umbral de similitud
3. Ingresa tu consulta en español
4. Revisa la respuesta con fuentes citadas

### Consultas SQL
1. Ve a la pestaña "SQL Console"
2. Escribe tu consulta en lenguaje natural o SQL directo
3. Ejemplos: "códigos ICPC-3 para diabetes", "mapeo ICD-10 para K86"

### Evaluación
1. Prepara un CSV con columnas: `question`, `relevant_docs`
2. Carga el archivo en la pestaña "Evaluation"
3. Revisa métricas nDCG, Recall, Precision y latencia

## 📁 Estructura del Proyecto

```
.
├── app.py                    # Aplicación principal Streamlit
├── src/
│   ├── ingest.py            # Ingesta de documentos y vectorstore
│   ├── state_store.py       # Tracking de estado incremental
│   ├── retrievers.py        # Multi-query, fusion, RAPTOR
│   ├── reranker.py          # BGE re-ranking
│   ├── routing.py           # Routing inteligente
│   ├── tools.py             # Text-to-SQL, utilidades DB
│   ├── graph.py             # Orquestación LangChain
│   ├── policies.py          # Guardrails clínicos
│   ├── evaluation.py        # Métricas nDCG, recall
│   ├── advanced_rag.py      # RAG-Fusion, HYDE
│   ├── raptor.py            # RAPTOR jerárquico
│   └── crag.py              # CRAG correctivo
├── docs/                     # Directorio de documentos clínicos
├── .streamlit/
│   └── config.toml          # Configuración Streamlit
└── README.md
```

## 🏗️ Arquitectura

### Pipeline RAG
1. **Ingesta**: Documentos → Chunking → Embeddings → ChromaDB
2. **Routing**: Query → Clasificación → Vectorstore/SQL/Hybrid
3. **Recuperación**: Query → Multi-variaciones → Retrieval (k=8)
4. **Re-ranking**: BGE cross-encoder → Top k=4
5. **Generación**: LLM + Contexto + Guardrails → Respuesta

### Detección de Cambios
- SHA256 hash del contenido del archivo
- Comparación de tamaño y mtime
- SQLite state store en `./db/document_state.db`
- IDs determinísticos: `{file_hash}_{chunk_index}`

## 🛡️ Cumplimiento Regulatorio

Sistema diseñado para cumplir con:
- **MDR (Medical Device Regulation)**: Información médica sin diagnósticos
- **AI Act**: Transparencia y limitaciones claras
- **GDPR**: No almacenamiento de datos personales

**⚠️ IMPORTANTE**: Este sistema es solo para fines informativos y educativos. NO proporciona diagnósticos, tratamientos ni consejos médicos. Siempre consulte a un profesional de la salud calificado.

## 📄 Licencia

MIT License

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de enviar un pull request.

## 💬 Soporte

Para preguntas o problemas, abre un issue en GitHub
