# 🧠 RAG NextHealth - LangGraph Edition

## Guía Completa de LangGraph Integration

### 🚀 ¿Qué es nuevo?

Esta versión mejorada de RAG NextHealth integra **LangGraph** para orquestación avanzada del flujo RAG, añadiendo:

1. **Memoria Conversacional**: Mantiene contexto entre preguntas
2. **Self-RAG**: Auto-evaluación de la calidad de respuestas
3. **CRAG Mejorado**: Ciclos de corrección automática (hasta 2 iteraciones)
4. **Checkpointing**: Recuperación automática ante fallos
5. **Flujo Visual**: Trazabilidad completa del proceso

---

## 📊 Arquitectura del Grafo

```
User Query
    ↓
[Contextualize] → Reformula con historial conversacional
    ↓
[Route] → Decide: SQL o Vector?
    ↓
[Retrieve] → Recupera documentos (Fusion/RAPTOR/HYDE/CRAG)
    ↓
[Rerank] → BGE cross-encoder (top 4)
    ↓
[Evaluate] → ¿Relevancia suficiente?
    ↓ (NO, relevancia < 0.4)
[Refine] → Reformula query médica
    ↓
[Retrieve] → Segundo intento ← CICLO!
    ↓ (SÍ, relevancia ≥ 0.6)
[Generate] → Respuesta final + Guardrails
    ↓
[Update Memory] → Actualiza historial
```

---

## 🔧 Instalación

### Dependencias adicionales

```bash
pip install langgraph langgraph-checkpoint-sqlite
```

Todas las demás dependencias ya están en `requirements.txt`.

---

## 🎯 Uso

### Opción 1: Streamlit App (Recomendado)

```bash
streamlit run app_langgraph.py --server.port 5000
```

**Features de la interfaz:**
- Chat conversacional con memoria
- Visualización del flujo de nodos
- Métricas de latencia por nodo
- Nueva conversación con un click
- Tracking de iteraciones CRAG

### Opción 2: Python Script

```python
from src.langgraph_rag import invoke_rag

# Primera pregunta
result = invoke_rag(
    question="¿Cuáles son los síntomas de la diabetes tipo 2?",
    conversation_id="user-123",
    chat_history=[],
    retrieval_mode="crag"
)

print(result["response"])

# Pregunta de seguimiento
result2 = invoke_rag(
    question="¿Y en niños?",  # Contexto automático!
    conversation_id="user-123",
    chat_history=result["chat_history"],  # Mantiene memoria
    retrieval_mode="crag"
)
```

### Opción 3: Pruebas

```bash
python test_langgraph.py
```

Ejecuta suite de tests:
- ✅ Invocación básica
- ✅ Memoria conversacional
- ✅ CRAG refinamiento
- ✅ SQL routing
- ✅ Estructura del grafo

---

## 🧩 Componentes del Grafo

### Nodos

#### 1. **Contextualize**
```python
def contextualize_question(state: RAGState) -> RAGState
```
- Reformula pregunta usando historial conversacional
- Crea queries "standalone" con contexto implícito
- **Ejemplo:**
  - User 1: "¿Qué es diabetes?"
  - User 2: "¿Y en niños?" → "¿Qué es la diabetes tipo 1 en niños?"

#### 2. **Route**
```python
def route_query(state: RAGState) -> RAGState
```
- Decide automáticamente: SQL vs Vector
- Detecta códigos ICPC-3, CIE-10, consultas estructuradas
- **Salida:** `route: "sql" | "vector"`

#### 3. **Retrieve**
```python
def retrieve_documents(state: RAGState) -> RAGState
```
- Ejecuta estrategia configurada:
  - `standard`: Multi-query + Fusion
  - `raptor`: Jerárquico con sumarios
  - `rag-fusion`: RRF combinado
  - `hyde`: Documentos hipotéticos
  - `crag`: Corrección automática
- **K inicial:** 8 documentos

#### 4. **Rerank**
```python
def rerank_documents(state: RAGState) -> RAGState
```
- BGE cross-encoder re-ranking
- **K final:** 4 documentos más relevantes
- Scores de relevancia en metadata

#### 5. **Evaluate** (Self-RAG)
```python
def evaluate_relevance(state: RAGState) -> RAGState
```
- Auto-evaluación de relevancia (0-1)
- **Umbral:** 0.6 para "can_answer"
- **Umbral refinamiento:** < 0.4 para "needs_refinement"

#### 6. **Refine** (condicional)
```python
def refine_query(state: RAGState) -> RAGState
```
- Reformula query con terminología médica precisa
- Solo si relevancia < 0.4
- **Límite:** 2 iteraciones máximo

#### 7. **Generate**
```python
def generate_response(state: RAGState) -> RAGState
```
- Genera respuesta con LLM
- Aplica guardrails médicos (MDR/AI Act)
- Actualiza memoria conversacional
- Añade disclaimer automático

### Edges Condicionales

```python
def should_refine_query(state: RAGState) -> Literal["refine", "generate"]:
    """
    Decide si refinar o generar basado en:
    - Relevancia < 0.4 → Refinar
    - Iteraciones >= 2 → Generar (límite)
    - Relevancia >= 0.6 → Generar
    """
```

**Esto crea CICLOS** en el grafo: `evaluate → refine → retrieve → rerank → evaluate`

---

## 📈 Estado del Grafo

```python
class RAGState(TypedDict):
    # Input
    question: str                     # Query actual (puede ser reformulada)
    original_question: str            # Query original del usuario
    conversation_id: str              # ID de sesión

    # Memoria
    messages: Sequence[BaseMessage]   # Historial formato LangChain
    chat_history: list[tuple]         # [(user, assistant), ...]

    # Routing
    route: str                        # "sql" | "vector"
    route_confidence: float           # 0-1
    route_reasoning: str              # Explicación

    # Retrieval
    documents: list[Document]         # Docs recuperados (k=8)
    retrieval_method: str             # "fusion" | "raptor" | etc
    retrieval_iteration: int          # 0, 1, 2 (CRAG)

    # Reranking
    reranked_docs: list[Document]     # Top 4 docs

    # Evaluation
    relevance_score: float            # 0-1
    can_answer: bool                  # ≥ 0.6?
    needs_refinement: bool            # < 0.4?

    # Generation
    response: str                     # Respuesta final

    # Metadata
    error: str | None                 # Error si ocurrió
    latency_ms: dict                  # Latencias por nodo
    node_sequence: list[str]          # Secuencia ejecutada
```

---

## 🔍 Ejemplos de Uso

### Ejemplo 1: Conversación Multi-turno

```python
from src.langgraph_rag import invoke_rag

# Turno 1
r1 = invoke_rag(
    question="¿Qué es la hipertensión arterial?",
    conversation_id="patient-001",
    chat_history=[]
)

# Turno 2 (con contexto)
r2 = invoke_rag(
    question="¿Cuál es el tratamiento?",
    # Sistema entiende: "¿Cuál es el tratamiento de la hipertensión arterial?"
    conversation_id="patient-001",
    chat_history=r1["chat_history"]
)

# Turno 3
r3 = invoke_rag(
    question="¿Y efectos secundarios?",
    # Sistema entiende contexto de tratamiento de hipertensión
    conversation_id="patient-001",
    chat_history=r2["chat_history"]
)
```

### Ejemplo 2: CRAG Automático

```python
# Query ambigua → Trigger refinamiento
result = invoke_rag(
    question="síntomas",  # Muy vaga
    retrieval_mode="crag"
)

# Sistema:
# 1. Retrieve con "síntomas" → Relevancia 0.2 (baja)
# 2. Refine → "¿Cuáles son los síntomas comunes de enfermedades crónicas?"
# 3. Retrieve nuevamente → Relevancia 0.7 (buena)
# 4. Generate respuesta

print(result["retrieval_iteration"])  # 1 (refinó una vez)
print(result["relevance_score"])      # 0.7
```

### Ejemplo 3: Routing Automático

```python
# Query SQL
result_sql = invoke_rag(
    question="¿Código ICPC-3 para diabetes tipo 2?"
)
print(result_sql["route"])  # "sql"

# Query Vector
result_vec = invoke_rag(
    question="¿Cómo se diagnostica la diabetes tipo 2?"
)
print(result_vec["route"])  # "vector"
```

---

## ⚙️ Configuración

### Variables de Entorno

```bash
# Requerido
export OPENAI_API_KEY="sk-..."

# Opcional
export PERSIST_DIR="./db/chroma"
export SQLITE_PATH="./db/icpc3.db"
export RETRIEVAL_MODE="crag"  # standard|raptor|rag-fusion|hyde|crag
export OPENAI_CHAT_MODEL="gpt-5"
export EMBEDDINGS_MODEL="intfloat/multilingual-e5-base"
```

### Checkpointing

El sistema guarda checkpoints automáticamente en:
```
./db/langgraph_checkpoints.db
```

Esto permite:
- ✅ Recuperación ante fallos
- ✅ Inspección de estados intermedios
- ✅ Replay de conversaciones

---

## 📊 Métricas y Debugging

### Ver Secuencia de Nodos

```python
result = invoke_rag(question="...")
print(result["node_sequence"])
# ['contextualize', 'route', 'retrieve', 'rerank', 'evaluate', 'generate']
```

### Latencias por Nodo

```python
for node, latency in result["latency_ms"].items():
    print(f"{node}: {latency:.1f}ms")

# Ejemplo:
# contextualize: 234.5ms
# route: 12.3ms
# retrieve: 1234.5ms
# rerank: 456.7ms
# evaluate: 789.0ms
# generate: 2345.6ms
```

### Debugging en LangSmith

LangGraph se integra automáticamente con LangSmith:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_PROJECT="rag-nexthealth"
```

Visualiza el grafo completo en https://smith.langchain.com

---

## 🆚 Comparación: Original vs LangGraph

| Feature | Original (graph.py) | LangGraph (langgraph_rag.py) |
|---------|-------------------|----------------------------|
| Memoria conversacional | ❌ | ✅ |
| Self-RAG | ❌ | ✅ |
| CRAG con ciclos | ⚠️ Básico | ✅ Avanzado (hasta 2 intentos) |
| Checkpointing | ❌ | ✅ |
| Estado persistente | ❌ | ✅ |
| Flujo visual | ❌ | ✅ |
| Edges condicionales | ⚠️ Simple | ✅ Completo |
| Debugging | ⚠️ Limitado | ✅ LangSmith integration |

---

## 🎓 Técnicas Avanzadas Implementadas

### 1. **Contextual Compression**
```python
# En contextualize_question()
# Combina historial + query actual → query standalone
```

### 2. **Adaptive Retrieval**
```python
# En evaluate_relevance()
# Decide automáticamente si necesita más documentos
```

### 3. **Query Refinement Loop**
```python
# evaluate → refine → retrieve → evaluate (máx 2 ciclos)
```

### 4. **Multi-stage Reranking**
```python
# retrieve (k=8) → BGE rerank (k=4)
```

### 5. **Guardrails Integration**
```python
# Políticas médicas MDR/AI Act aplicadas automáticamente
```

---

## 🐛 Troubleshooting

### Error: "No se pudo cargar el retriever"

**Solución:**
```bash
# Asegúrate de que el vector store existe
ls ./db/chroma

# Si no existe, ejecuta ingesta:
python -c "from src.ingest import main; main('./db/chroma', './docs')"
```

### Error: "OpenAI API key not configured"

**Solución:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Relevancia siempre baja

**Posibles causas:**
1. Vector store vacío → Ejecuta ingesta
2. Query muy vaga → Sistema refinará automáticamente
3. Modelo de embeddings diferente → Verifica configuración

---

## 📚 Recursos Adicionales

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangSmith:** https://smith.langchain.com
- **Paper CRAG:** https://arxiv.org/abs/2401.15884
- **Paper Self-RAG:** https://arxiv.org/abs/2310.11511

---

## 🤝 Contribuciones

El sistema LangGraph está diseñado para ser extensible. Puedes añadir:

- Nuevos nodos (ej: `web_search_fallback`)
- Nuevas condiciones en edges
- Nuevas estrategias de recuperación
- Nuevos evaluadores de relevancia

---

## 📄 Licencia

MIT License - Ver archivo LICENSE

---

## 🎉 Próximos Pasos

1. ✅ LangGraph implementado
2. ✅ Memoria conversacional
3. ✅ Self-RAG
4. ✅ CRAG con ciclos
5. ⏳ Web search fallback
6. ⏳ Multi-agent orchestration
7. ⏳ Streaming responses
8. ⏳ Voice interface

---

**Versión:** 2.0.0 (LangGraph Edition)
**Autor:** RAG NextHealth Team
**Fecha:** 2025
