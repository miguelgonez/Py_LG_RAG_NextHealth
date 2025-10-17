# ⚡ Quick Start - RAG NextHealth v2.0 (LangGraph)

## 🚀 Inicio en 5 Minutos

### 1️⃣ Configurar API Key

```bash
export OPENAI_API_KEY="sk-tu-key-aqui"
```

### 2️⃣ Verificar Instalación

```bash
pip list | grep langgraph
# Deberías ver: langgraph==0.6.8
```

### 3️⃣ Ejecutar Demo

```bash
python demo_langgraph.py
```

Verás 5 demos:
- ✅ Conversación multi-turno
- ✅ CRAG automático
- ✅ Routing SQL/Vector
- ✅ Métricas de latencia
- ✅ Self-RAG evaluation

### 4️⃣ Lanzar App Web

```bash
streamlit run app_langgraph.py --server.port 5000
```

Abre: http://localhost:5000

---

## 💬 Ejemplo Básico

```python
from src.langgraph_rag import invoke_rag

# Primera pregunta
result = invoke_rag(
    question="¿Qué es la diabetes tipo 2?",
    conversation_id="usuario-123",
    chat_history=[]
)

print(result["response"])

# Pregunta de seguimiento
result2 = invoke_rag(
    question="¿Cuáles son sus síntomas?",
    conversation_id="usuario-123",
    chat_history=result["chat_history"]  # ← Memoria!
)

print(result2["response"])
```

---

## 🗂️ Preparar Datos (Opcional)

Si quieres usar tus propios documentos:

### 1. Añadir documentos
```bash
# Coloca PDFs, TXTs, HTMLs en:
./docs/
```

### 2. Ejecutar ingesta
```python
from src.ingest import main as ingest_main

ingest_main(
    persist_dir="./db/chroma",
    data_dir="./docs",
    mode="incremental"  # o "full"
)
```

### 3. Verificar
```bash
ls ./db/chroma
# Deberías ver archivos de ChromaDB
```

---

## 🎯 Uso Avanzado

### Conversación Multi-turno

```python
conv_id = "paciente-001"
history = []

# Turno 1
r1 = invoke_rag("¿Qué es la hipertensión?", conv_id, history)
history = r1["chat_history"]

# Turno 2 (con contexto automático)
r2 = invoke_rag("¿Cómo se trata?", conv_id, history)
history = r2["chat_history"]

# Turno 3
r3 = invoke_rag("¿Efectos secundarios?", conv_id, history)
```

### CRAG con Refinamiento

```python
# Query ambigua → Auto-refinamiento
result = invoke_rag(
    question="tratamiento",
    retrieval_mode="crag"
)

print(f"Iteraciones: {result['retrieval_iteration']}")
print(f"Relevancia: {result['relevance_score']}")
```

### Routing Automático

```python
# SQL
sql_result = invoke_rag("¿Código ICPC-3 para diabetes?")
print(sql_result["route"])  # "sql"

# Vector
vec_result = invoke_rag("¿Qué es la diabetes?")
print(vec_result["route"])  # "vector"
```

---

## 🔧 Configuración

### Variables de Entorno

```bash
# Requerido
export OPENAI_API_KEY="sk-..."

# Opcional
export PERSIST_DIR="./db/chroma"
export SQLITE_PATH="./db/icpc3.db"
export RETRIEVAL_MODE="crag"
export OPENAI_CHAT_MODEL="gpt-5"
```

### Modos de Recuperación

- `standard` → Multi-query + BGE reranking
- `raptor` → Jerárquico con sumarios
- `rag-fusion` → RRF combinado
- `hyde` → Documentos hipotéticos
- `crag` → Corrección automática ⭐ **Recomendado**

---

## 📊 Ver Métricas

```python
result = invoke_rag("tu pregunta")

# Flujo de nodos
print(result["node_sequence"])
# ['contextualize', 'route', 'retrieve', 'rerank', 'evaluate', 'generate']

# Latencias
for node, ms in result["latency_ms"].items():
    print(f"{node}: {ms:.1f}ms")

# Metadata
print(f"Ruta: {result['route']}")
print(f"Relevancia: {result['relevance_score']:.2f}")
print(f"Iteraciones: {result['retrieval_iteration']}")
```

---

## 🐛 Troubleshooting

### Error: "No se pudo cargar el retriever"
```bash
# Vector store vacío. Ejecuta ingesta:
python -c "from src.ingest import main; main('./db/chroma', './docs')"
```

### Error: "OpenAI API key not configured"
```bash
export OPENAI_API_KEY="sk-..."
```

### Relevancia siempre baja
- Verifica que hay documentos: `ls ./db/chroma`
- Ejecuta ingesta si está vacío
- CRAG refinará automáticamente queries vagas

---

## 📚 Documentación Completa

- **Guía LangGraph:** `LANGGRAPH_GUIDE.md`
- **README Original:** `README.md`
- **Resumen Upgrade:** `UPGRADE_SUMMARY.md`

---

## 🎓 Tests

```bash
# Suite completa
python test_langgraph.py

# Solo tests específicos (edita el archivo)
```

---

## ✨ Features Principales

✅ **Memoria conversacional** → "¿Y en niños?" entiende contexto
✅ **Self-RAG** → Auto-evalúa calidad de respuestas
✅ **CRAG** → Refina automáticamente queries ambiguas
✅ **Checkpointing** → Recupera ante fallos
✅ **Trazabilidad** → Ve el flujo completo de nodos

---

## 🎯 Próximos Pasos

1. **Ejecuta la demo:** `python demo_langgraph.py`
2. **Prueba la app:** `streamlit run app_langgraph.py`
3. **Lee la guía:** `LANGGRAPH_GUIDE.md`
4. **Experimenta** con tus propias queries

---

## 💡 Ejemplos Rápidos

### Ejemplo 1: Chat Básico
```python
from src.langgraph_rag import invoke_rag

result = invoke_rag("¿Qué es la diabetes tipo 2?")
print(result["response"])
```

### Ejemplo 2: Con Historial
```python
r1 = invoke_rag("¿Qué es hipertensión?", "user1", [])
r2 = invoke_rag("¿Tratamiento?", "user1", r1["chat_history"])
```

### Ejemplo 3: CRAG Automático
```python
result = invoke_rag("síntomas", retrieval_mode="crag")
# Si relevancia baja → Refina automáticamente
```

---

**¿Listo?** 🚀

```bash
# ¡Empecemos!
python demo_langgraph.py
```

---

**Versión:** 2.0.0
**Actualizado:** 2025-10-03
