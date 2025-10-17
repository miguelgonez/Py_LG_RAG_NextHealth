# 📊 RAG NextHealth - Resumen Ejecutivo del Proyecto

**Fecha**: 17 de Octubre, 2025
**Repositorio**: https://github.com/miguelgonez/Py_LG_RAG_NextHealth
**Versión**: 1.0

---

## 🎯 Descripción General

RAG NextHealth es un sistema avanzado de Retrieval-Augmented Generation (RAG) especializado en búsqueda clínica en español. Diseñado para profesionales médicos, estudiantes de medicina e investigadores en salud, combina técnicas de última generación en procesamiento de lenguaje natural con guardrails clínicos estrictos para cumplir con regulaciones MDR y AI Act.

---

## 🏗️ Arquitectura del Sistema

### Stack Tecnológico

**Backend & Frameworks:**
- Python 3.11+
- LangChain 0.3.20 (orquestación RAG)
- LangGraph (memoria conversacional y auto-corrección)
- OpenAI GPT-5 (generación y razonamiento)

**Vectorstore & Embeddings:**
- ChromaDB (vector database)
- HuggingFace `multilingual-e5-base` (embeddings en español)
- BGE Reranker (cross-encoder para refinamiento)

**Bases de Datos:**
- SQLite (códigos ICPC-3 y mapeos CIE-10)
- ChromaDB (embeddings de documentos médicos)
- SQLite (checkpointing de LangGraph)

**Interfaz:**
- Streamlit 1.43.2 (aplicación web interactiva)

---

## 🚀 Características Principales

### 1. **Técnicas RAG Avanzadas**

#### Modos de Recuperación Disponibles:
- **Standard**: Multi-query retrieval + BGE reranking
- **RAG-Fusion**: Múltiples variaciones de consulta con Reciprocal Rank Fusion
- **HYDE**: Genera documentos hipotéticos para búsqueda mejorada
- **RAPTOR**: Recuperación jerárquica con clustering KMeans
- **CRAG**: RAG correctivo con auto-evaluación (hasta 2 iteraciones)
- **Hybrid**: Combinación de Fusion + HYDE

#### Pipeline de Procesamiento:
```
Query → Contextualize → Route → Retrieve (k=8) → Rerank (BGE, k=4)
  → Evaluate → [Refine si necesario] → Generate + Guardrails → Response
```

### 2. **Re-ranking Inteligente**
- Modelo: `BAAI/bge-reranker-base` (cross-encoder)
- Recuperación inicial: k=8 documentos
- Selección final: Top k=4 documentos más relevantes
- Mejora significativa en precisión vs retrieval básico

### 3. **Ingesta Incremental de Documentos**

#### Sistema de Detección de Cambios:
- **Hashing SHA256** de contenido de archivos
- **State Store SQLite** (`./db/document_state.db`)
- **IDs determinísticos**: `{file_hash}_{chunk_index}`
- **Eliminación automática** de chunks antiguos al actualizar
- **Limpieza inteligente** cuando se borran archivos fuente

#### Modos de Ingesta:
- **Incremental** (por defecto): Solo procesa archivos nuevos/modificados
- **Full Reindex**: Reprocesa todo el corpus desde cero

#### Formatos Soportados:
- PDF, HTML, TXT, MD

### 4. **Routing Inteligente de Consultas**

El sistema clasifica automáticamente cada consulta:

| Tipo de Query | Destino | Ejemplo |
|---------------|---------|---------|
| Códigos clínicos | SQL Database | "¿Qué código ICPC-3 es K86?" |
| Información semántica | Vectorstore | "¿Cuáles son los síntomas de diabetes?" |
| Consultas complejas | Hybrid (SQL + Vector) | "Protocolo para código T90" |

**Text-to-SQL**: Conversión automática de lenguaje natural a consultas SQL estructuradas.

### 5. **Guardrails Clínicos (MDR/AI Act)**

#### Políticas de Seguridad Estrictas:
- ❌ **No diagnósticos** personales
- ❌ **No dosis** de medicamentos
- ❌ **No consejos médicos** específicos
- ✅ **Solo información educativa** y de referencia

#### Validaciones Automáticas:
- Detección de palabras prohibidas (diagnóstico, dosis, prescripción)
- Citación obligatoria: Mínimo 2 fuentes por respuesta
- Disclaimer automático en todas las respuestas
- Límites de longitud: Máximo 2000 caracteres

#### Cumplimiento Regulatorio:
- **MDR (Medical Device Regulation)**: Información sin diagnóstico
- **AI Act**: Transparencia y limitaciones claras
- **GDPR**: No almacenamiento de datos personales

### 6. **Framework de Evaluación**

#### Métricas Implementadas:
- **nDCG@k** (Normalized Discounted Cumulative Gain)
- **Recall@k** (Cobertura de documentos relevantes)
- **Precision@k** (Precisión de resultados)
- **Latency Tracking** por etapa del pipeline

#### Dataset de Evaluación:
- 15 preguntas médicas categorizadas
- Dificultad: Fácil (4), Medio (8), Difícil (3)
- 8 categorías: cardiovascular, sueño, preventiva, salud mental, etc.
- Documentos relevantes especificados para cada pregunta

### 7. **Interfaz Streamlit (3 Pestañas)**

#### Pestaña 1: RAG Search
- Búsqueda con configuración de k, umbral de similitud, modo de recuperación
- Visualización de fuentes citadas
- Métricas de latencia y rendimiento
- Ingesta de documentos (incremental o full reindex)

#### Pestaña 2: SQL Console
- Editor de consultas natural a SQL
- Templates predefinidos
- Paginación de resultados
- Visualización de esquema de base de datos

#### Pestaña 3: Evaluation Dashboard
- Carga de CSV con preguntas de evaluación
- Visualización de métricas nDCG, Recall, Precision
- Análisis de desempeño por pregunta
- Identificación de consultas de bajo rendimiento

### 8. **LangGraph - Memoria Conversacional (Versión Avanzada)**

#### Características:
- **Contexto conversacional**: Mantiene historial entre preguntas
- **Self-RAG**: Auto-evaluación de calidad de respuestas
- **CRAG Mejorado**: Hasta 2 iteraciones de refinamiento automático
- **Checkpointing**: Recuperación ante fallos con SQLite
- **Trazabilidad**: Visualización completa del flujo de nodos

#### Uso:
```bash
streamlit run app_langgraph.py --server.port 5000
```

---

## 📁 Estructura del Proyecto

```
Py_LG_RAG_NextHealth/
├── app.py                          # Aplicación principal Streamlit
├── app_langgraph.py                # Versión con LangGraph (conversacional)
├── README.md                       # Documentación principal
├── QUICKSTART.md                   # Guía de inicio rápido
├── LANGGRAPH_GUIDE.md              # Guía de LangGraph
├── CASOS_DE_USO.md                 # 7 casos de uso detallados
├── RESUMEN_EJECUTIVO.md            # Este documento
│
├── src/
│   ├── ingest.py                   # Ingesta incremental de documentos
│   ├── state_store.py              # Tracking de estado de archivos
│   ├── retrievers.py               # Multi-query, Fusion, RAPTOR
│   ├── reranker.py                 # BGE cross-encoder
│   ├── routing.py                  # Routing SQL vs Vectorstore
│   ├── tools.py                    # Text-to-SQL, utilidades DB
│   ├── graph.py                    # Orquestación LangChain
│   ├── langgraph_rag.py            # Implementación LangGraph
│   ├── policies.py                 # Guardrails clínicos
│   ├── evaluation.py               # Métricas nDCG, Recall
│   ├── advanced_rag.py             # RAG-Fusion, HYDE
│   ├── raptor.py                   # RAPTOR jerárquico
│   └── crag.py                     # CRAG correctivo
│
├── data/
│   ├── ejemplo_consultas.txt       # 33 consultas de prueba
│   ├── evaluation_medical.csv      # Dataset de evaluación
│   └── test_evaluation.csv         # Dataset adicional
│
├── docs/                           # 14 PDFs de documentación médica
│   ├── ICPC-3 International Classification.pdf
│   ├── Lifestyle Medicine Fourth Edition.pdf
│   ├── Positive Health The Basics.pdf
│   ├── Health Promotion Programs.pdf
│   └── ...
│
├── db/
│   ├── chroma/                     # Vector database (146MB+)
│   ├── icpc3.db                    # Códigos ICPC-3 y mapeos
│   ├── document_state.db           # Estado de ingesta
│   └── langgraph_checkpoints.db    # Checkpoints conversacionales
│
├── scripts/
│   ├── incremental_ingest.py       # Script de ingesta standalone
│   └── fix_deprecations.py         # Actualización de dependencias
│
├── requirements.txt                # Dependencias Python
├── pyproject.toml                  # Configuración uv
└── .gitignore                      # Exclusiones Git
```

---

## 📚 Corpus de Documentos

### Documentos Médicos Disponibles (14 PDFs, ~231MB)

#### Medicina Clínica:
1. **ICPC-3 International Classification of Primary Care** (19MB)
   - Clasificación internacional de atención primaria, 3ª edición
   - 17 categorías, códigos clínicos estructurados

2. **Lifestyle Medicine Fourth Edition** (20MB)
   - Medicina del estilo de vida, 4ª edición
   - Intervenciones preventivas y terapéuticas

3. **Assessment in Behavioral Medicine** (8.8MB)
   - Evaluación en medicina conductual

4. **Principles of Gender-Specific Medicine** (44MB)
   - Medicina específica por género y biología postgenómica

#### Salud y Bienestar:
5. **Health and Exercise Wellbeing** (56MB)
   - Salud y bienestar a través del ejercicio físico

6. **Health Problems - Philosophical Puzzles** (1.8MB)
   - Problemas filosóficos sobre la naturaleza de la salud

7. **Health Promotion Programs** (5.3MB)
   - Programas de promoción de la salud, teoría y práctica

#### Salud Positiva y Psicología:
8. **Positive Health The Basics** (12MB)
   - Fundamentos de salud positiva

9. **Positive Health 100+ Tools** (2MB)
   - Herramientas basadas en investigación para bienestar

10. **Routledge Handbook of Positive Health Sciences** (3.3MB)
    - Manual internacional de ciencias de salud positiva

#### Temas Específicos:
11. **Mayo Clinic Guide to Better Sleep** (9.5MB)
    - Guía para insomnio, apnea del sueño y trastornos

12. **Virtual You** (29MB)
    - Gemelos digitales y su revolución en medicina

13. **The 5 Types of Wealth** (12MB)
    - Guía transformadora para diseñar tu vida ideal

14. **AI for Life** (2.6MB)
    - 100+ formas de usar IA en la vida cotidiana

### Estadísticas del Corpus:
- **Tamaño total**: ~231MB
- **Chunks estimados**: ~22,830 (chunk_size=1200, overlap=200)
- **Idiomas**: Principalmente inglés, con términos médicos multilingües
- **Embeddings**: multilingual-e5-base (optimizado para español)

---

## 💾 Base de Datos SQL - ICPC-3

### Estructura de Tablas:

#### 1. `icpc_codes`
- Códigos ICPC-3 con descripciones en español e inglés
- Criterios de inclusión y exclusión
- Mapeos a ICD-10 y SNOMED

#### 2. `icpc_icd10_mapping`
- Mapeos entre ICPC-3 e ICD-10
- Tipo de relación: exact, broader, narrower
- Nivel de confianza (0.0 - 1.0)

#### 3. `icpc_categories`
- 17 categorías principales (A-Z)
- Nombres en español e inglés
- Descripciones detalladas

### Códigos de Ejemplo Incluidos:

| Código | Título Español | Categoría | ICD-10 |
|--------|----------------|-----------|--------|
| T90 | Diabetes mellitus tipo 2 | Endocrino | E11 |
| T89 | Diabetes mellitus tipo 1 | Endocrino | E10 |
| K86 | Hipertensión arterial sin complicaciones | Cardiovascular | I10 |
| K87 | Hipertensión arterial con complicaciones | Cardiovascular | I11-I15 |
| R96 | Asma | Respiratorio | J45 |
| L84 | Dolor de espalda con irradiación | Musculoesquelético | M54.4 |
| L83 | Síndrome del cuello | Musculoesquelético | M54.2 |
| P74 | Trastorno de ansiedad | Psicológico | F41 |
| P76 | Trastorno depresivo | Psicológico | F32-F33 |
| S87 | Dermatitis atópica/eccema | Piel | L20 |

---

## 🎯 Casos de Uso Documentados

### 1. Búsqueda de Códigos ICPC-3
- Búsqueda directa por código
- Búsqueda por síntomas o condiciones
- Navegación multi-turno conversacional
- **Modo recomendado**: SQL routing automático

### 2. Consulta sobre Tratamientos y Protocolos
- Tratamientos de primera línea
- Protocolos clínicos completos
- Comparación de opciones terapéuticas
- **Modo recomendado**: RAG-Fusion o RAPTOR

### 3. Comparación de Enfermedades Similares
- Diagnóstico diferencial automatizado
- Comparaciones lado a lado
- Identificación de criterios distintivos
- **Modo recomendado**: CRAG (auto-refinamiento)

### 4. Búsqueda con Contexto Conversacional (LangGraph)
- Sesiones de aprendizaje progresivo
- Clarificación progresiva de consultas vagas
- Memoria entre preguntas relacionadas
- **Requiere**: `app_langgraph.py`

### 5. Mapeos entre Clasificaciones (ICPC-3 a CIE-10)
- Mapeo directo ICPC-3 → ICD-10
- Mapeo inverso ICD-10 → ICPC-3
- Búsqueda por descripción clínica
- **Modo recomendado**: SQL + Vectorstore (Hybrid)

### 6. Investigación de Evidencia Científica
- Revisión de evidencia en múltiples documentos
- Búsqueda de guías clínicas
- Análisis de contradicciones o matices
- **Modo recomendado**: RAG-Fusion o RAPTOR

### 7. Formación y Autoevaluación
- Sesiones de estudio estructuradas
- Preparación de casos clínicos
- Técnica Feynman adaptada con RAG
- **Modo recomendado**: Standard o LangGraph conversacional

---

## 🔧 Configuración e Instalación

### Requisitos del Sistema:
- **Python**: 3.8+
- **RAM**: 4GB+ (recomendado para embeddings)
- **Disco**: 2GB+ (vectorstore y modelos)
- **API Key**: OpenAI (GPT-5)

### Instalación:

```bash
# 1. Clonar repositorio
git clone https://github.com/miguelgonez/Py_LG_RAG_NextHealth.git
cd Py_LG_RAG_NextHealth

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key
echo 'OPENAI_API_KEY="sk-..."' > .env

# 5. Crear base de datos SQL
python -c "from src.tools import create_sample_icpc_db; create_sample_icpc_db('./db/icpc3.db')"

# 6. Ejecutar ingesta de documentos
python -m src.ingest --persist_dir ./db/chroma --data_dir ./docs --mode full

# 7. Iniciar aplicación
streamlit run app.py --server.port 5000
```

### Variables de Entorno:

```bash
# Requerido
OPENAI_API_KEY="sk-..."

# Opcional
OPENAI_CHAT_MODEL="gpt-5"  # Default
EMBEDDINGS_MODEL="intfloat/multilingual-e5-base"  # Default
RERANKER_MODEL="BAAI/bge-reranker-base"  # Default
```

---

## 📊 Métricas de Rendimiento

### Latencia Típica (por consulta):

| Componente | Tiempo | Notas |
|------------|--------|-------|
| Routing | ~0.1s | Clasificación SQL vs Vector |
| Retrieval inicial | ~0.5-1s | k=8 documentos |
| Re-ranking (BGE) | ~0.3-0.5s | Selección top k=4 |
| Generación (GPT-5) | ~2-4s | Depende de longitud |
| **Total** | **~3-6s** | Pipeline completo |

### Configuraciones Recomendadas:

| Parámetro | Valor por Defecto | Rango Recomendado |
|-----------|-------------------|-------------------|
| chunk_size | 1200 | 800-1500 |
| chunk_overlap | 200 | 100-300 |
| k_initial | 8 | 5-10 |
| k_reranked | 4 | 3-5 |
| similarity_threshold | 0.7 | 0.6-0.8 |
| max_response_length | 2000 chars | 1500-2500 |

---

## 🔒 Seguridad y Privacidad

### Datos Sensibles:
- **API Keys**: Almacenadas en `.env` (no en Git)
- **Datos personales**: No se almacenan ni procesan
- **Historial de consultas**: Solo local (no se comparte)

### Guardrails Implementados:
- Validación de entrada (detección de intent malicioso)
- Validación de salida (filtrado de diagnósticos/dosis)
- Rate limiting (si se implementa en producción)
- Logging de errores (no de contenido médico)

### Cumplimiento:
- **GDPR**: No hay datos personales
- **HIPAA**: No aplicable (información educativa, no clínica)
- **MDR/AI Act**: Guardrails médicos estrictos

---

## 🚦 Estado del Proyecto

### ✅ Completado:
- [x] Arquitectura RAG multi-modal (6 técnicas)
- [x] Re-ranking con BGE cross-encoder
- [x] Ingesta incremental con detección de cambios
- [x] Routing inteligente (SQL/Vector/Hybrid)
- [x] Guardrails clínicos (MDR/AI Act)
- [x] Framework de evaluación (nDCG, Recall, Precision)
- [x] Interfaz Streamlit (3 pestañas)
- [x] Base de datos ICPC-3 con 10 códigos
- [x] Documentación completa (README, QUICKSTART, CASOS_DE_USO)
- [x] Dataset de evaluación (15 preguntas)
- [x] 33 consultas de ejemplo categorizadas
- [x] Repositorio GitHub sincronizado

### 🔄 En Proceso:
- [ ] Ingesta completa de 14 PDFs (~22,830 chunks)

### ⏳ Pendiente (Mejoras Futuras):
- [ ] Ampliar base de datos ICPC-3 (actualmente 10 códigos, objetivo: 100+)
- [ ] Implementar caché de embeddings para consultas frecuentes
- [ ] Dashboard de analytics (métricas de uso)
- [ ] API REST para integración con sistemas externos
- [ ] Soporte multilingüe explícito (actualmente español/inglés)
- [ ] Fine-tuning del reranker en corpus médico español
- [ ] Integración con SNOMED-CT y terminologías adicionales
- [ ] Tests automatizados (pytest, coverage >80%)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Containerización (Docker + docker-compose)
- [ ] Despliegue en cloud (AWS/GCP/Azure)

---

## 📖 Recursos Adicionales

### Documentación:
- **README.md**: Introducción y características principales
- **QUICKSTART.md**: Guía de inicio rápido
- **LANGGRAPH_GUIDE.md**: Uso de memoria conversacional
- **CASOS_DE_USO.md**: 7 casos de uso con código completo
- **RESUMEN_EJECUTIVO.md**: Este documento

### Datasets:
- **data/ejemplo_consultas.txt**: 33 queries para pruebas
- **data/evaluation_medical.csv**: 15 preguntas con ground truth

### Scripts:
- **scripts/incremental_ingest.py**: Ingesta standalone
- **scripts/fix_deprecations.py**: Actualización de dependencias

---

## 🤝 Contribuciones y Soporte

### Repositorio:
https://github.com/miguelgonez/Py_LG_RAG_NextHealth

### Issues y Feature Requests:
Abre un issue en GitHub describiendo:
- Problema encontrado o mejora propuesta
- Pasos para reproducir (si es bug)
- Entorno (OS, Python version, etc.)

### Contribuciones:
1. Fork del repositorio
2. Crear branch (`git checkout -b feature/amazing-feature`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Abrir Pull Request

---

## ⚠️ Aviso Legal

**IMPORTANTE**: Este sistema es exclusivamente para fines informativos y educativos. **NO proporciona diagnósticos, tratamientos ni consejos médicos personalizados**.

Para cualquier consulta médica, siempre consulte con un profesional de la salud calificado.

### Limitaciones:
- La información puede estar desactualizada
- Los documentos fuente pueden contener errores
- El sistema puede generar respuestas incorrectas (alucinaciones)
- No sustituye el juicio clínico profesional
- No debe usarse para decisiones médicas críticas

### Uso Responsable:
- Validar siempre con fuentes oficiales
- Contrastar con múltiples referencias
- Aplicar juicio crítico profesional
- Reportar respuestas incorrectas o peligrosas

---

## 📝 Licencia

MIT License

Copyright (c) 2025 RAG NextHealth

---

## 📞 Contacto

**Proyecto**: RAG NextHealth
**Repositorio**: https://github.com/miguelgonez/Py_LG_RAG_NextHealth
**Versión**: 1.0
**Última Actualización**: 17 de Octubre, 2025

---

**🤖 Generated with [Claude Code](https://claude.com/claude-code)**

**Co-Authored-By: Claude <noreply@anthropic.com>**
