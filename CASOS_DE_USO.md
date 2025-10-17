# Casos de Uso Practicos - RAG NextHealth

## Introduccion

Este documento proporciona casos de uso reales y detallados para aprovechar al maximo el sistema RAG NextHealth. Cada caso incluye ejemplos especificos, configuraciones recomendadas y mejores practicas para profesionales medicos, estudiantes y personal sanitario.

---

## Caso de Uso 1: Busqueda de Codigos ICPC-3

### Descripcion
Busqueda y consulta de codigos de la Clasificacion Internacional de Atencion Primaria (ICPC-3) para codificacion clinica precisa.

### Perfil de Usuario
- Medicos de atencion primaria
- Personal de codificacion clinica
- Estudiantes de medicina

### Ejemplos de Consultas

#### Ejemplo 1.1: Busqueda directa de codigo
```python
from src.langgraph_rag import invoke_rag

result = invoke_rag(
    question="¿Cual es el codigo ICPC-3 para diabetes tipo 2?",
    conversation_id="medico-001",
    chat_history=[]
)
```

**Query SQL esperada:**
```sql
SELECT code, title_es, component
FROM icpc3_codes
WHERE title_es LIKE '%diabetes%tipo%2%'
```

#### Ejemplo 1.2: Busqueda por sintomas
```python
result = invoke_rag(
    question="Codigo ICPC-3 para dolor toracico",
    conversation_id="medico-001",
    chat_history=[]
)
```

#### Ejemplo 1.3: Busqueda multi-turno
```python
# Primer turno
r1 = invoke_rag(
    question="Codigos ICPC-3 para enfermedades cardiovasculares",
    conversation_id="medico-001",
    chat_history=[]
)

# Segundo turno (con contexto)
r2 = invoke_rag(
    question="¿Y para arritmias especificamente?",
    conversation_id="medico-001",
    chat_history=r1["chat_history"]
)
```

### Modo de Recuperacion
- **Recomendado:** SQL routing automatico
- **Alternativa:** `standard` si la consulta es ambigua

### Configuracion Optima

```bash
# Variables de entorno
export RETRIEVAL_MODE="standard"  # Permite routing automatico
export OPENAI_CHAT_MODEL="gpt-5"
```

**Parametros en app:**
- K: 4-6 documentos
- Umbral de similitud: No aplica (SQL directo)
- Re-ranking: No necesario para SQL

### Resultados Esperados

```
Ruta: sql
Confianza: 0.95
Respuesta:
"El codigo ICPC-3 para diabetes tipo 2 es T90. Este codigo pertenece al
componente de enfermedades endocrinas, metabolicas y nutricionales..."

Fuentes citadas:
- icpc3_codes tabla (SQL query)
- [Documento de referencia ICPC-3]
```

### Tips de Optimizacion

1. **Usa terminologia exacta:** "ICPC-3", "codigo", "clasificacion"
2. **Especifica el tipo:** "diabetes tipo 2" vs "diabetes"
3. **Consultas en lote:** Si necesitas multiples codigos, usa conversacion multi-turno
4. **Verifica siempre:** Contrasta con fuentes oficiales WONCA

### Errores Comunes a Evitar

- Usar acronimos sin contexto ("DM2" en vez de "diabetes tipo 2")
- Mezclar clasificaciones ("codigo CIE" cuando quieres ICPC-3)
- Consultas demasiado vagas ("codigos de enfermedades")

---

## Caso de Uso 2: Consulta sobre Tratamientos y Protocolos

### Descripcion
Busqueda de informacion basada en evidencia sobre tratamientos, protocolos clinicos y guias de practica.

### Perfil de Usuario
- Medicos residentes
- Enfermeras especialistas
- Farmaceuticos clinicos

### Ejemplos de Consultas

#### Ejemplo 2.1: Tratamiento de primera linea
```python
result = invoke_rag(
    question="¿Cual es el tratamiento de primera linea para hipertension arterial en adultos?",
    conversation_id="residente-001",
    chat_history=[],
    retrieval_mode="rag-fusion"  # Combina multiples perspectivas
)
```

#### Ejemplo 2.2: Protocolo especifico
```python
result = invoke_rag(
    question="Protocolo de manejo de cetoacidosis diabetica en urgencias",
    conversation_id="urgencias-001",
    chat_history=[],
    retrieval_mode="raptor"  # Recuperacion jerarquica
)
```

#### Ejemplo 2.3: Comparacion de tratamientos
```python
# Primera consulta
r1 = invoke_rag(
    question="¿Cuales son las opciones de tratamiento para asma moderada?",
    conversation_id="neumologia-001",
    chat_history=[]
)

# Seguimiento
r2 = invoke_rag(
    question="¿Cual tiene menos efectos secundarios?",
    conversation_id="neumologia-001",
    chat_history=r1["chat_history"]
)
```

### Modo de Recuperacion

- **Opcion 1 (Recomendado):** `rag-fusion`
  - Combina multiples variaciones de la consulta
  - Mejor para preguntas amplias sobre tratamientos

- **Opcion 2:** `raptor`
  - Ideal para protocolos complejos multi-paso
  - Recupera sumarios jerarquicos

- **Opcion 3:** `crag`
  - Mejor para consultas ambiguas
  - Auto-refina hasta encontrar informacion relevante

### Configuracion Optima

```python
# En Streamlit
retrieval_mode = "rag-fusion"
k = 6  # Mas documentos para cobertura completa
similarity_threshold = 0.7  # Alta precision
```

### Resultados Esperados

```
Modo: rag-fusion
Documentos recuperados: 6
Re-ranked top: 4
Relevancia promedio: 0.82

Respuesta:
"El tratamiento de primera linea para hipertension arterial en adultos incluye:

1. Modificaciones en el estilo de vida:
   - Reduccion de sodio en la dieta
   - Ejercicio aerobico regular
   - Control de peso

2. Farmacoterapia inicial:
   - Diureticos tiazidicos (ej. hidroclorotiazida)
   - IECA o ARA-II en pacientes con diabetes o enfermedad renal

[Fuente 1: Guia ESC/ESH 2023]
[Fuente 2: Protocolo HTA - Ministerio Sanidad]

IMPORTANTE: Esta informacion es educativa. Consulte a un profesional
sanitario para decisiones clinicas individuales."
```

### Tips de Optimizacion

1. **Especifica el contexto:**
   - "tratamiento en adultos mayores"
   - "protocolo en atencion primaria"
   - "manejo en urgencias"

2. **Usa RAG-Fusion para:**
   - Revisiones completas de tratamientos
   - Comparaciones farmacologicas
   - Guias de practica clinica

3. **Usa RAPTOR para:**
   - Protocolos multi-etapa
   - Algoritmos de decision
   - Manejo de patologias complejas

4. **Incrementa K a 6-8** para:
   - Temas con multiple evidencia
   - Comparaciones extensas
   - Protocolos detallados

### Limitaciones del Sistema (Guardrails)

El sistema NO proporcionara:
- Dosis especificas de medicamentos
- Diagnosticos personalizados
- Recomendaciones individualizadas

Siempre mostrara disclaimer de uso educativo.

---

## Caso de Uso 3: Comparacion de Enfermedades Similares

### Descripcion
Analisis comparativo de enfermedades con presentaciones clinicas similares para apoyo en diagnostico diferencial.

### Perfil de Usuario
- Estudiantes de medicina (clinica)
- Medicos en formacion
- Personal docente

### Ejemplos de Consultas

#### Ejemplo 3.1: Diagnostico diferencial
```python
result = invoke_rag(
    question="Diferencias entre bronquitis aguda y neumonia en sintomas y signos clinicos",
    conversation_id="estudiante-001",
    chat_history=[],
    retrieval_mode="crag"  # Auto-refina para precision
)
```

#### Ejemplo 3.2: Comparacion multi-turno
```python
# Turno 1: Solicitar comparacion
r1 = invoke_rag(
    question="¿Cuales son las diferencias entre diabetes tipo 1 y tipo 2?",
    conversation_id="estudiante-002",
    chat_history=[]
)

# Turno 2: Profundizar en aspecto especifico
r2 = invoke_rag(
    question="¿Y en el tratamiento farmacologico?",
    conversation_id="estudiante-002",
    chat_history=r1["chat_history"]
)

# Turno 3: Casos especiales
r3 = invoke_rag(
    question="¿Como se diferencian en poblacion pediatrica?",
    conversation_id="estudiante-002",
    chat_history=r2["chat_history"]
)
```

#### Ejemplo 3.3: Tabla comparativa
```python
result = invoke_rag(
    question="Comparacion entre insuficiencia cardiaca sistolica y diastolica: fisiopatologia, sintomas y tratamiento",
    conversation_id="cardio-001",
    chat_history=[],
    retrieval_mode="rag-fusion"
)
```

### Modo de Recuperacion

- **Primera opcion (Recomendado):** `crag`
  - Refina automaticamente consultas ambiguas
  - Ideal cuando no estas seguro de la terminologia exacta
  - Hasta 2 iteraciones de refinamiento

- **Segunda opcion:** `rag-fusion`
  - Mejor para comparaciones amplias
  - Combina multiples perspectivas

- **Tercera opcion:** `hyde`
  - Genera documentos hipotetiques de comparacion
  - Util cuando hay poca documentacion directa

### Configuracion Optima

```python
# Configuracion recomendada
retrieval_mode = "crag"
k = 8  # Inicial (se reduce a 4 tras re-ranking)
similarity_threshold = 0.65  # Moderado para permitir refinamiento

# Monitorizar
print(f"Iteraciones de refinamiento: {result['retrieval_iteration']}")
print(f"Score de relevancia final: {result['relevance_score']}")
```

### Flujo CRAG Esperado

```
Query inicial: "diferencias bronquitis neumonia"
    ↓
[Retrieve] → 8 docs, Relevancia: 0.3 (baja)
    ↓
[Refine] → "Comparacion clinica entre bronquitis aguda y neumonia comunitaria:
            sintomas, signos radiologicos y tratamiento"
    ↓
[Retrieve] → 8 docs, Relevancia: 0.8 (alta)
    ↓
[Rerank] → 4 docs mas relevantes
    ↓
[Generate] → Respuesta comparativa estructurada
```

### Resultados Esperados

```
Iteraciones CRAG: 1
Relevancia final: 0.85
Documentos finales: 4

Respuesta:
"COMPARACION: Bronquitis Aguda vs Neumonia

1. ETIOLOGIA:
   - Bronquitis aguda: Mayormente viral (90%)
   - Neumonia: Bacterial (S. pneumoniae mas frecuente)

2. SINTOMAS:
   - Bronquitis: Tos productiva, sin fiebre alta, disnea leve
   - Neumonia: Fiebre alta, tos con expectoracion purulenta, disnea moderada-severa

3. EXPLORACION:
   - Bronquitis: Roncus y sibilancias difusas
   - Neumonia: Crepitantes focales, matidez a la percusion

4. RADIOLOGIA:
   - Bronquitis: Radiografia normal
   - Neumonia: Infiltrado alveolar o intersticial

5. TRATAMIENTO:
   - Bronquitis: Sintomatico (antipireticos, antitusivos)
   - Neumonia: Antibioticoterapia empirica

[Fuente 1: Guia SEPAR Infecciones Respiratorias]
[Fuente 2: Protocolo Neumonia - SemFYC]"
```

### Tips de Optimizacion

1. **Estructura la pregunta:**
   - "Diferencias entre X e Y en [aspecto]"
   - "Comparacion de X vs Y: sintomas, diagnostico, tratamiento"

2. **Usa conversacion multi-turno:**
   - Primera consulta: Comparacion general
   - Segunda: Profundizar en diferencias especificas
   - Tercera: Casos especiales o excepciones

3. **Aprovecha CRAG:**
   - No necesitas terminologia perfecta
   - El sistema refinara automaticamente
   - Revisa `retrieval_iteration` para ver si hubo refinamiento

4. **Incrementa K para comparaciones:**
   - K=8 permite capturar informacion de ambas condiciones
   - Re-ranking selecciona lo mas relevante

### Casos de Uso Educativo

**Ejemplo: Preparacion para OSCE**
```python
# Pregunta de examen tipica
result = invoke_rag(
    question="Diagnostico diferencial de dolor toracico: angina estable vs inestable vs infarto",
    retrieval_mode="rag-fusion",
    k=8
)
```

---

## Caso de Uso 4: Busqueda con Contexto Conversacional (LangGraph)

### Descripcion
Interacciones multi-turno donde el sistema mantiene contexto y memoria conversacional para responder preguntas de seguimiento.

### Perfil de Usuario
- Estudiantes investigando temas complejos
- Profesionales en formacion continua
- Personal sanitario consultando durante guardias

### Ejemplos de Consultas

#### Ejemplo 4.1: Sesion de aprendizaje progresivo
```python
from src.langgraph_rag import invoke_rag

# Sesion de estudio sobre EPOC
conv_id = "estudiante-epoc-001"
history = []

# Turno 1: Conceptos basicos
r1 = invoke_rag(
    question="¿Que es la EPOC?",
    conversation_id=conv_id,
    chat_history=history
)
history = r1["chat_history"]
print(r1["response"])

# Turno 2: Profundizar (el sistema entiende "EPOC" del contexto)
r2 = invoke_rag(
    question="¿Cuales son los criterios diagnosticos?",
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]
print(r2["response"])

# Turno 3: Tratamiento
r3 = invoke_rag(
    question="¿Y el tratamiento?",
    conversation_id=conv_id,
    chat_history=history
)
history = r3["chat_history"]

# Turno 4: Casos especiales
r4 = invoke_rag(
    question="¿Como cambia en pacientes con insuficiencia cardiaca?",
    conversation_id=conv_id,
    chat_history=history
)
```

**Reformulacion interna (invisible al usuario):**
- Turno 2: "¿Cuales son los criterios diagnosticos de EPOC?"
- Turno 3: "¿Cual es el tratamiento de EPOC?"
- Turno 4: "¿Como cambia el manejo de EPOC en pacientes con insuficiencia cardiaca?"

#### Ejemplo 4.2: Consulta clinica interactiva
```python
# Caso clinico paso a paso
conv_id = "urgencias-turno-noche"
history = []

# Paso 1: Presentacion del caso
r1 = invoke_rag(
    question="Paciente con disnea aguda y taquicardia. ¿Diagnostico diferencial?",
    conversation_id=conv_id,
    chat_history=history
)
history = r1["chat_history"]

# Paso 2: Datos adicionales
r2 = invoke_rag(
    question="Si ademas tiene dolor pleuritico, ¿que considerarias?",
    conversation_id=conv_id,
    chat_history=history  # Mantiene contexto de disnea+taquicardia
)
history = r2["chat_history"]

# Paso 3: Exploracion especifica
r3 = invoke_rag(
    question="¿Que pruebas complementarias necesito?",
    conversation_id=conv_id,
    chat_history=history  # Contexto completo del caso
)
```

#### Ejemplo 4.3: Clarificacion progresiva
```python
# Usuario empieza con consulta vaga
conv_id = "medico-aps-001"
history = []

r1 = invoke_rag(
    question="tratamiento hipertension",  # Vaga
    conversation_id=conv_id,
    chat_history=history,
    retrieval_mode="crag"  # Auto-refina
)
history = r1["chat_history"]

# Sistema refino automaticamente, ahora el usuario especifica
r2 = invoke_rag(
    question="en pacientes diabeticos",  # Contexto: HTA + diabetes
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]

r3 = invoke_rag(
    question="y si tienen insuficiencia renal?",  # Contexto: HTA+DM+IRC
    conversation_id=conv_id,
    chat_history=history
)
```

### Modo de Recuperacion

**Todos los modos son compatibles con memoria conversacional:**

- **`standard`:** Rapido, bueno para consultas directas
- **`crag`:** Mejor para sesiones donde las consultas iniciales son vagas
- **`rag-fusion`:** Ideal para temas complejos que requieren multiples perspectivas
- **`raptor`:** Perfecto para protocolos multi-paso discutidos en varios turnos

### Configuracion Optima

```python
# Configuracion recomendada para sesiones conversacionales
retrieval_mode = "crag"  # Auto-ajuste de calidad
k = 6  # Balance entre velocidad y cobertura

# Importante: Siempre pasar el historial actualizado
result = invoke_rag(
    question="tu pregunta",
    conversation_id="id-unico-sesion",  # Mantener igual en toda la sesion
    chat_history=previous_result["chat_history"]  # ← CRUCIAL!
)
```

### Anatomia del Estado Conversacional

```python
# Despues de 3 turnos, el chat_history contiene:
[
    ("¿Que es la EPOC?", "La EPOC es una enfermedad..."),
    ("¿Cuales son los criterios diagnosticos?", "Los criterios incluyen..."),
    ("¿Y el tratamiento?", "El tratamiento de EPOC se basa en...")
]

# El nodo 'contextualize' usa esto para reformular:
"¿Y el tratamiento?" → "¿Cual es el tratamiento de EPOC?"
```

### Flujo LangGraph en Conversacion

```
Turno 2: "¿Y el tratamiento?"
    ↓
[Contextualize] → Reformula: "¿Tratamiento de EPOC?"
    |             Usa: chat_history con turno 1
    ↓
[Route] → Decide: Vector (no es consulta SQL)
    ↓
[Retrieve] → Busca docs sobre "tratamiento EPOC"
    ↓
[Rerank] → Top 4 mas relevantes
    ↓
[Evaluate] → Relevancia: 0.9 (excelente)
    ↓
[Generate] → Respuesta + Actualiza memoria
    ↓
Updated chat_history: [turno1, turno2]
```

### Resultados Esperados

**Metricas de sesion:**
```python
# Turno 1
node_sequence: ['contextualize', 'route', 'retrieve', 'rerank', 'evaluate', 'generate']
retrieval_iteration: 0
relevance_score: 0.85

# Turno 2 (con contexto)
node_sequence: ['contextualize', 'route', 'retrieve', 'rerank', 'evaluate', 'generate']
retrieval_iteration: 0
relevance_score: 0.92  # Mejor gracias al contexto!
```

**Calidad de respuestas:**
- Turno 1: Informacion general
- Turno 2-4: Respuestas cada vez mas especificas
- Coherencia: Sistema recuerda tema principal

### Tips de Optimizacion

1. **Usa conversation_id consistente:**
   ```python
   # BIEN
   conv_id = f"user-{user_id}-session-{timestamp}"

   # MAL - Cambia cada turno
   conv_id = str(time.time())  # ❌
   ```

2. **Actualiza chat_history correctamente:**
   ```python
   # BIEN
   history = result["chat_history"]  # Actualiza
   next_result = invoke_rag(..., chat_history=history)

   # MAL
   next_result = invoke_rag(..., chat_history=[])  # Pierde contexto
   ```

3. **Estructura sesiones tematicas:**
   - Una sesion = Un tema principal
   - Reinicia conversation_id para temas nuevos

4. **Aprovecha la contextualizacion:**
   - Preguntas cortas: "¿Y en niños?", "¿Efectos secundarios?"
   - El sistema expande automaticamente con contexto

5. **Monitoriza la memoria:**
   ```python
   print(f"Turnos en memoria: {len(result['chat_history'])}")

   # Si > 10 turnos, considera reiniciar sesion
   if len(history) > 10:
       history = []  # Reinicia para evitar contexto muy largo
   ```

### Limitaciones y Consideraciones

1. **Ventana de contexto:**
   - LLM tiene limite de tokens
   - Sesiones muy largas (>15 turnos) pueden perder contexto inicial
   - Solucion: Reiniciar sesion periodicamente

2. **Costo de tokens:**
   - Cada turno incluye todo el historial en el prompt
   - Sesiones largas = Mayor costo

3. **Checkpointing:**
   - Sistema guarda estado en `./db/langgraph_checkpoints.db`
   - Permite recuperar sesiones interrumpidas

### Casos de Uso Avanzado

#### A. Sesion de formacion estructurada
```python
# Programa de 4 semanas - Diabetes
semana_1 = "diabetes-formacion-semana1"
semana_2 = "diabetes-formacion-semana2"
# ... diferentes conversation_id por modulo
```

#### B. Consulta durante guardia
```python
# Una sesion por paciente
paciente_1 = "guardia-2025-01-15-paciente-001"
paciente_2 = "guardia-2025-01-15-paciente-002"
```

#### C. Investigacion bibliografica
```python
# Sesion de revision sistematica
revision_id = "revision-hta-2025"
# Multiple turnos refinando busqueda
```

---

## Caso de Uso 5: Mapeos entre Clasificaciones (ICPC-3 a CIE-10)

### Descripcion
Conversion y mapeo entre diferentes sistemas de clasificacion clinica, especialmente ICPC-3 (atencion primaria) y CIE-10 (hospitales).

### Perfil de Usuario
- Personal de codificacion clinica
- Medicos que trabajan en multiples entornos
- Auditores medicos
- Gestores de informacion sanitaria

### Ejemplos de Consultas

#### Ejemplo 5.1: Mapeo directo ICPC-3 a CIE-10
```python
result = invoke_rag(
    question="¿Cual es el codigo CIE-10 equivalente al codigo ICPC-3 K86?",
    conversation_id="codificacion-001",
    chat_history=[]
)
```

**Query SQL esperada:**
```sql
SELECT i.code as icpc3_code, i.title_es as icpc3_title,
       m.icd10_code, c.title as icd10_title
FROM icpc3_codes i
JOIN icpc3_icd10_mapping m ON i.code = m.icpc3_code
JOIN icd10_codes c ON m.icd10_code = c.code
WHERE i.code = 'K86'
```

#### Ejemplo 5.2: Mapeo inverso CIE-10 a ICPC-3
```python
result = invoke_rag(
    question="¿Que codigos ICPC-3 corresponden al CIE-10 E11 (diabetes tipo 2)?",
    conversation_id="codificacion-002",
    chat_history=[]
)
```

#### Ejemplo 5.3: Mapeo por descripcion
```python
# El usuario no conoce los codigos
result = invoke_rag(
    question="Mapeo entre ICPC-3 y CIE-10 para hipertension arterial",
    conversation_id="codificacion-003",
    chat_history=[],
    retrieval_mode="standard"  # Routing automatico
)
```

**Flujo interno:**
```
[Route] → Detecta "mapeo" + clasificaciones → Hybrid
    ↓
[Retrieve SQL] → Busca codigos con "hipertension"
[Retrieve Vector] → Documentacion sobre mapeos
    ↓
[Combine] → Respuesta con tabla de mapeo + explicacion
```

#### Ejemplo 5.4: Mapeo multi-turno con contexto
```python
conv_id = "auditoria-mapeos-001"
history = []

# Turno 1: Identificar codigo
r1 = invoke_rag(
    question="¿Codigo ICPC-3 para insuficiencia cardiaca?",
    conversation_id=conv_id,
    chat_history=history
)
history = r1["chat_history"]
# Respuesta: "K77"

# Turno 2: Pedir mapeo (sistema recuerda K77)
r2 = invoke_rag(
    question="¿Y su equivalente en CIE-10?",
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]

# Turno 3: Validar uso clinico
r3 = invoke_rag(
    question="¿Cuando se usa uno u otro?",
    conversation_id=conv_id,
    chat_history=history
)
```

### Modo de Recuperacion

**SQL Routing automatico para:**
- Consultas directas con codigos especificos
- "codigo ICPC-3 K86 a CIE-10"
- "mapeo E11 a ICPC-3"

**Hybrid para:**
- Consultas por descripcion clinica
- "mapeo hipertension ICPC-3 CIE-10"
- Cuando se necesita contexto adicional

**Vector para:**
- Diferencias conceptuales entre clasificaciones
- "¿Cuando usar ICPC-3 vs CIE-10?"
- Documentacion sobre los sistemas

### Configuracion Optima

```python
# Configuracion base
retrieval_mode = "standard"  # Permite routing automatico
k = 4  # Suficiente para mapeos

# Para consultas SQL directas
# El sistema detecta automaticamente y usa SQL
```

### Estructura de Datos de Mapeo

**Base de datos SQLite (`./db/icpc3.db`):**

```sql
-- Tabla ICPC-3
CREATE TABLE icpc3_codes (
    code TEXT PRIMARY KEY,
    title_es TEXT,
    component TEXT
);

-- Tabla CIE-10
CREATE TABLE icd10_codes (
    code TEXT PRIMARY KEY,
    title TEXT
);

-- Tabla de mapeo
CREATE TABLE icpc3_icd10_mapping (
    icpc3_code TEXT,
    icd10_code TEXT,
    mapping_type TEXT,  -- "exact", "approximate", "broader", "narrower"
    FOREIGN KEY (icpc3_code) REFERENCES icpc3_codes(code),
    FOREIGN KEY (icd10_code) REFERENCES icd10_codes(code)
);
```

### Resultados Esperados

#### Mapeo directo:
```
Ruta: sql
Query ejecutada: SELECT i.code, i.title_es, m.icd10_code, c.title...

Respuesta:
"El codigo ICPC-3 K86 (Hipertension sin complicaciones) se mapea a:

CIE-10: I10 - Hipertension esencial (primaria)

Tipo de mapeo: Exact match

Notas:
- ICPC-3 K86 es mas especifico para atencion primaria
- CIE-10 I10 se usa en codificacion hospitalaria
- Ambos representan la misma condicion clinica

[Fuente: Tabla icpc3_icd10_mapping]"
```

#### Mapeo multiple (1 a muchos):
```
Respuesta:
"El codigo ICPC-3 T90 (Diabetes no insulinodependiente) se mapea a
multiples codigos CIE-10:

1. E11.0 - Diabetes tipo 2 con coma
2. E11.1 - Diabetes tipo 2 con cetoacidosis
3. E11.2 - Diabetes tipo 2 con complicaciones renales
4. E11.3 - Diabetes tipo 2 con complicaciones oftalmicas
...

Tipo de mapeo: One-to-many (1:N)

Razon: CIE-10 tiene mayor granularidad para complicaciones especificas.

RECOMENDACION: Selecciona el codigo CIE-10 mas especifico segun las
complicaciones presentes en el paciente.

[Fuente: Tabla icpc3_icd10_mapping]"
```

### Tips de Optimizacion

1. **Especifica ambas clasificaciones:**
   ```python
   # BIEN
   "mapeo ICPC-3 K86 a CIE-10"

   # REGULAR (funciona pero menos preciso)
   "mapeo K86"
   ```

2. **Usa codigos exactos cuando los conoces:**
   - Routing SQL directo = Mas rapido
   - Resultados 100% precisos

3. **Para mapeos por descripcion:**
   - Usa terminologia clinica estandar
   - Activa hybrid routing

4. **Valida mapeos complejos:**
   ```python
   # Paso 1: Obtener mapeo
   r1 = invoke_rag("mapeo ICPC-3 D73 a CIE-10")

   # Paso 2: Validar uso clinico
   r2 = invoke_rag(
       "¿Cuando se usa cada codigo?",
       chat_history=r1["chat_history"]
   )
   ```

5. **Consultas en lote:**
   ```python
   # Mapear multiples codigos en una sesion
   conv_id = "mapeo-lote-001"
   history = []

   for code in ["K86", "K87", "T90"]:
       result = invoke_rag(
           f"mapeo ICPC-3 {code} a CIE-10",
           conv_id,
           history
       )
       history = result["chat_history"]
       # Procesa resultado...
   ```

### Casos Especiales

#### A. Mapeos 1:N (uno a muchos)
```python
# ICPC-3 mas general → Multiples CIE-10 especificos
result = invoke_rag(
    "mapeo completo ICPC-3 K76 (cardiopatia isquemica) a todos sus CIE-10"
)

# Puede retornar: I20, I21, I22, I23, I24, I25
```

#### B. Mapeos N:1 (muchos a uno)
```python
# Multiples ICPC-3 → Un CIE-10
result = invoke_rag(
    "que codigos ICPC-3 se mapean a CIE-10 J06 (infeccion respiratoria alta)"
)
```

#### C. Mapeos sin equivalencia exacta
```python
result = invoke_rag(
    "mapeo ICPC-3 A98 (medicina preventiva) a CIE-10"
)

# Respuesta esperada:
# "No existe mapeo directo. ICPC-3 A98 es un codigo de proceso,
#  mientras que CIE-10 codifica diagnosticos."
```

### Workflow Completo: Codificacion Clinica

```python
# Caso real: Codificar un episodio clinico

conv_id = "episodio-paciente-123"
history = []

# Paso 1: Identificar condicion principal
r1 = invoke_rag(
    question="codigo ICPC-3 para hipertension esencial",
    conversation_id=conv_id,
    chat_history=history
)
history = r1["chat_history"]
# Obtienes: K86

# Paso 2: Mapear a CIE-10 para registro hospitalario
r2 = invoke_rag(
    question="su equivalente en CIE-10",
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]
# Obtienes: I10

# Paso 3: Validar que el mapeo es correcto
r3 = invoke_rag(
    question="¿es correcto usar I10 para hipertension sin complicaciones?",
    conversation_id=conv_id,
    chat_history=history
)
```

### Integracion con Sistemas Externos

```python
# Ejemplo: Exportar mapeos a CSV
import csv

def mapear_lote(codigos_icpc3):
    """
    Mapea una lista de codigos ICPC-3 a CIE-10
    """
    results = []
    conv_id = f"batch-mapping-{timestamp}"
    history = []

    for code in codigos_icpc3:
        result = invoke_rag(
            question=f"mapeo ICPC-3 {code} a CIE-10",
            conversation_id=conv_id,
            chat_history=history
        )

        # Extraer codigos CIE-10 de la respuesta
        # (requeriria parsing)

        results.append({
            'icpc3': code,
            'icd10': extract_icd10(result['response']),
            'confidence': result['route_confidence']
        })

        history = result['chat_history']

    return results

# Uso
codigos = ["K86", "T90", "R96"]
mapeos = mapear_lote(codigos)

# Exportar
with open('mapeos.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['icpc3', 'icd10', 'confidence'])
    writer.writeheader()
    writer.writerows(mapeos)
```

### Recursos Adicionales

**Consultas sobre las clasificaciones:**
```python
# Entender diferencias conceptuales
invoke_rag("diferencias entre ICPC-3 y CIE-10")

# Cuando usar cada una
invoke_rag("cuando se usa ICPC-3 vs CIE-10 en Espana")

# Granularidad
invoke_rag("que clasificacion es mas detallada ICPC-3 o CIE-10")
```

---

## Caso de Uso 6: Investigacion de Evidencia Cientifica

### Descripcion
Busqueda y sintesis de evidencia cientifica para toma de decisiones basada en evidencia (MBE).

### Perfil de Usuario
- Investigadores clinicos
- Residentes preparando presentaciones
- Medicos consultando guias de practica

### Ejemplos de Consultas

#### Ejemplo 6.1: Revision de evidencia
```python
result = invoke_rag(
    question="¿Cual es la evidencia actual sobre el uso de estatinas en prevencion primaria cardiovascular?",
    conversation_id="investigacion-001",
    chat_history=[],
    retrieval_mode="rag-fusion"  # Combina multiples perspectivas
)
```

#### Ejemplo 6.2: Busqueda jerarquica de guias
```python
result = invoke_rag(
    question="Recomendaciones de guias clinicas para manejo de fibrilacion auricular",
    conversation_id="guias-001",
    chat_history=[],
    retrieval_mode="raptor"  # Recuperacion jerarquica
)
```

#### Ejemplo 6.3: Analisis de contradicciones
```python
conv_id = "revision-sistematica-001"
history = []

# Primera consulta
r1 = invoke_rag(
    question="beneficios de aspirina en prevencion cardiovascular",
    conversation_id=conv_id,
    chat_history=history
)
history = r1["chat_history"]

# Consulta sobre riesgos
r2 = invoke_rag(
    question="¿Y los riesgos?",
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]

# Balance riesgo-beneficio
r3 = invoke_rag(
    question="¿En que pacientes el beneficio supera el riesgo?",
    conversation_id=conv_id,
    chat_history=history
)
```

### Modo de Recuperacion

**RAG-Fusion (Recomendado):**
- Combina multiples variaciones de la consulta
- Ideal para revisiones sistematicas
- RRF (Reciprocal Rank Fusion) mejora recall

**RAPTOR:**
- Excelente para guias clinicas extensas
- Recupera sumarios jerarquicos
- Multi-nivel: detalles + sintesis

**HYDE:**
- Util cuando la terminologia no coincide exactamente
- Genera "abstract" hipotetico de respuesta
- Busqueda semantica mejorada

### Configuracion Optima

```python
# Para revision de evidencia
retrieval_mode = "rag-fusion"
k = 8  # Mas documentos = Mejor cobertura
similarity_threshold = 0.65  # Moderado para recall alto

# Para guias jerarquicas
retrieval_mode = "raptor"
k = 6

# Monitorizar fuentes
print(f"Fuentes citadas: {len(result['reranked_docs'])}")
for doc in result['reranked_docs']:
    print(f"- {doc.metadata['source']}")
```

### Resultados Esperados

```
Modo: rag-fusion
Documentos iniciales: 8
Re-ranked: 4
Fuentes:

Respuesta:
"EVIDENCIA ACTUAL: Estatinas en Prevencion Primaria Cardiovascular

1. ENSAYOS CLINICOS PRINCIPALES:
   - JUPITER (2008): Reduccion 44% eventos CV con rosuvastatina
   - HOPE-3 (2016): Beneficio en pacientes riesgo intermedio
   - Meta-analisis Cochrane (2022): NNT = 138 para prevenir 1 IAM

2. GUIAS ACTUALES:
   - ESC/EAS 2019: Recomendacion IIa para riesgo SCORE ≥5%
   - ACC/AHA 2019: Considerar si riesgo 10-años ≥7.5%

3. CALIDAD DE EVIDENCIA: Alta (grado A)

4. CONSIDERACIONES:
   - Beneficio mayor en riesgo alto (>10%)
   - Balance riesgo-beneficio individual
   - Miopatia: 0.1-0.2% (NNH = 500)

[Fuente 1: Guia ESC Dislipemias 2019]
[Fuente 2: Meta-analisis Cochrane Estatinas]
[Fuente 3: JUPITER Trial - NEJM]
[Fuente 4: ACC/AHA Guidelines]

IMPORTANTE: Esta informacion es educativa. Las decisiones clinicas
deben individualizarse segun perfil de riesgo del paciente."
```

### Tips de Optimizacion

1. **Usa RAG-Fusion para:**
   - Revisiones sistematicas
   - Preguntas con multiples enfoques
   - Sintesis de evidencia

2. **Usa RAPTOR para:**
   - Guias clinicas extensas (>100 pags)
   - Documentos con estructura jerarquica
   - Cuando necesitas "big picture" + detalles

3. **Incrementa K:**
   - K=8-10 para revisiones completas
   - Mas fuentes = Mayor credibilidad

4. **Valida fuentes:**
   ```python
   # Revisar metadata de documentos
   for doc in result['reranked_docs']:
       print(f"Fuente: {doc.metadata['source']}")
       print(f"Pagina: {doc.metadata.get('page', 'N/A')}")
       print(f"Score: {doc.metadata.get('relevance_score', 'N/A')}")
   ```

5. **Conversacion para refinamiento:**
   - Primera consulta amplia
   - Refina con turnos subsiguientes
   - Contrasta fuentes

---

## Caso de Uso 7: Formacion y Autoevaluacion

### Descripcion
Sistema de aprendizaje interactivo para estudiantes y residentes, con autoevaluacion y feedback.

### Perfil de Usuario
- Estudiantes de medicina
- Residentes en formacion
- Profesionales en educacion medica continua

### Ejemplos de Consultas

#### Ejemplo 7.1: Sesion de estudio estructurada
```python
# Programa de estudio: Insuficiencia Cardiaca
conv_id = "estudio-ic-dia1"
history = []

# Bloque 1: Fundamentos
r1 = invoke_rag(
    question="Explica la fisiopatologia de la insuficiencia cardiaca",
    conversation_id=conv_id,
    chat_history=history,
    retrieval_mode="raptor"  # Jerarquico: concepto → detalles
)
history = r1["chat_history"]

# Bloque 2: Diagnostico
r2 = invoke_rag(
    question="¿Como se diagnostica?",
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]

# Bloque 3: Clasificacion
r3 = invoke_rag(
    question="Explica la clasificacion NYHA",
    conversation_id=conv_id,
    chat_history=history
)
history = r3["chat_history"]

# Bloque 4: Tratamiento
r4 = invoke_rag(
    question="¿Y el tratamiento farmacologico?",
    conversation_id=conv_id,
    chat_history=history
)
```

#### Ejemplo 7.2: Preparacion de casos clinicos
```python
# Simulacion de caso clinico
conv_id = "caso-clinico-sim-001"
history = []

# Presentacion
r1 = invoke_rag(
    question="¿Cual es el diagnostico diferencial de un paciente con disnea, ortopnea y edemas?",
    conversation_id=conv_id,
    chat_history=history,
    retrieval_mode="crag"  # Auto-refina si es necesario
)
history = r1["chat_history"]

# Exploracion
r2 = invoke_rag(
    question="¿Que exploracion fisica esperarias?",
    conversation_id=conv_id,
    chat_history=history
)
history = r2["chat_history"]

# Pruebas
r3 = invoke_rag(
    question="¿Que pruebas complementarias pedirías?",
    conversation_id=conv_id,
    chat_history=history
)
history = r3["chat_history"]

# Manejo
r4 = invoke_rag(
    question="Si el BNP esta elevado y hay cardiomegalia, ¿cual es el manejo inicial?",
    conversation_id=conv_id,
    chat_history=history
)
```

#### Ejemplo 7.3: Repaso de conceptos clave
```python
# Lista de conceptos a repasar
conceptos = [
    "fisiopatologia asma",
    "tratamiento escalonado asma",
    "crisis asmatica manejo",
    "asma vs EPOC diferencias"
]

conv_id = f"repaso-neumologia-{fecha}"
history = []

for concepto in conceptos:
    result = invoke_rag(
        question=concepto,
        conversation_id=conv_id,
        chat_history=history,
        retrieval_mode="standard"
    )

    # Guardar para revision
    guardar_nota(concepto, result['response'])

    history = result['chat_history']
```

### Modo de Recuperacion

**RAPTOR (Recomendado para estudio):**
- Estructura jerarquica: concepto → detalles
- Perfecto para aprendizaje progresivo
- Sumarios de alto nivel + informacion especifica

**CRAG (Para autoevaluacion):**
- Refina automaticamente si la respuesta no es clara
- Util cuando el estudiante no sabe formular bien

**Standard:**
- Rapido para repasos
- Bueno para consultas directas

### Configuracion Optima

```python
# Sesion de estudio
retrieval_mode = "raptor"
k = 6
similarity_threshold = 0.7

# Autoevaluacion con CRAG
retrieval_mode = "crag"
k = 6  # CRAG puede aumentar si refina
```

### Patron de Estudio Recomendado

```python
# Tecnica Feynman adaptada
def sesion_feynman(tema):
    """
    Metodo Feynman con RAG
    """
    conv_id = f"feynman-{tema}-{timestamp}"
    history = []

    # Paso 1: Concepto basico
    r1 = invoke_rag(
        f"Explica {tema} de forma simple",
        conv_id,
        history,
        retrieval_mode="raptor"
    )
    history = r1["chat_history"]

    # Paso 2: Detalles tecnicos
    r2 = invoke_rag(
        "Ahora mas detalles tecnicos",
        conv_id,
        history
    )
    history = r2["chat_history"]

    # Paso 3: Aplicacion clinica
    r3 = invoke_rag(
        "¿Como se aplica esto en la practica clinica?",
        conv_id,
        history
    )
    history = r3["chat_history"]

    # Paso 4: Casos especiales
    r4 = invoke_rag(
        "¿Excepciones o casos especiales?",
        conv_id,
        history
    )

    return {
        'concepto': r1['response'],
        'detalles': r2['response'],
        'aplicacion': r3['response'],
        'excepciones': r4['response']
    }

# Uso
material = sesion_feynman("insuficiencia cardiaca")
```

### Resultados Esperados

**Sesion de estudio (RAPTOR):**
```
Iteracion: 0
Relevancia: 0.88
Documentos: 4 (2 sumarios de alto nivel + 2 detalles especificos)

Respuesta:
"FISIOPATOLOGIA DE LA INSUFICIENCIA CARDIACA

1. CONCEPTO BASICO:
   La IC es la incapacidad del corazon para bombear sangre suficiente
   para las necesidades metabolicas del organismo.

2. MECANISMOS:
   A. Disfuncion sistolica (FE <40%)
      - Perdida de contractilidad
      - Remodelado ventricular

   B. Disfuncion diastolica (FE preservada ≥50%)
      - Alteracion de la relajacion
      - Compliance disminuida

3. COMPENSACION NEUROHORMONAL:
   - Sistema renina-angiotensina-aldosterona ↑
   - Sistema nervioso simpatico ↑
   - Peptidos natriureticos ↑

4. CONSECUENCIAS:
   - Congestion pulmonar y/o sistemica
   - Bajo gasto cardiaco
   - Sintomas: disnea, fatiga, edemas

[Fuente 1: Guia ESC Insuficiencia Cardiaca]
[Fuente 2: Fisiopatologia IC - Manual Harrison]"
```

### Tips de Optimizacion

1. **Estructura sesiones tematicas:**
   - Un conversation_id por tema
   - Progresion logica: basico → avanzado

2. **Usa RAPTOR para:**
   - Conceptos complejos multi-nivel
   - Revisiones completas de temas

3. **Guarda tus sesiones:**
   ```python
   # Exportar conversacion completa
   def exportar_sesion(conv_id, chat_history):
       with open(f"{conv_id}.txt", 'w') as f:
           for user_q, assistant_a in chat_history:
               f.write(f"P: {user_q}\n")
               f.write(f"R: {assistant_a}\n\n")
   ```

4. **Repaso espaciado:**
   ```python
   # Scheduler de repaso
   # Dia 1, 3, 7, 14, 30
   temas_repaso = {
       'insuficiencia_cardiaca': fecha_ultimo_estudio,
       'diabetes': fecha_ultimo_estudio,
       # ...
   }

   def necesita_repaso(tema, dias_desde_ultimo):
       intervalos = [1, 3, 7, 14, 30]
       return dias_desde_ultimo in intervalos
   ```

5. **Autoevaluacion:**
   ```python
   # Preguntas de autoevaluacion
   preguntas = [
       "fisiopatologia diabetes tipo 2",
       "complicaciones microvasculares diabetes",
       "objetivos HbA1c en diabetes"
   ]

   for p in preguntas:
       result = invoke_rag(p, conv_id, history)
       # Compara tu respuesta mental con la del sistema
       history = result['chat_history']
   ```

### Integracion con Sistemas de Flashcards

```python
# Generar flashcards automaticos
def generar_flashcard(tema):
    """
    Crea pregunta-respuesta para Anki/Quizlet
    """
    result = invoke_rag(
        f"Resumen clave de {tema} en 3 puntos",
        conv_id=f"flashcard-{tema}",
        retrieval_mode="standard"
    )

    return {
        'front': tema,
        'back': result['response'],
        'sources': [doc.metadata['source'] for doc in result['reranked_docs']]
    }

# Exportar a Anki
temas = ["hipertension", "diabetes", "asma"]
flashcards = [generar_flashcard(t) for t in temas]

# Formato Anki CSV
with open('flashcards_anki.csv', 'w') as f:
    for card in flashcards:
        f.write(f"{card['front']};{card['back']}\n")
```

---

## Mejores Practicas Generales

### 1. Seleccion de Modo de Recuperacion

| Escenario | Modo Recomendado | Razon |
|-----------|------------------|-------|
| Consulta directa | `standard` | Rapido, eficiente |
| Codigos ICPC-3/CIE-10 | `standard` (SQL auto) | Routing automatico |
| Comparacion de temas | `rag-fusion` | Multiples perspectivas |
| Protocolos complejos | `raptor` | Jerarquico |
| Consulta vaga/ambigua | `crag` | Auto-refinamiento |
| Estudio progresivo | `raptor` | Concepto → Detalle |
| Revision de evidencia | `rag-fusion` | Mayor recall |
| Busqueda semantica dificil | `hyde` | Documentos hipoteticos |

### 2. Configuracion de K (numero de documentos)

```python
# K bajo (4): Consultas directas, rapidas
k = 4

# K medio (6): Balance velocidad/cobertura (RECOMENDADO)
k = 6

# K alto (8-10): Revisiones completas, comparaciones
k = 8
```

### 3. Umbrales de Similitud

```python
# Alta precision (0.8): Solo documentos muy relevantes
similarity_threshold = 0.8

# Balance (0.7): Recomendado general
similarity_threshold = 0.7

# Alto recall (0.6): Explorar mas documentos
similarity_threshold = 0.6

# CRAG no usa threshold (auto-ajusta)
```

### 4. Uso de Memoria Conversacional

**HACER:**
- Usar conversation_id consistente en una sesion
- Actualizar chat_history tras cada turno
- Reiniciar sesion al cambiar de tema

**NO HACER:**
- Cambiar conversation_id cada pregunta
- Olvidar pasar chat_history
- Sesiones muy largas (>15 turnos) sin reiniciar

### 5. Validacion de Respuestas

```python
# Siempre revisar:
print(f"Ruta: {result['route']}")  # sql/vector
print(f"Relevancia: {result['relevance_score']}")  # 0-1
print(f"Iteraciones: {result['retrieval_iteration']}")  # CRAG
print(f"Fuentes: {len(result['reranked_docs'])}")

# Verificar calidad
if result['relevance_score'] < 0.6:
    print("WARNING: Relevancia baja")
```

### 6. Manejo de Errores

```python
# Always check for errors
if result.get('error'):
    print(f"Error: {result['error']}")
    # Retry con modo diferente o query reformulada
```

### 7. Monitorizacion de Latencia

```python
# Ver donde se gasta el tiempo
for node, ms in result['latency_ms'].items():
    print(f"{node}: {ms:.1f}ms")

# Identificar cuellos de botella
# retrieve/rerank suelen ser los mas lentos
```

### 8. Citacion de Fuentes

**El sistema SIEMPRE cita fuentes:**
- Minimo 2 fuentes por respuesta
- Metadata incluye: source, page, chunk_id
- Verificar fuentes para decisiones criticas

### 9. Limitaciones del Sistema

**Recuerda que el sistema NO puede:**
- Proporcionar dosis especificas de medicamentos
- Realizar diagnosticos individuales
- Sustituir consulta con profesional sanitario
- Acceder a datos de pacientes reales

**Guardrails automaticos evitan:**
- Respuestas con dosis
- Diagnosticos directos
- Consejos medicos personalizados

### 10. Optimizacion de Costos

```python
# Reducir costos de API:
# 1. Usar K menor cuando sea posible
k = 4  # vs k=8

# 2. Sesiones conversacionales cortas
if len(history) > 10:
    history = []  # Reiniciar

# 3. Preferir SQL cuando aplique (mas barato que LLM)
# El routing automatico ya lo hace

# 4. Cache de consultas frecuentes (en tu aplicacion)
cache = {}
def query_con_cache(question):
    if question in cache:
        return cache[question]
    result = invoke_rag(question)
    cache[question] = result
    return result
```

---

## Recursos Adicionales

### Documentacion del Sistema
- **README.md**: Caracteristicas generales y arquitectura
- **QUICKSTART.md**: Inicio rapido en 5 minutos
- **LANGGRAPH_GUIDE.md**: Guia completa de LangGraph
- **UPGRADE_SUMMARY.md**: Novedades version 2.0

### Scripts de Ejemplo
```bash
# Demo interactiva
python demo_langgraph.py

# Tests completos
python test_langgraph.py

# Aplicacion web
streamlit run app_langgraph.py --server.port 5000
```

### Soporte y Comunidad
- Abrir issues en GitHub para bugs
- Consultar documentacion oficial de LangGraph
- Revisar papers: CRAG, Self-RAG, RAPTOR

---

## Glosario

- **RAG**: Retrieval-Augmented Generation
- **CRAG**: Corrective RAG (con refinamiento automatico)
- **RAPTOR**: Recursive Abstractive Processing for Tree-Organized Retrieval
- **HYDE**: Hypothetical Document Embeddings
- **RRF**: Reciprocal Rank Fusion
- **BGE**: Beijing Academy of Artificial Intelligence reranker
- **ICPC-3**: International Classification of Primary Care v3
- **CIE-10**: Clasificacion Internacional de Enfermedades v10
- **LangGraph**: Framework de orquestacion de LLM con grafos
- **Self-RAG**: RAG con auto-evaluacion de calidad

---

## Casos de Uso Futuros (Roadmap)

### En desarrollo:
1. **Web Search Fallback**: Busqueda en internet cuando no hay resultados locales
2. **Multi-agent Orchestration**: Multiples agentes especializados
3. **Streaming Responses**: Respuestas en tiempo real
4. **Voice Interface**: Interaccion por voz

### Propuestas:
- Integracion con historia clinica electronica
- Sistema de alertas de interacciones medicamentosas
- Generacion automatica de informes clinicos
- Traduccion multilingue (ES/EN/FR/DE)

---

**Version:** 1.0
**Fecha:** 2025-10-17
**Autor:** RAG NextHealth Team

**Licencia:** MIT

---

## Feedback y Contribuciones

Si tienes casos de uso adicionales o mejoras a estos ejemplos, por favor:
1. Abre un issue en GitHub
2. Describe tu caso de uso
3. Comparte ejemplos de consultas
4. Indica configuracion recomendada

Ayudanos a mejorar este documento para toda la comunidad.
