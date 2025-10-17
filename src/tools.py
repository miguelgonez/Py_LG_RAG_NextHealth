"""
Herramientas SQL y utilidades para RAG NextHealth
Implementa Text-to-SQL con SQLite para consultas estructuradas sobre ICPC-3
"""

import os
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


def create_sample_icpc_db(sqlite_path: str):
    """
    Crea una base de datos SQLite de ejemplo con códigos ICPC-3 para demostración
    """
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    # Tabla principal de códigos ICPC-3
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS icpc_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code VARCHAR(10) UNIQUE NOT NULL,
            title_es TEXT NOT NULL,
            title_en TEXT,
            category VARCHAR(1),
            inclusion_criteria TEXT,
            exclusion_criteria TEXT,
            notes TEXT,
            icd10_mapping TEXT,
            snomed_mapping TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabla de mapeos a CIE-10
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS icpc_icd10_mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icpc_code VARCHAR(10),
            icd10_code VARCHAR(10),
            relationship_type VARCHAR(20),
            confidence REAL,
            FOREIGN KEY (icpc_code) REFERENCES icpc_codes (code)
        )
    ''')
    
    # Tabla de categorías
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS icpc_categories (
            category VARCHAR(1) PRIMARY KEY,
            name_es TEXT NOT NULL,
            name_en TEXT,
            description TEXT
        )
    ''')
    
    # Datos de ejemplo - Categorías principales
    categories_data = [
        ('A', 'General e inespecífico', 'General and unspecified', 'Síntomas generales y problemas no específicos'),
        ('B', 'Sangre y órganos hematopoyéticos', 'Blood and blood-forming organs', 'Problemas relacionados con sangre'),
        ('D', 'Sistema digestivo', 'Digestive system', 'Problemas del sistema digestivo'),
        ('F', 'Ojo', 'Eye', 'Problemas oculares'),
        ('H', 'Oído', 'Ear', 'Problemas auditivos'),
        ('K', 'Sistema cardiovascular', 'Cardiovascular system', 'Problemas cardiovasculares'),
        ('L', 'Sistema musculoesquelético', 'Musculoskeletal system', 'Problemas músculo-esqueléticos'),
        ('N', 'Sistema nervioso', 'Neurological system', 'Problemas neurológicos'),
        ('P', 'Problemas psicológicos', 'Psychological', 'Problemas de salud mental'),
        ('R', 'Sistema respiratorio', 'Respiratory system', 'Problemas respiratorios'),
        ('S', 'Piel', 'Skin', 'Problemas dermatológicos'),
        ('T', 'Endocrino, metabólico y nutricional', 'Endocrine/metabolic/nutritional', 'Problemas endocrinos'),
        ('U', 'Sistema urinario', 'Urological', 'Problemas urológicos'),
        ('W', 'Embarazo, parto y planificación familiar', 'Pregnancy/childbirth/family planning', 'Problemas relacionados con embarazo'),
        ('X', 'Sistema genital femenino', 'Female genital system', 'Problemas ginecológicos'),
        ('Y', 'Sistema genital masculino', 'Male genital system', 'Problemas urológicos masculinos'),
        ('Z', 'Problemas sociales', 'Social problems', 'Problemas sociales y familiares')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO icpc_categories (category, name_es, name_en, description) 
        VALUES (?, ?, ?, ?)
    ''', categories_data)
    
    # Datos de ejemplo - Códigos ICPC-3 más comunes
    icpc_data = [
        ('T90', 'Diabetes mellitus tipo 2', 'Diabetes mellitus type 2', 'T', 
         'Diabetes no insulinodependiente, resistencia a la insulina', 
         'Diabetes tipo 1, diabetes gestacional', 
         'Incluye todas las formas de diabetes tipo 2', 'E11', '44054006'),
        
        ('T89', 'Diabetes mellitus tipo 1', 'Diabetes mellitus type 1', 'T',
         'Diabetes insulinodependiente, déficit absoluto de insulina',
         'Diabetes tipo 2, diabetes MODY',
         'Requiere insulina desde el diagnóstico', 'E10', '46635009'),
        
        ('K86', 'Hipertensión arterial sin complicaciones', 'Hypertension uncomplicated', 'K',
         'Presión arterial sistólica ≥140 mmHg y/o diastólica ≥90 mmHg',
         'Hipertensión secundaria, hipertensión en embarazo',
         'Incluye hipertensión esencial', 'I10', '38341003'),
        
        ('K87', 'Hipertensión arterial con complicaciones', 'Hypertension complicated', 'K',
         'Hipertensión con daño de órgano diana',
         'Hipertensión sin complicaciones',
         'Incluye retinopatía, nefropatía hipertensiva', 'I11-I15', '194774006'),
        
        ('R96', 'Asma', 'Asthma', 'R',
         'Enfermedad inflamatoria crónica de vías aéreas',
         'EPOC, bronquitis aguda',
         'Incluye asma alérgica y no alérgica', 'J45', '195967001'),
        
        ('L84', 'Dolor de espalda con irradiación', 'Back pain with radiation', 'L',
         'Dolor lumbar que se irradia a extremidades',
         'Dolor de espalda sin irradiación',
         'Incluye ciática y radiculopatía', 'M54.4', '23056005'),
        
        ('L83', 'Síndrome del cuello', 'Neck syndrome', 'L',
         'Dolor y rigidez cervical',
         'Fracturas cervicales, tumores',
         'Incluye contractura cervical', 'M54.2', '81680005'),
        
        ('P74', 'Trastorno de ansiedad/estado de ansiedad', 'Anxiety disorder/anxiety state', 'P',
         'Ansiedad persistente que interfiere con funcionamiento',
         'Ansiedad adaptativa normal',
         'Incluye trastorno de ansiedad generalizada', 'F41', '48694002'),
        
        ('P76', 'Trastorno depresivo', 'Depressive disorder', 'P',
         'Episodio depresivo mayor o distimia',
         'Tristeza adaptativa, duelo normal',
         'Incluye depresión mayor y menor', 'F32-F33', '35489007'),
        
        ('S87', 'Dermatitis atópica/eccema', 'Atopic dermatitis/eczema', 'S',
         'Inflamación crónica de la piel con prurito',
         'Dermatitis de contacto, psoriasis',
         'Incluye eccema constitucional', 'L20', '24079001')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO icpc_codes 
        (code, title_es, title_en, category, inclusion_criteria, exclusion_criteria, notes, icd10_mapping, snomed_mapping) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', icpc_data)
    
    # Mapeos ICPC-3 a CIE-10
    mapping_data = [
        ('T90', 'E11.9', 'exact', 0.95),
        ('T89', 'E10.9', 'exact', 0.95),
        ('K86', 'I10', 'exact', 0.90),
        ('K87', 'I11.9', 'broader', 0.85),
        ('R96', 'J45.9', 'exact', 0.90),
        ('L84', 'M54.4', 'exact', 0.88),
        ('L83', 'M54.2', 'exact', 0.90),
        ('P74', 'F41.9', 'exact', 0.85),
        ('P76', 'F32.9', 'broader', 0.80),
        ('S87', 'L20.9', 'exact', 0.92)
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO icpc_icd10_mapping 
        (icpc_code, icd10_code, relationship_type, confidence) 
        VALUES (?, ?, ?, ?)
    ''', mapping_data)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Base de datos ICPC-3 creada exitosamente en {sqlite_path}")
    print(f"   - {len(categories_data)} categorías")
    print(f"   - {len(icpc_data)} códigos ICPC-3")
    print(f"   - {len(mapping_data)} mapeos a CIE-10")


def get_sql_chain(sqlite_path: str):
    """
    Crea una cadena de Text-to-SQL usando LangChain
    """
    if not sqlite_path.startswith("sqlite:///"):
        sqlite_path = f"sqlite:///{sqlite_path}"
    
    try:
        db = SQLDatabase.from_uri(sqlite_path)
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Custom prompt for Spanish medical queries
        SQL_PROMPT = PromptTemplate(
            input_variables=["input", "table_info", "top_k"],
            template="""Eres un experto en bases de datos médicas ICPC-3. 
            Dada una pregunta en español sobre códigos médicos, genera una consulta SQL precisa.

            Información de las tablas:
            {table_info}

            Instrucciones específicas:
            - Usa LIKE con % para búsquedas de texto parciales
            - Para búsquedas por síntomas/condiciones, busca en title_es, inclusion_criteria y notes
            - Para mapeos, usa las tablas de relación icpc_icd10_mapping
            - Limita los resultados a {top_k} si no se especifica
            - Incluye información relevante como categoría y criterios
            - Si buscas por categoría de enfermedad, filtra por el campo 'category'

            Pregunta: {input}

            SQL Query (solo la consulta, sin explicaciones):"""
        )
        
        chain = create_sql_query_chain(llm, db, prompt=SQL_PROMPT)
        
        return chain, db
        
    except Exception as e:
        print(f"❌ Error creando SQL chain: {str(e)}")
        raise


def run_sql(sqlite_path: str, sql: str) -> pd.DataFrame:
    """
    Ejecuta una consulta SQL y devuelve los resultados como DataFrame
    """
    if not sqlite_path.startswith("sqlite:///"):
        sqlite_path = f"sqlite:///{sqlite_path}"
    
    try:
        engine = create_engine(sqlite_path)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df
    
    except Exception as e:
        print(f"❌ Error ejecutando SQL: {str(e)}")
        raise


def analyze_sql_intent(query: str) -> Dict[str, Any]:
    """
    Analiza la intención de la consulta para determinar si debe usar SQL o vectorstore
    """
    sql_keywords = [
        'código', 'codigo', 'icpc', 'cie-10', 'cie10', 'icd-10', 'icd10',
        'snomed', 'mapeo', 'mapear', 'equivalencia', 'tabla', 'categoría',
        'categoria', 'clasificación', 'clasificacion', 'estadística',
        'estadistica', 'contar', 'cuántos', 'cuantos', 'listar', 'todos los'
    ]
    
    vector_keywords = [
        'explicar', 'explicación', 'explicacion', 'qué es', 'que es',
        'cómo', 'como', 'síntomas', 'sintomas', 'tratamiento', 'causa',
        'diagnóstico', 'diagnostico', 'criterios', 'recomendación',
        'recomendacion', 'guía', 'guia', 'protocolo'
    ]
    
    query_lower = query.lower()
    
    sql_score = sum(1 for keyword in sql_keywords if keyword in query_lower)
    vector_score = sum(1 for keyword in vector_keywords if keyword in query_lower)
    
    # Patrones específicos
    if any(pattern in query_lower for pattern in ['código para', 'codigo para', 'mapeo de', 'equivale a']):
        sql_score += 2
    
    if any(pattern in query_lower for pattern in ['explica', 'describe', 'criterios de', 'cómo se', 'como se']):
        vector_score += 2
    
    intent = "sql" if sql_score > vector_score else "vector"
    confidence = abs(sql_score - vector_score) / max(sql_score + vector_score, 1)
    
    return {
        "intent": intent,
        "confidence": confidence,
        "sql_score": sql_score,
        "vector_score": vector_score,
        "reasoning": f"SQL keywords: {sql_score}, Vector keywords: {vector_score}"
    }


def get_table_schema(sqlite_path: str) -> Dict[str, Any]:
    """
    Obtiene el esquema de la base de datos para mostrar información al usuario
    """
    try:
        engine = create_engine(f"sqlite:///{sqlite_path}")
        
        with engine.connect() as conn:
            # Get table names
            tables_df = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table';"), conn)
            tables = tables_df['name'].tolist()
            
            schema = {"tables": {}}
            
            for table in tables:
                # Get column info
                columns_df = pd.read_sql(text(f"PRAGMA table_info({table});"), conn)
                schema["tables"][table] = {
                    "columns": columns_df[['name', 'type']].to_dict('records'),
                    "count": pd.read_sql(text(f"SELECT COUNT(*) as count FROM {table};"), conn)['count'].iloc[0]
                }
        
        return schema
        
    except Exception as e:
        print(f"❌ Error obteniendo esquema: {str(e)}")
        return {"error": str(e)}


# Funciones de utilidad para análisis
def get_icpc_statistics(sqlite_path: str) -> Dict[str, Any]:
    """
    Obtiene estadísticas básicas de la base de datos ICPC-3
    """
    try:
        stats = {}
        
        # Total codes by category
        df = run_sql(sqlite_path, """
            SELECT c.name_es as categoria, COUNT(i.code) as total_codes
            FROM icpc_categories c
            LEFT JOIN icpc_codes i ON c.category = i.category
            GROUP BY c.category, c.name_es
            ORDER BY total_codes DESC;
        """)
        
        stats["codes_by_category"] = df.to_dict('records')
        
        # Total mappings
        mappings_df = run_sql(sqlite_path, "SELECT COUNT(*) as total FROM icpc_icd10_mapping;")
        stats["total_mappings"] = mappings_df['total'].iloc[0]
        
        # Confidence distribution
        confidence_df = run_sql(sqlite_path, """
            SELECT 
                CASE 
                    WHEN confidence >= 0.9 THEN 'Alta (≥0.9)'
                    WHEN confidence >= 0.8 THEN 'Media (0.8-0.9)'
                    ELSE 'Baja (<0.8)'
                END as confidence_level,
                COUNT(*) as count
            FROM icpc_icd10_mapping
            GROUP BY confidence_level;
        """)
        
        stats["confidence_distribution"] = confidence_df.to_dict('records')
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}
