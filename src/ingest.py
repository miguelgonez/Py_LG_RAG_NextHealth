"""
Módulo de ingesta de documentos para RAG NextHealth
Maneja la carga, procesamiento y indexado de documentos clínicos
Soporta ingesta incremental y reindexado completo
"""

import argparse
import os
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredFileLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from src.state_store import DocumentStateStore


SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.html', '.htm'}


def load_single_file(file_path: Path) -> List[Document]:
    """
    Carga un único archivo y retorna sus documentos
    
    Args:
        file_path: Path al archivo
        
    Returns:
        Lista de documentos del archivo
    """
    try:
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            loader = UnstructuredFileLoader(str(file_path))
        
        docs = loader.load()
        
        for doc in docs:
            doc.metadata.update({
                'source_file': str(file_path),
                'file_type': file_path.suffix.lower(),
                'file_size': file_path.stat().st_size
            })
        
        return docs
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        return []


def load_specific_files(file_paths: List[Path]) -> List[Document]:
    """
    Carga una lista específica de archivos
    
    Args:
        file_paths: Lista de Paths a cargar
        
    Returns:
        Lista de documentos de todos los archivos
    """
    documents = []
    
    for file_path in file_paths:
        docs = load_single_file(file_path)
        if docs:
            documents.extend(docs)
            print(f"✅ Loaded {len(docs)} docs from {file_path.name}")
    
    return documents


def load_documents(data_dir: str) -> List:
    """
    Carga documentos desde el directorio especificado
    Soporta PDF, TXT, MD, HTML
    """
    data_path = Path(data_dir)
    documents = []
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} no existe")
    
    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs = load_single_file(file_path)
            if docs:
                documents.extend(docs)
                print(f"✅ Loaded {len(docs)} chunks from {file_path.name}")
    
    return documents


def split_documents(documents: List, chunk_size: int = 1200, chunk_overlap: int = 200) -> List:
    """
    Divide documentos en chunks usando RecursiveCharacterTextSplitter
    Optimizado para contenido clínico en español
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # Párrafos
            "\n",    # Líneas
            ". ",    # Oraciones
            ", ",    # Clausulas
            " ",     # Palabras
            ""       # Caracteres
        ],
        length_function=len,
    )
    
    splits = splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, split in enumerate(splits):
        split.metadata.update({
            'chunk_id': i,
            'chunk_size': len(split.page_content),
            'language': 'es'  # Assumimos español
        })
    
    return splits


def create_embeddings_model(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Crea el modelo de embeddings
    Por defecto usa multilingual-e5-base optimizado para español
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-base")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Usa CPU por compatibilidad
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        print(f"❌ Error loading embeddings model {model_name}: {str(e)}")
        # Fallback to simpler model
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"🔄 Trying fallback model: {fallback_model}")
        return HuggingFaceEmbeddings(model_name=fallback_model)


def generate_doc_id(file_path: str, chunk_index: int, file_hash: str) -> str:
    """
    Genera un ID determinístico para un chunk de documento
    
    Args:
        file_path: Ruta del archivo
        chunk_index: Índice del chunk
        file_hash: Hash del archivo
        
    Returns:
        ID único y determinístico
    """
    id_string = f"{file_path}::{chunk_index}::{file_hash}"
    return hashlib.md5(id_string.encode()).hexdigest()


def create_vector_store(splits: List, embeddings, persist_dir: str) -> Chroma:
    """
    Crea el vector store con ChromaDB
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="rag_nexthealth"
        )
        
        vectorstore.persist()
        
        return vectorstore
    
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        raise


def delete_file_from_vectorstore(vectorstore: Chroma, source_file: str):
    """
    Elimina todos los chunks de un archivo del vector store
    
    Args:
        vectorstore: Vector store de Chroma
        source_file: Ruta del archivo fuente a eliminar
    """
    try:
        results = vectorstore.get(where={"source_file": source_file})
        
        if results and results['ids']:
            vectorstore.delete(ids=results['ids'])
            print(f"   🗑️ Eliminados {len(results['ids'])} chunks antiguos")
    except Exception as e:
        print(f"   ⚠️ Warning eliminando chunks antiguos: {str(e)}")


def add_to_vector_store(
    splits: List[Document],
    embeddings,
    persist_dir: str,
    file_hash: str,
    source_file: str = None,
    delete_old: bool = True
) -> Chroma:
    """
    Agrega documentos a un vector store existente (ingesta incremental)
    Por defecto elimina versiones anteriores del archivo después de agregar exitosamente

    Args:
        splits: Chunks de documentos a agregar
        embeddings: Modelo de embeddings
        persist_dir: Directorio del vector store
        file_hash: Hash del archivo para generar IDs
        source_file: Ruta del archivo (para eliminar versiones anteriores)
        delete_old: Si True, elimina versiones antiguas DESPUÉS de agregar exitosamente

    Returns:
        Vector store actualizado
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="rag_nexthealth"
        )

        # Limpiar metadata para evitar problemas con '_type'
        cleaned_splits = []
        for split in splits:
            # Crear nuevo documento sin el campo _type problemático
            clean_metadata = {k: v for k, v in split.metadata.items() if not k.startswith('_')}
            clean_doc = Document(page_content=split.page_content, metadata=clean_metadata)
            cleaned_splits.append(clean_doc)

        ids = [
            generate_doc_id(
                split.metadata.get('source_file', 'unknown'),
                i,
                file_hash
            )
            for i, split in enumerate(cleaned_splits)
        ]

        vectorstore.add_documents(documents=cleaned_splits, ids=ids)
        
        if delete_old and source_file:
            try:
                results = vectorstore.get(where={"source_file": source_file})
                
                if results and results['ids']:
                    old_ids = [id for id in results['ids'] if id not in ids]
                    if old_ids:
                        vectorstore.delete(ids=old_ids)
                        print(f"   🗑️ Eliminados {len(old_ids)} chunks antiguos")
            except Exception as e:
                print(f"   ⚠️ Warning eliminando chunks antiguos: {str(e)}")
        
        vectorstore.persist()
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error adding to vector store: {str(e)}")
        raise


def ingest_incremental(
    persist_dir: str, 
    data_dir: str, 
    chunk_size: int = 1200,
    state_db_path: str = "./db/document_state.db"
) -> Tuple[Optional[Chroma], dict]:
    """
    Ingesta incremental: procesa solo documentos nuevos o modificados
    
    Args:
        persist_dir: Directorio del vector store
        data_dir: Directorio de documentos
        chunk_size: Tamaño de chunks
        state_db_path: Ruta al DB de estado
        
    Returns:
        Tupla (vectorstore, stats)
    """
    print(f"🔄 Iniciando ingesta INCREMENTAL...")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Persist directory: {persist_dir}")
    
    state_store = DocumentStateStore(state_db_path)
    
    print("\n1️⃣ Detectando cambios...")
    changed_files, all_files = state_store.get_changed_files(data_dir, SUPPORTED_EXTENSIONS)
    
    print(f"📊 Archivos encontrados: {len(all_files)}")
    print(f"🆕 Archivos nuevos/modificados: {len(changed_files)}")
    
    if not changed_files:
        print("\n✅ No hay documentos nuevos para procesar")
        stats = state_store.get_stats()
        return None, stats
    
    print("\n2️⃣ Cargando modelo de embeddings...")
    embeddings = create_embeddings_model()
    print("✅ Modelo de embeddings cargado")
    
    print("\n3️⃣ Limpiando archivos eliminados...")
    removed_files = state_store.remove_missing_files(data_dir)
    
    if removed_files and Path(persist_dir).exists():
        try:
            temp_vs = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="rag_nexthealth"
            )
            
            for file_path_str in removed_files:
                delete_file_from_vectorstore(temp_vs, file_path_str)
                print(f"   🗑️ Eliminado del vector store: {Path(file_path_str).name}")
            
            temp_vs.persist()
        except Exception as e:
            print(f"   ⚠️ Warning limpiando archivos eliminados: {str(e)}")
    
    if not removed_files:
        print("   ✅ No hay archivos eliminados")
    
    print("\n4️⃣ Procesando documentos nuevos/modificados...")
    total_chunks_added = 0
    
    for file_path in changed_files:
        print(f"\n📄 Procesando: {file_path.name}")
        
        docs = load_single_file(file_path)
        if not docs:
            continue
        
        splits = split_documents(docs, chunk_size=chunk_size)
        print(f"   Chunks: {len(splits)}")
        
        file_hash = state_store.calculate_file_hash(file_path)
        
        try:
            vectorstore = add_to_vector_store(
                splits, 
                embeddings, 
                persist_dir, 
                file_hash,
                source_file=str(file_path)
            )
            
            state_store.upsert_document_state(
                str(file_path),
                file_hash,
                file_path.stat().st_size,
                file_path.stat().st_mtime,
                len(splits)
            )
            
            total_chunks_added += len(splits)
            print(f"   ✅ Agregado al vector store")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    print(f"\n🎉 Ingesta incremental completada!")
    print(f"📊 Estadísticas:")
    print(f"   - Archivos procesados: {len(changed_files)}")
    print(f"   - Chunks agregados: {total_chunks_added}")
    
    stats = state_store.get_stats()
    print(f"   - Total documentos en sistema: {stats['total_documents']}")
    print(f"   - Total chunks en sistema: {stats['total_chunks']}")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="rag_nexthealth"
        )
        return vectorstore, stats
    except:
        return None, stats


def ingest_full_reindex(
    persist_dir: str, 
    data_dir: str, 
    chunk_size: int = 1200,
    state_db_path: str = "./db/document_state.db"
) -> Tuple[Optional[Chroma], dict]:
    """
    Reindexado completo: borra todo y procesa todos los documentos
    
    Args:
        persist_dir: Directorio del vector store
        data_dir: Directorio de documentos
        chunk_size: Tamaño de chunks
        state_db_path: Ruta al DB de estado
        
    Returns:
        Tupla (vectorstore, stats)
    """
    print(f"🔄 Iniciando REINDEXADO COMPLETO...")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Persist directory: {persist_dir}")
    
    state_store = DocumentStateStore(state_db_path)
    
    print("\n1️⃣ Limpiando estado anterior...")
    state_store.clear_all_state()
    
    import shutil
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
        print("   ✅ Vector store anterior eliminado")
    
    print("\n2️⃣ Cargando documentos...")
    documents = load_documents(data_dir)
    
    if not documents:
        print("❌ No se encontraron documentos válidos")
        return None, {'total_documents': 0, 'total_chunks': 0, 'total_size_bytes': 0}
    
    print(f"✅ Cargados {len(documents)} documentos")
    
    print("\n3️⃣ Dividiendo documentos en chunks...")
    splits = split_documents(documents, chunk_size=chunk_size)
    print(f"✅ Creados {len(splits)} chunks")
    
    print("\n4️⃣ Cargando modelo de embeddings...")
    embeddings = create_embeddings_model()
    print("✅ Modelo de embeddings cargado")
    
    print("\n5️⃣ Creando vector store...")
    vectorstore = create_vector_store(splits, embeddings, persist_dir)
    print(f"✅ Vector store creado con {len(splits)} chunks")
    
    print("\n6️⃣ Actualizando estado...")
    file_splits = {}
    for split in splits:
        source_file = split.metadata.get('source_file', 'unknown')
        if source_file not in file_splits:
            file_splits[source_file] = []
        file_splits[source_file].append(split)
    
    for file_path_str, file_chunks in file_splits.items():
        file_path = Path(file_path_str)
        if file_path.exists():
            file_hash = state_store.calculate_file_hash(file_path)
            state_store.upsert_document_state(
                str(file_path),
                file_hash,
                file_path.stat().st_size,
                file_path.stat().st_mtime,
                len(file_chunks)
            )
    
    print("\n7️⃣ Verificando vector store...")
    try:
        test_results = vectorstore.similarity_search("diabetes", k=3)
        print(f"✅ Test search exitoso: {len(test_results)} resultados")
    except Exception as e:
        print(f"⚠️ Warning en test search: {str(e)}")
    
    print(f"\n🎉 Reindexado completado exitosamente!")
    stats = state_store.get_stats()
    print(f"📊 Estadísticas:")
    print(f"   - Documentos procesados: {stats['total_documents']}")
    print(f"   - Chunks creados: {stats['total_chunks']}")
    print(f"   - Persist directory: {persist_dir}")
    
    return vectorstore, stats


def main(
    persist_dir: str, 
    data_dir: str, 
    chunk_size: int = 1200,
    mode: str = "incremental",
    state_db_path: str = "./db/document_state.db"
):
    """
    Función principal de ingesta con soporte para modo incremental y completo
    
    Args:
        persist_dir: Directorio del vector store
        data_dir: Directorio de documentos
        chunk_size: Tamaño de chunks
        mode: "incremental" o "full" 
        state_db_path: Ruta al DB de estado
    """
    if mode == "full":
        return ingest_full_reindex(persist_dir, data_dir, chunk_size, state_db_path)
    else:
        return ingest_incremental(persist_dir, data_dir, chunk_size, state_db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG NextHealth Document Ingestion")
    parser.add_argument("--persist_dir", type=str, default="./db/chroma", 
                       help="ChromaDB persist directory")
    parser.add_argument("--data_dir", type=str, default="./docs",
                       help="Directory containing documents to ingest")
    parser.add_argument("--chunk_size", type=int, default=1200,
                       help="Chunk size for text splitting")
    parser.add_argument("--mode", type=str, default="incremental",
                       choices=["incremental", "full"],
                       help="Ingestion mode: incremental (only new/changed files) or full (reindex all)")
    parser.add_argument("--state_db", type=str, default="./db/document_state.db",
                       help="Path to state database")
    
    args = parser.parse_args()
    
    main(args.persist_dir, args.data_dir, args.chunk_size, args.mode, args.state_db)
