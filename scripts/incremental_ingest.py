#!/usr/bin/env python3
"""Incremental ingestion utility for maintaining a Chroma vector store."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configure tokenizers before importing models
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IncrementalRAGIngestor:
    """Incremental ingestion pipeline with change detection and cleanup."""

    def __init__(
        self,
        data_dir: str = "./docs",
        persist_dir: str = "./db/chroma",
        metadata_file: str = "./db/metadata.json",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 5000,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.metadata_file = Path(metadata_file)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        self.file_metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load metadata for previously processed files."""
        if self.metadata_file.exists():
            with self.metadata_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _save_metadata(self) -> None:
        """Persist metadata about processed files."""
        with self.metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(self.file_metadata, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _get_file_hash(file_path: Path) -> str:
        """Calculate an MD5 hash for a file path."""
        digest = hashlib.md5()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _detect_changes(self) -> Tuple[List[Path], List[str]]:
        """Detect new/modified and deleted files."""
        current_files: Dict[str, str] = {}
        files_to_process: List[Path] = []

        for file_path in self.data_dir.glob("**/*.pdf"):
            file_hash = self._get_file_hash(file_path)
            file_key = str(file_path.relative_to(self.data_dir))
            current_files[file_key] = file_hash

            if file_key not in self.file_metadata or self.file_metadata[file_key]["hash"] != file_hash:
                files_to_process.append(file_path)

        deleted_files = [key for key in self.file_metadata if key not in current_files]
        return files_to_process, deleted_files

    @staticmethod
    def _add_documents_in_batches(vectorstore: Chroma, documents: List[Document], batch_size: int) -> None:
        """Add documents to the vector store in batches."""
        total_docs = len(documents)
        if total_docs == 0:
            return

        total_batches = (total_docs + batch_size - 1) // batch_size
        for index in range(0, total_docs, batch_size):
            batch = documents[index : index + batch_size]
            vectorstore.add_documents(batch)
            batch_no = index // batch_size + 1
            print(f"   ✅ Batch {batch_no}/{total_batches} agregado: {len(batch)} chunks")

    @staticmethod
    def _remove_documents(vectorstore: Chroma, file_key: str) -> None:
        """Remove all documents for a given file from the vector store."""
        try:
            vectorstore.delete(where={"source": file_key})
            print(f"   ✅ Eliminado: {file_key}")
        except Exception as exc:  # noqa: BLE001
            print(f"   ❌ Error eliminando {file_key}: {exc}")

    def process_document(self, file_path: Path) -> List[Document]:
        """Load a PDF and split it into chunks."""
        print(f"📄 Procesando: {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)

        file_key = str(file_path.relative_to(self.data_dir))
        for chunk in chunks:
            chunk.metadata["source"] = file_key
            chunk.metadata["file_name"] = file_path.name

        print(f"   Chunks: {len(chunks)}")
        return chunks

    def ingest(self, force_rebuild: bool = False) -> None:
        """Run the incremental ingestion pipeline."""
        print("🔄 Iniciando ingesta INCREMENTAL...")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"💾 Persist directory: {self.persist_dir}")

        if force_rebuild:
            print("⚠️  MODO REBUILD: Eliminando datos existentes...")
            if self.persist_dir.exists():
                import shutil

                shutil.rmtree(self.persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.file_metadata = {}

        print("\n1️⃣ Detectando cambios...")
        files_to_process, deleted_files = self._detect_changes()

        all_files = list(self.data_dir.glob("**/*.pdf"))
        print(f"📊 Archivos encontrados: {len(all_files)}")
        print(f"🆕 Archivos nuevos/modificados: {len(files_to_process)}")
        print(f"🗑️  Archivos eliminados: {len(deleted_files)}")

        if not files_to_process and not deleted_files:
            print("✅ No hay cambios. Vector store está actualizado.")
            return

        print("\n2️⃣ Cargando modelo de embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ Modelo de embeddings cargado")

        print("\n3️⃣ Inicializando vector store...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
            collection_name="rag_collection",
        )

        print("\n4️⃣ Limpiando archivos eliminados...")
        if deleted_files:
            for file_key in deleted_files:
                self._remove_documents(vectorstore, file_key)
                self.file_metadata.pop(file_key, None)
            self._save_metadata()
        else:
            print("   ✅ No hay archivos eliminados")

        if files_to_process:
            print("\n5️⃣ Procesando documentos nuevos/modificados...")
            for file_path in files_to_process:
                try:
                    chunks = self.process_document(file_path)
                    self._add_documents_in_batches(vectorstore, chunks, self.batch_size)

                    file_key = str(file_path.relative_to(self.data_dir))
                    self.file_metadata[file_key] = {
                        "hash": self._get_file_hash(file_path),
                        "chunks": len(chunks),
                        "last_processed": datetime.now().isoformat(),
                    }
                    print(f"   ✅ Completado: {file_path.name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"   ❌ Error procesando {file_path.name}: {exc}")

            self._save_metadata()

        print("\n✅ Ingesta completada exitosamente!")
        print(f"📊 Total de archivos en metadata: {len(self.file_metadata)}")


def main() -> None:
    """Entry point for running the ingestion pipeline from CLI."""
    ingestor = IncrementalRAGIngestor()
    ingestor.ingest(force_rebuild=False)


if __name__ == "__main__":
    main()
