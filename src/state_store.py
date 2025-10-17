"""
Módulo de gestión de estado para indexación incremental
Mantiene un catálogo de documentos procesados con hash, metadata y timestamps
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class DocumentStateStore:
    """
    Gestiona el estado de documentos procesados para ingesta incremental
    Usa SQLite para persistir metadata: path, hash, size, mtime
    """
    
    def __init__(self, db_path: str = "./db/document_state.db"):
        """
        Inicializa el store de estado
        
        Args:
            db_path: Ruta al archivo SQLite
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Crea la tabla de estado si no existe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_state (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_mtime REAL NOT NULL,
                processed_at TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calcula el hash SHA256 de un archivo
        
        Args:
            file_path: Path al archivo
            
        Returns:
            Hash hexadecimal del archivo
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def get_document_state(self, file_path: str) -> Optional[Dict]:
        """
        Obtiene el estado de un documento procesado
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Dict con metadata o None si no existe
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT file_hash, file_size, file_mtime, processed_at, chunk_count FROM document_state WHERE file_path = ?",
            (str(file_path),)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'file_path': file_path,
                'file_hash': row[0],
                'file_size': row[1],
                'file_mtime': row[2],
                'processed_at': row[3],
                'chunk_count': row[4]
            }
        
        return None
    
    def upsert_document_state(
        self, 
        file_path: str, 
        file_hash: str, 
        file_size: int, 
        file_mtime: float,
        chunk_count: int = 0
    ):
        """
        Inserta o actualiza el estado de un documento
        
        Args:
            file_path: Ruta del archivo
            file_hash: Hash SHA256 del archivo
            file_size: Tamaño en bytes
            file_mtime: Timestamp de modificación
            chunk_count: Número de chunks creados
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        processed_at = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO document_state 
            (file_path, file_hash, file_size, file_mtime, processed_at, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(file_path), file_hash, file_size, file_mtime, processed_at, chunk_count))
        
        conn.commit()
        conn.close()
    
    def has_file_changed(self, file_path: Path) -> bool:
        """
        Detecta si un archivo es nuevo o ha sido modificado
        
        Args:
            file_path: Path al archivo
            
        Returns:
            True si el archivo es nuevo o cambió, False si no ha cambiado
        """
        if not file_path.exists():
            return False
        
        state = self.get_document_state(str(file_path))
        
        if state is None:
            return True
        
        current_hash = self.calculate_file_hash(file_path)
        current_size = file_path.stat().st_size
        current_mtime = file_path.stat().st_mtime
        
        return (
            current_hash != state['file_hash'] or
            current_size != state['file_size'] or
            abs(current_mtime - state['file_mtime']) > 1.0
        )
    
    def get_changed_files(self, data_dir: str, supported_extensions: set) -> Tuple[List[Path], List[Path]]:
        """
        Escanea un directorio y detecta archivos nuevos o modificados
        
        Args:
            data_dir: Directorio a escanear
            supported_extensions: Set de extensiones soportadas (ej: {'.pdf', '.txt'})
            
        Returns:
            Tupla: (archivos_cambiados, todos_los_archivos)
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return [], []
        
        all_files = []
        changed_files = []
        
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                all_files.append(file_path)
                
                if self.has_file_changed(file_path):
                    changed_files.append(file_path)
        
        return changed_files, all_files
    
    def get_all_processed_files(self) -> List[Dict]:
        """
        Obtiene la lista completa de archivos procesados
        
        Returns:
            Lista de dicts con metadata de todos los documentos
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT file_path, file_hash, file_size, file_mtime, processed_at, chunk_count FROM document_state"
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'file_path': row[0],
                'file_hash': row[1],
                'file_size': row[2],
                'file_mtime': row[3],
                'processed_at': row[4],
                'chunk_count': row[5]
            }
            for row in rows
        ]
    
    def clear_all_state(self):
        """Borra todo el estado (para reindexado completo)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM document_state")
        
        conn.commit()
        conn.close()
    
    def remove_missing_files(self, data_dir: str) -> List[str]:
        """
        Identifica y elimina del estado los archivos que ya no existen en el filesystem
        
        Args:
            data_dir: Directorio de datos
            
        Returns:
            Lista de rutas de archivos que fueron eliminados
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_path FROM document_state")
        rows = cursor.fetchall()
        
        removed_files = []
        
        for row in rows:
            file_path = Path(row[0])
            if not file_path.exists():
                removed_files.append(str(file_path))
                cursor.execute("DELETE FROM document_state WHERE file_path = ?", (str(file_path),))
                print(f"🗑️ Eliminado del estado: {file_path.name} (archivo no existe)")
        
        conn.commit()
        conn.close()
        
        return removed_files
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas del estado
        
        Returns:
            Dict con estadísticas
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(chunk_count), SUM(file_size) FROM document_state")
        row = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_documents': row[0] or 0,
            'total_chunks': row[1] or 0,
            'total_size_bytes': row[2] or 0
        }
