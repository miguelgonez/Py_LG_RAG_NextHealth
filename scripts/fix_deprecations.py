#!/usr/bin/env python3
"""Utility script to update deprecated LangChain imports and Streamlit options."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable

# Mapping from deprecated imports to their new package locations
LANGCHAIN_REPLACEMENTS: Dict[str, str] = {
    "from langchain.chat_models import ChatOpenAI": "from langchain_openai import ChatOpenAI",
    "from langchain.embeddings import HuggingFaceEmbeddings": "from langchain_huggingface import HuggingFaceEmbeddings",
    "from langchain.vectorstores import Chroma": "from langchain_chroma import Chroma",
    "from langchain.llms import OpenAI": "from langchain_openai import OpenAI",
    "from langchain.embeddings import OpenAIEmbeddings": "from langchain_openai import OpenAIEmbeddings",
    "from langchain_community.chat_models import ChatOpenAI": "from langchain_openai import ChatOpenAI",
    "from langchain_community.embeddings import HuggingFaceEmbeddings": "from langchain_huggingface import HuggingFaceEmbeddings",
    "from langchain_community.vectorstores import Chroma": "from langchain_chroma import Chroma",
}

STREAMLIT_TRUE_PATTERN = re.compile(r"use_container_width\s*=\s*True")
STREAMLIT_FALSE_PATTERN = re.compile(r"use_container_width\s*=\s*False")

EXCLUDED_DIRS = {".git", "__pycache__", "venv", "env", "node_modules", ".mypy_cache"}
MAIN_FILES = ("app.py", "main.py", "streamlit_app.py", "app_langgraph.py")
TOKENIZER_LINE = 'os.environ["TOKENIZERS_PARALLELISM"] = "false"'
TOKENIZER_BLOCK = f"{TOKENIZER_LINE}\n\n"

SCRIPT_PATH = Path(__file__).resolve()


def should_skip(path: Path) -> bool:
    """Return True if the file lives inside an excluded directory."""
    return any(part in EXCLUDED_DIRS for part in path.parts)


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield Python source files under root, respecting excluded directories."""
    for path in root.rglob("*.py"):
        if path.resolve() == SCRIPT_PATH:
            continue
        if should_skip(path.relative_to(root)):
            continue
        yield path


def fix_file_imports(file_path: Path, replacements: Dict[str, str]) -> bool:
    """Replace deprecated import statements in the given file."""
    if not file_path.exists():
        print(f"❌ Archivo no encontrado: {file_path}")
        return False

    try:
        original_content = file_path.read_text(encoding="utf-8")
    except Exception as err:  # noqa: BLE001
        print(f"❌ Error leyendo {file_path}: {err}")
        return False

    updated_content = original_content
    for old_import, new_import in replacements.items():
        updated_content = updated_content.replace(old_import, new_import)

    if updated_content == original_content:
        return False

    try:
        file_path.write_text(updated_content, encoding="utf-8")
        print(f"✅ Actualizado: {file_path}")
        return True
    except Exception as err:  # noqa: BLE001
        print(f"❌ Error escribiendo {file_path}: {err}")
        return False


def fix_streamlit_width(file_path: Path) -> bool:
    """Update deprecated `use_container_width` arguments in the target file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as err:  # noqa: BLE001
        print(f"❌ Error leyendo {file_path}: {err}")
        return False

    updated = STREAMLIT_TRUE_PATTERN.sub('width="stretch"', content)
    updated = STREAMLIT_FALSE_PATTERN.sub('width="content"', updated)

    if updated == content:
        return False

    try:
        file_path.write_text(updated, encoding="utf-8")
        print(f"✅ Actualizado Streamlit en: {file_path}")
        return True
    except Exception as err:  # noqa: BLE001
        print(f"❌ Error escribiendo {file_path}: {err}")
        return False


def ensure_tokenizer_setting(file_path: Path) -> bool:
    """Ensure the TOKENIZERS_PARALLELISM env var is disabled in the main Streamlit file."""
    if not file_path.exists():
        return False

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as err:  # noqa: BLE001
        print(f"❌ Error leyendo {file_path}: {err}")
        return False

    if "TOKENIZERS_PARALLELISM" in content:
        return False

    if re.search(r"^import os\b", content, re.MULTILINE):
        updated_content = re.sub(
            r"^import os\b.*$",
            lambda match: f"{match.group(0)}\n{TOKENIZER_BLOCK}",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        updated_content = f"import os\n{TOKENIZER_BLOCK}{content}"

    try:
        file_path.write_text(updated_content, encoding="utf-8")
        print(f"✅ Agregada configuración de tokenizers a: {file_path}")
        return True
    except Exception as err:  # noqa: BLE001
        print(f"❌ Error escribiendo {file_path}: {err}")
        return False


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    print("🔧 Iniciando corrección de deprecaciones...\n")

    # 1. Ensure TOKENIZERS_PARALLELISM is disabled in the first available main file
    for filename in MAIN_FILES:
        if ensure_tokenizer_setting(root / filename):
            break

    # 2. Update known LangChain imports in targeted files first
    print("📦 Actualizando imports de LangChain...")
    targets = [
        root / "src" / "graph.py",
        root / "src" / "retrievers.py",
        root / "src" / "chains.py",
        root / "src" / "models.py",
    ]

    for target in targets:
        if target.exists():
            fix_file_imports(target, LANGCHAIN_REPLACEMENTS)

    # 3. Scan remaining Python files for deprecated imports
    print("\n🔍 Buscando más archivos con imports deprecados...")
    for py_file in iter_python_files(root):
        fix_file_imports(py_file, LANGCHAIN_REPLACEMENTS)

    # 4. Update Streamlit width usages
    print("\n🎨 Actualizando uso de Streamlit...")
    for py_file in iter_python_files(root):
        fix_streamlit_width(py_file)

    print("\n✅ ¡Corrección de deprecaciones completada!")
    print("\n📋 Próximos pasos:")
    print("1. Instalar las nuevas dependencias:")
    print("   pip install -U langchain-openai langchain-huggingface langchain-chroma")
    print("2. Reiniciar tu aplicación Streamlit")
    print("3. Verificar que todo funcione correctamente")


if __name__ == "__main__":
    main()
