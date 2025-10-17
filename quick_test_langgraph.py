#!/usr/bin/env python3
"""Test rápido de LangGraph RAG"""

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["PERSIST_DIR"] = "./db/chroma"

print("🧠 LangGraph RAG - Quick Test")
print("="*60)

from src.langgraph_rag import invoke_rag

# Test simple
question = "¿Qué es la salud positiva?"

print(f"Pregunta: {question}\n")
print("Procesando...")

result = invoke_rag(
    question=question,
    conversation_id="test",
    chat_history=[]
)

print("\n✅ Resultado:")
print(f"Ruta: {result['route']}")
print(f"Relevancia: {result['relevance_score']:.2f}")
print(f"Docs: {len(result['reranked_docs'])}")
print(f"Nodos: {' → '.join(result['node_sequence'])}")

print("\nRespuesta:")
print("-"*60)
print(result['response'][:400] + "...")
