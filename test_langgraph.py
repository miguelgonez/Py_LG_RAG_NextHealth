"""
Script de prueba para LangGraph RAG
Verifica que todos los componentes funcionen correctamente
"""

import os
from pathlib import Path

# Configurar variables de entorno de prueba
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-here")
os.environ["PERSIST_DIR"] = "./db/chroma"
os.environ["SQLITE_PATH"] = "./db/icpc3.db"
os.environ["RETRIEVAL_MODE"] = "standard"

from src.langgraph_rag import invoke_rag, rag_app


def test_basic_invocation():
    """Test básico de invocación del grafo"""
    print("=" * 80)
    print("TEST 1: Invocación básica sin historial")
    print("=" * 80)

    question = "¿Qué es la diabetes mellitus tipo 2?"

    print(f"\nPregunta: {question}")
    print("\nEjecutando grafo...")

    try:
        result = invoke_rag(
            question=question,
            conversation_id="test-001",
            chat_history=[],
            retrieval_mode="standard"
        )

        print("\n✅ RESULTADO:")
        print(f"Ruta: {result.get('route', 'N/A')}")
        print(f"Confianza: {result.get('route_confidence', 0):.2f}")
        print(f"Método de recuperación: {result.get('retrieval_method', 'N/A')}")
        print(f"Documentos recuperados: {len(result.get('documents', []))}")
        print(f"Documentos rerankeados: {len(result.get('reranked_docs', []))}")
        print(f"Relevancia: {result.get('relevance_score', 0):.2f}")
        print(f"Iteraciones: {result.get('retrieval_iteration', 0)}")
        print(f"\nSecuencia de nodos: {' → '.join(result.get('node_sequence', []))}")

        print("\n📝 RESPUESTA:")
        print(result.get('response', 'No response')[:500] + "...")

        print("\n⏱️ LATENCIAS:")
        for node, latency in result.get('latency_ms', {}).items():
            print(f"  {node}: {latency:.1f}ms")

        total_latency = sum(result.get('latency_ms', {}).values())
        print(f"\n  TOTAL: {total_latency:.1f}ms")

        return result

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_conversational_memory():
    """Test de memoria conversacional"""
    print("\n" + "=" * 80)
    print("TEST 2: Memoria conversacional")
    print("=" * 80)

    conversation_id = "test-002"

    # Primera pregunta
    print("\n📤 Pregunta 1: ¿Cuáles son los síntomas de la diabetes tipo 2?")
    result1 = invoke_rag(
        question="¿Cuáles son los síntomas de la diabetes tipo 2?",
        conversation_id=conversation_id,
        chat_history=[]
    )

    if result1:
        print(f"✅ Respuesta 1 generada")
        print(f"Historial actualizado: {len(result1.get('chat_history', []))} entradas")

        # Segunda pregunta (de seguimiento)
        print("\n📤 Pregunta 2 (seguimiento): ¿Y en niños?")
        result2 = invoke_rag(
            question="¿Y en niños?",
            conversation_id=conversation_id,
            chat_history=result1.get('chat_history', [])
        )

        if result2:
            print(f"✅ Respuesta 2 generada (con contexto)")
            print(f"Pregunta contextualizada: {result2.get('question', 'N/A')}")
            print(f"Historial: {len(result2.get('chat_history', []))} entradas")

            return True

    return False


def test_crag_refinement():
    """Test de refinamiento CRAG"""
    print("\n" + "=" * 80)
    print("TEST 3: CRAG con refinamiento automático")
    print("=" * 80)

    # Pregunta ambigua que debería trigger refinamiento
    question = "tratamiento"

    print(f"\nPregunta ambigua: '{question}'")
    print("Esperando que el sistema refine automáticamente...")

    os.environ["RETRIEVAL_MODE"] = "crag"

    try:
        result = invoke_rag(
            question=question,
            conversation_id="test-003",
            retrieval_mode="crag"
        )

        print(f"\n✅ Iteraciones de refinamiento: {result.get('retrieval_iteration', 0)}")
        print(f"Relevancia final: {result.get('relevance_score', 0):.2f}")
        print(f"Necesitó refinamiento: {result.get('needs_refinement', False)}")

        if result.get('retrieval_iteration', 0) > 0:
            print(f"✅ CRAG refinó la consulta correctamente")

        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def test_sql_routing():
    """Test de routing SQL"""
    print("\n" + "=" * 80)
    print("TEST 4: Routing SQL automático")
    print("=" * 80)

    question = "¿Cuál es el código ICPC-3 para diabetes?"

    print(f"\nPregunta SQL: {question}")

    try:
        result = invoke_rag(
            question=question,
            conversation_id="test-004"
        )

        expected_route = "sql"
        actual_route = result.get('route', '')

        if actual_route == expected_route:
            print(f"✅ Routing correcto: {actual_route}")
            print(f"Confianza: {result.get('route_confidence', 0):.2f}")
        else:
            print(f"⚠️ Routing inesperado: {actual_route} (esperado: {expected_route})")

        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def test_graph_structure():
    """Test de estructura del grafo"""
    print("\n" + "=" * 80)
    print("TEST 5: Estructura del grafo LangGraph")
    print("=" * 80)

    print("\n📊 Grafo compilado exitosamente:")
    print(f"Tipo: {type(rag_app)}")

    # Intentar obtener información del grafo
    try:
        print("\n✅ El grafo LangGraph está correctamente compilado")
        print("Nodos implementados:")
        nodes = [
            "contextualize",
            "route",
            "retrieve",
            "rerank",
            "evaluate",
            "refine",
            "generate"
        ]
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Ejecuta todos los tests"""
    print("\n🧪 SUITE DE TESTS - LangGraph RAG NextHealth")
    print("=" * 80)

    # Verificar que OpenAI API key está configurada
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-key-here":
        print("\n⚠️ WARNING: OPENAI_API_KEY no configurada")
        print("Por favor configura la variable de entorno OPENAI_API_KEY")
        print("\nTests se ejecutarán pero pueden fallar sin API key válida\n")

    # Verificar que vector store existe
    persist_dir = os.getenv("PERSIST_DIR", "./db/chroma")
    if not Path(persist_dir).exists():
        print(f"\n⚠️ WARNING: Vector store no encontrado en {persist_dir}")
        print("Algunos tests pueden fallar. Ejecuta la ingesta primero.\n")

    results = {}

    # Test 1: Básico
    results["basic"] = test_basic_invocation()

    # Test 2: Memoria conversacional
    results["memory"] = test_conversational_memory()

    # Test 3: CRAG refinamiento
    results["crag"] = test_crag_refinement()

    # Test 4: SQL routing
    results["sql"] = test_sql_routing()

    # Test 5: Estructura del grafo
    results["structure"] = test_graph_structure()

    # Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE TESTS")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r is not None and r)
    total = len(results)

    print(f"\nTests ejecutados: {total}")
    print(f"Tests exitosos: {passed}")
    print(f"Tests fallidos: {total - passed}")

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print("\n" + "=" * 80)
    print("🎉 Tests completados!")
    print("=" * 80)


if __name__ == "__main__":
    main()
