"""
Demostración rápida de LangGraph RAG con ejemplos prácticos
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar
print("🔧 Configurando entorno...")
os.environ["PERSIST_DIR"] = "./db/chroma"
os.environ["SQLITE_PATH"] = "./db/icpc3.db"

print("=" * 80)
print("🧠 DEMO: RAG NextHealth - LangGraph Edition")
print("=" * 80)

# Verificar que la API key está configurada
if not os.getenv("OPENAI_API_KEY"):
    print("\n⚠️ IMPORTANTE: Configura tu OPENAI_API_KEY!")
    print("Crea un archivo .env con:")
    print("OPENAI_API_KEY=sk-...")
    print("\nO ejecuta: export OPENAI_API_KEY='sk-...'\n")
    exit(1)

print(f"✅ API Key cargada: {os.getenv('OPENAI_API_KEY')[:10]}...")

from src.langgraph_rag import invoke_rag
import json


def demo_conversation():
    """Demo de conversación con memoria"""
    print("\n" + "=" * 80)
    print("📝 DEMO 1: Conversación Multi-turno con Memoria")
    print("=" * 80)

    conversation_id = "demo-001"

    # Pregunta 1
    print("\n👤 Usuario: ¿Qué es la diabetes mellitus tipo 2?")
    result1 = invoke_rag(
        question="¿Qué es la diabetes mellitus tipo 2?",
        conversation_id=conversation_id,
        chat_history=[]
    )

    if result1:
        print(f"\n🤖 Asistente:")
        print(f"Ruta: {result1['route']}")
        print(f"Relevancia: {result1['relevance_score']:.2f}")
        print(f"Respuesta (primeros 300 chars):")
        print(result1['response'][:300] + "...")

        # Pregunta 2 (seguimiento)
        print("\n" + "-" * 80)
        print("\n👤 Usuario: ¿Cuáles son sus síntomas principales?")
        result2 = invoke_rag(
            question="¿Cuáles son sus síntomas principales?",
            conversation_id=conversation_id,
            chat_history=result1['chat_history']  # ← Memoria!
        )

        if result2:
            print(f"\n🤖 Asistente:")
            print(f"Query contextualizada: {result2['question']}")
            print(f"Respuesta (primeros 300 chars):")
            print(result2['response'][:300] + "...")

            # Pregunta 3
            print("\n" + "-" * 80)
            print("\n👤 Usuario: ¿Y en niños?")
            result3 = invoke_rag(
                question="¿Y en niños?",
                conversation_id=conversation_id,
                chat_history=result2['chat_history']  # ← Memoria acumulada!
            )

            if result3:
                print(f"\n🤖 Asistente:")
                print(f"Query contextualizada: {result3['question']}")
                print(f"Historial: {len(result3['chat_history'])} intercambios")

    print("\n✅ Demo de conversación completada")


def demo_crag():
    """Demo de CRAG con refinamiento automático"""
    print("\n" + "=" * 80)
    print("🔄 DEMO 2: CRAG - Refinamiento Automático")
    print("=" * 80)

    # Query ambigua
    print("\n👤 Usuario: tratamiento")
    print("(Query muy vaga, debería trigger refinamiento automático)")

    result = invoke_rag(
        question="tratamiento",
        conversation_id="demo-002",
        retrieval_mode="crag"
    )

    if result:
        print(f"\n🤖 Sistema:")
        print(f"Iteraciones de refinamiento: {result['retrieval_iteration']}")
        print(f"Relevancia final: {result['relevance_score']:.2f}")
        print(f"Necesitó refinamiento: {result.get('needs_refinement', False)}")
        print(f"Flujo de nodos: {' → '.join(result['node_sequence'])}")

        if result['retrieval_iteration'] > 0:
            print(f"\n✅ CRAG refinó automáticamente la consulta")

    print("\n✅ Demo de CRAG completada")


def demo_routing():
    """Demo de routing SQL vs Vector"""
    print("\n" + "=" * 80)
    print("🧭 DEMO 3: Routing Inteligente (SQL vs Vector)")
    print("=" * 80)

    queries = [
        ("¿Cuál es el código ICPC-3 para hipertensión?", "sql"),
        ("¿Qué es la hipertensión arterial?", "vector"),
        ("Lista todos los códigos de diabetes", "sql"),
        ("¿Cómo se trata la diabetes tipo 2?", "vector")
    ]

    for query, expected_route in queries:
        print(f"\n👤 Query: {query}")
        print(f"   Ruta esperada: {expected_route}")

        result = invoke_rag(
            question=query,
            conversation_id="demo-003"
        )

        if result:
            actual_route = result['route']
            confidence = result['route_confidence']

            status = "✅" if actual_route == expected_route else "⚠️"
            print(f"   {status} Ruta detectada: {actual_route} (confianza: {confidence:.2f})")

    print("\n✅ Demo de routing completada")


def demo_latency():
    """Demo de métricas de latencia"""
    print("\n" + "=" * 80)
    print("⏱️ DEMO 4: Métricas de Latencia por Nodo")
    print("=" * 80)

    print("\n👤 Usuario: ¿Qué es la diabetes?")

    result = invoke_rag(
        question="¿Qué es la diabetes?",
        conversation_id="demo-004"
    )

    if result:
        print(f"\n📊 Latencias:")

        latencies = result.get('latency_ms', {})
        total = sum(latencies.values())

        # Ordenar por tiempo
        sorted_latencies = sorted(latencies.items(), key=lambda x: x[1], reverse=True)

        for node, ms in sorted_latencies:
            percentage = (ms / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)  # Scale para visualización
            bar = "█" * bar_length

            print(f"  {node:15s} {ms:7.1f}ms  {bar} {percentage:.1f}%")

        print(f"\n  {'TOTAL':15s} {total:7.1f}ms")

    print("\n✅ Demo de latencia completada")


def demo_self_rag():
    """Demo de Self-RAG evaluation"""
    print("\n" + "=" * 80)
    print("🎯 DEMO 5: Self-RAG - Auto-evaluación")
    print("=" * 80)

    queries = [
        "¿Cuáles son los síntomas de la diabetes tipo 2?",  # Buena pregunta
        "xyz123abc"  # Pregunta sin sentido
    ]

    for query in queries:
        print(f"\n👤 Query: {query}")

        result = invoke_rag(
            question=query,
            conversation_id="demo-005"
        )

        if result:
            print(f"   Relevancia: {result['relevance_score']:.2f}")
            print(f"   Puede responder: {result['can_answer']}")
            print(f"   Necesita refinamiento: {result.get('needs_refinement', False)}")
            print(f"   Documentos encontrados: {len(result.get('reranked_docs', []))}")

    print("\n✅ Demo de Self-RAG completada")


def main():
    """Ejecuta todas las demos"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🧠 RAG NextHealth - LangGraph" + " " * 28 + "║")
    print("║" + " " * 25 + "Demostración Interactiva" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")

    try:
        # Demo 1: Conversación
        demo_conversation()

        # Demo 2: CRAG
        demo_crag()

        # Demo 3: Routing
        demo_routing()

        # Demo 4: Latencia
        demo_latency()

        # Demo 5: Self-RAG
        demo_self_rag()

        # Resumen final
        print("\n" + "=" * 80)
        print("🎉 ¡TODAS LAS DEMOS COMPLETADAS!")
        print("=" * 80)

        print("""
✨ Features demostradas:
  ✅ Memoria conversacional multi-turno
  ✅ CRAG con refinamiento automático
  ✅ Routing inteligente SQL/Vector
  ✅ Métricas de latencia por nodo
  ✅ Self-RAG auto-evaluación

📚 Próximos pasos:
  1. Ejecuta: streamlit run app_langgraph.py
  2. Prueba tus propias queries
  3. Explora el flujo de nodos en la interfaz
  4. Revisa LANGGRAPH_GUIDE.md para más detalles

🔗 Documentación:
  - LANGGRAPH_GUIDE.md
  - README.md
        """)

    except Exception as e:
        print(f"\n❌ Error durante las demos: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
