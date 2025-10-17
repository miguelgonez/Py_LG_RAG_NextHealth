"""Conversational RAG orchestration helpers."""

from __future__ import annotations

import operator
import os
from typing import Annotated, List, TypedDict

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .retrievers import RAGRetriever, create_retriever

# Configure tokenizers to avoid parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GraphState(TypedDict):
    """State carried across the conversational graph."""

    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    context: str
    query: str
    response: str


class RAGConversationalGraph:
    """LangGraph-based conversational RAG pipeline."""

    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_prompt()

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        self.graph = self._create_graph()

    @staticmethod
    def _default_prompt() -> str:
        return (
            "Eres un asistente médico experto especializado en Medicina del Estilo de Vida y salud integral.\n\n"
            "Tu rol es proporcionar información precisa, basada en evidencia científica.\n\n"
            "Contexto relevante:\n{context}\n\n"
            "Historial de conversación:\n{chat_history}\n\n"
            "Pregunta del usuario: {question}"
        )

    def _retrieve_context(self, state: GraphState) -> GraphState:
        query = state["query"]
        docs = self.retriever.retrieve(query)
        context = "\n\n---\n\n".join(
            f"Fuente: {doc.metadata.get('file_name', 'Unknown')}\n{doc.page_content}" for doc in docs
        )
        return {"context": context}

    def _generate_response(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        context = state.get("context", "")
        query = state["query"]

        if messages:
            chat_history = "\n".join(
                f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
                for msg in messages[-6:]
            )
        else:
            chat_history = "No hay historial previo."

        prompt = self.system_prompt.format(
            context=context or "Sin contexto disponible.",
            chat_history=chat_history,
            question=query,
        )

        response = self.llm.invoke([SystemMessage(content=prompt)])

        new_messages: List[HumanMessage | AIMessage] = [HumanMessage(content=query), response]
        return {"messages": new_messages, "response": response.content}

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def chat(self, query: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({"query": query}, config=config)
        return result["response"]

    def stream_chat(self, query: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}
        for event in self.graph.stream({"query": query}, config=config):
            payload = event.get("generate")
            if payload and "messages" in payload:
                ai_messages = [msg for msg in payload["messages"] if isinstance(msg, AIMessage)]
                if ai_messages:
                    yield ai_messages[-1].content

    def get_conversation_history(self, thread_id: str = "default") -> List[HumanMessage | AIMessage]:
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        return state.values.get("messages", [])

    def clear_history(self, thread_id: str = "default") -> None:
        """Reset the stored history for a thread (not yet implemented)."""
        _ = thread_id
        # MemorySaver does not expose clearing yet; placeholder for future support.


class SimpleRAGChain:
    """Lightweight conversational chain without LangGraph."""

    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Eres un asistente médico experto en Medicina del Estilo de Vida.\n\n"
                    "Usa el siguiente contexto para responder la pregunta del usuario:\n\n{context}\n\n"
                    "Si no encuentras la información en el contexto, dilo claramente.",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{question}"),
            ]
        )
        self.chat_history: List[HumanMessage | AIMessage] = []

    def chat(self, query: str) -> str:
        docs = self.retriever.retrieve(query)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        messages = self.prompt_template.format_messages(
            context=context or "Sin contexto disponible.",
            question=query,
            chat_history=self.chat_history[-6:],
        )
        response = self.llm.invoke(messages)
        self.chat_history.extend([HumanMessage(content=query), response])
        return response.content

    def clear_history(self) -> None:
        self.chat_history = []


def create_rag_chain(
    retriever: RAGRetriever,
    use_graph: bool = True,
    **kwargs,
) -> RAGConversationalGraph | SimpleRAGChain:
    """Create a conversational RAG pipeline."""
    if use_graph:
        return RAGConversationalGraph(retriever=retriever, **kwargs)
    return SimpleRAGChain(retriever=retriever, **kwargs)


class LegacyRAGChain:
    """Compatibility wrapper exposing an invoke API."""

    def __init__(self, retriever: RAGRetriever, use_graph: bool = True, **kwargs) -> None:
        self.retriever = retriever
        self.pipeline = create_rag_chain(retriever=self.retriever, use_graph=use_graph, **kwargs)

    def invoke(self, inputs: dict) -> AIMessage:
        query = inputs.get("question", "")
        response_text = self.pipeline.chat(query)
        return AIMessage(content=response_text)


def build_rag_chain(
    persist_dir: str,
    sqlite_path: str | None = None,
    retrieval_mode: str | None = None,
    use_graph: bool = True,
    **kwargs,
) -> LegacyRAGChain:
    """Legacy factory kept for backwards compatibility with the Streamlit UI."""
    retriever = create_retriever(persist_dir=persist_dir)
    return LegacyRAGChain(retriever=retriever, use_graph=use_graph, **kwargs)
