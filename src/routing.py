"""
Intelligent routing module for deciding between vectorstore and SQL queries
"""

import re
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class QueryType(Enum):
    """
    Enum for different query types
    """
    VECTORSTORE = "vectorstore"  # For semantic/content search
    SQL = "sql"                  # For structured data queries
    HYBRID = "hybrid"           # For queries needing both


@dataclass
class RoutingDecision:
    """
    Data class for routing decisions
    """
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_modifications: List[str] = None


class QueryRouter:
    """
    Intelligent query router that decides between vectorstore and SQL
    """
    
    # Keywords that suggest SQL queries (structured data)
    SQL_KEYWORDS = {
        # Codes and mappings
        'código', 'codes', 'mapeo', 'mapping', 'icpc', 'icd', 'snomed',
        'clasificación', 'classification',
        
        # Quantitative queries  
        'cuántos', 'cuántas', 'how many', 'count', 'número', 'total',
        'lista', 'list', 'todos los', 'all the',
        
        # Database operations
        'buscar por código', 'search by code', 'tabla', 'table',
        'registro', 'record', 'base de datos', 'database',
        
        # Specific clinical coding terms
        'capítulo', 'chapter', 'componente', 'component',
        'criterio de inclusión', 'criterio de exclusión',
        'inclusion criteria', 'exclusion criteria'
    }
    
    # Keywords that suggest vectorstore queries (content/semantic)
    VECTORSTORE_KEYWORDS = {
        # Information seeking
        'qué es', 'what is', 'explica', 'explain', 'describe',
        'síntomas', 'symptoms', 'tratamiento', 'treatment',
        'diagnóstico', 'diagnosis', 'causas', 'causes',
        
        # Guidelines and recommendations
        'guía', 'guide', 'guideline', 'recomendación', 'recommendation',
        'protocolo', 'protocol', 'criterios diagnósticos',
        'diagnostic criteria', 'manejo', 'management',
        
        # Clinical content
        'fisiopatología', 'pathophysiology', 'epidemiología', 'epidemiology',
        'factores de riesgo', 'risk factors', 'pronóstico', 'prognosis',
        'complicaciones', 'complications', 'prevención', 'prevention'
    }
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the query router
        
        Args:
            use_llm: Whether to use LLM for sophisticated routing decisions
        """
        self.use_llm = use_llm
        
        if use_llm:
            try:
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                # do not change this unless explicitly requested by the user
                self.llm = ChatOpenAI(
                    model="gpt-5",
                    max_completion_tokens=256
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM for routing: {e}")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = None
    
    def _rule_based_routing(self, query: str) -> RoutingDecision:
        """
        Rule-based routing using keyword matching
        """
        query_lower = query.lower()
        
        # Count keyword matches
        sql_matches = sum(1 for keyword in self.SQL_KEYWORDS if keyword in query_lower)
        vectorstore_matches = sum(1 for keyword in self.VECTORSTORE_KEYWORDS if keyword in query_lower)
        
        # Specific patterns for SQL
        sql_patterns = [
            r'\bcódigo\s+[A-Z0-9]+\b',  # "código K86"
            r'\bicpc[-\s]*3\b',         # "ICPC-3" 
            r'\bicd[-\s]*11\b',         # "ICD-11"
            r'\bsnomed\b',              # "SNOMED"
            r'cuánt[oa]s?\s+\w+',       # "cuántos códigos"
            r'list[ar]?\s+\w+',         # "listar códigos"
            r'tod[oa]s?\s+l[oa]s?\s+\w+' # "todos los códigos"
        ]
        
        sql_pattern_matches = sum(1 for pattern in sql_patterns if re.search(pattern, query_lower))
        
        # Decision logic
        total_sql_score = sql_matches + (sql_pattern_matches * 2)  # Patterns have more weight
        total_vectorstore_score = vectorstore_matches
        
        if total_sql_score > total_vectorstore_score and total_sql_score > 0:
            confidence = min(0.9, 0.5 + (total_sql_score * 0.1))
            return RoutingDecision(
                query_type=QueryType.SQL,
                confidence=confidence,
                reasoning=f"SQL indicators: {total_sql_score}, Content indicators: {total_vectorstore_score}",
                suggested_modifications=["Consider using specific code formats (e.g., K86, A01)"]
            )
        
        elif total_vectorstore_score > 0:
            confidence = min(0.9, 0.5 + (total_vectorstore_score * 0.1))
            return RoutingDecision(
                query_type=QueryType.VECTORSTORE,
                confidence=confidence,
                reasoning=f"Content indicators: {total_vectorstore_score}, SQL indicators: {total_sql_score}",
                suggested_modifications=["Consider adding clinical context for better results"]
            )
        
        else:
            # Default to vectorstore with low confidence
            return RoutingDecision(
                query_type=QueryType.VECTORSTORE,
                confidence=0.3,
                reasoning="No clear indicators found, defaulting to content search",
                suggested_modifications=["Try being more specific about what information you need"]
            )
    
    def _llm_routing(self, query: str) -> RoutingDecision:
        """
        LLM-based sophisticated routing
        """
        if not self.llm:
            return self._rule_based_routing(query)
        
        routing_prompt = PromptTemplate.from_template("""
Eres un experto en sistemas de información clínica. Tu tarea es decidir cómo procesar una consulta médica.

Tienes dos opciones:
1. SQL: Para consultas sobre códigos, mapeos, clasificaciones, conteos, listas estructuradas
2. VECTORSTORE: Para contenido clínico, explicaciones, síntomas, tratamientos, guías

Ejemplos SQL:
- "¿Cuál es el código ICPC-3 para diabetes?"
- "Lista todos los códigos del capítulo cardiovascular"
- "¿Cuántos códigos hay en la base?"
- "Busca el mapeo de K86 a ICD-11"

Ejemplos VECTORSTORE:
- "¿Cuáles son los síntomas de la diabetes?"
- "Explica el manejo de la hipertensión"
- "¿Qué dicen las guías sobre depresión?"
- "Criterios diagnósticos para EPOC"

Consulta: "{query}"

Responde SOLO con:
TIPO: [SQL|VECTORSTORE]
CONFIANZA: [0.0-1.0]
RAZON: [Breve explicación]
""")
        
        try:
            response = self.llm.invoke(routing_prompt.format(query=query))
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            lines = response_text.strip().split('\n')
            
            query_type = QueryType.VECTORSTORE  # default
            confidence = 0.5  # default
            reasoning = "LLM routing response parsed"
            
            for line in lines:
                line = line.strip()
                if line.startswith('TIPO:'):
                    type_str = line.split(':', 1)[1].strip().upper()
                    if type_str == 'SQL':
                        query_type = QueryType.SQL
                    elif type_str == 'VECTORSTORE':
                        query_type = QueryType.VECTORSTORE
                
                elif line.startswith('CONFIANZA:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    except ValueError:
                        pass
                
                elif line.startswith('RAZON:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return RoutingDecision(
                query_type=query_type,
                confidence=confidence,
                reasoning=f"LLM decision: {reasoning}"
            )
        
        except Exception as e:
            print(f"Error in LLM routing: {e}")
            return self._rule_based_routing(query)
    
    def route_query(self, query: str) -> RoutingDecision:
        """
        Main routing function
        """
        if self.use_llm and self.llm:
            return self._llm_routing(query)
        else:
            return self._rule_based_routing(query)


# Convenience function for simple routing
def route_query(query: str, use_llm: bool = True) -> QueryType:
    """
    Simple function to get query type
    """
    router = QueryRouter(use_llm=use_llm)
    decision = router.route_query(query)
    return decision.query_type


class AdaptiveRouter:
    """
    Adaptive router that learns from user feedback
    """
    
    def __init__(self):
        self.router = QueryRouter()
        self.feedback_history = []
    
    def route_with_feedback(self, query: str) -> RoutingDecision:
        """
        Route query and prepare for feedback collection
        """
        decision = self.router.route_query(query)
        
        # Store for potential feedback
        self.feedback_history.append({
            'query': query,
            'decision': decision,
            'timestamp': None,  # Would be filled with actual timestamp
            'feedback': None    # To be filled when feedback is provided
        })
        
        return decision
    
    def provide_feedback(self, query: str, correct_type: QueryType, user_satisfaction: float):
        """
        Provide feedback on routing decision
        
        Args:
            query: The original query
            correct_type: What the correct routing should have been
            user_satisfaction: Score from 0-5
        """
        # Find the most recent decision for this query
        for entry in reversed(self.feedback_history):
            if entry['query'] == query and entry['feedback'] is None:
                entry['feedback'] = {
                    'correct_type': correct_type,
                    'satisfaction': user_satisfaction,
                    'was_correct': entry['decision'].query_type == correct_type
                }
                break
        
        # In a production system, this feedback would be used to retrain/adapt the router
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing performance
        """
        if not self.feedback_history:
            return {"message": "No feedback data available"}
        
        total_decisions = len(self.feedback_history)
        feedback_entries = [entry for entry in self.feedback_history if entry['feedback'] is not None]
        
        if not feedback_entries:
            return {"total_decisions": total_decisions, "feedback_received": 0}
        
        correct_decisions = sum(1 for entry in feedback_entries if entry['feedback']['was_correct'])
        avg_satisfaction = sum(entry['feedback']['satisfaction'] for entry in feedback_entries) / len(feedback_entries)
        
        type_distribution = {}
        for entry in feedback_entries:
            query_type = entry['decision'].query_type.value
            if query_type not in type_distribution:
                type_distribution[query_type] = {'count': 0, 'correct': 0}
            type_distribution[query_type]['count'] += 1
            if entry['feedback']['was_correct']:
                type_distribution[query_type]['correct'] += 1
        
        return {
            'total_decisions': total_decisions,
            'feedback_received': len(feedback_entries),
            'accuracy': correct_decisions / len(feedback_entries),
            'avg_satisfaction': avg_satisfaction,
            'type_distribution': type_distribution
        }


if __name__ == "__main__":
    # Test the routing system
    router = QueryRouter(use_llm=True)
    
    test_queries = [
        "¿Cuál es el código ICPC-3 para diabetes?",
        "¿Cuáles son los síntomas de la diabetes?", 
        "Lista todos los códigos del capítulo cardiovascular",
        "Explica el tratamiento de la hipertensión",
        "¿Cuántos códigos hay en total?",
        "¿Qué dice la guía sobre el manejo de EPOC?",
        "Busca el mapeo de K86 a ICD-11",
        "¿Cuáles son los criterios diagnósticos para depresión mayor?"
    ]
    
    print("Testing Query Routing:")
    print("=" * 50)
    
    for query in test_queries:
        decision = router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"Type: {decision.query_type.value}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        if decision.suggested_modifications:
            print(f"Suggestions: {decision.suggested_modifications}")
