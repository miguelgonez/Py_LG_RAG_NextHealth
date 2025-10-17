"""
Políticas y guardarraíles para RAG NextHealth
Implementa restricciones clínicas y formateo de respuestas
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class ResponsePolicy:
    """
    Política de respuesta con guardarraíles clínicos
    """
    min_docs: int = 2
    max_response_length: int = 2000
    require_sources: bool = True
    include_disclaimer: bool = True
    
    # Restricciones clínicas
    forbidden_medical_advice: bool = True
    forbidden_dosage_info: bool = True
    forbidden_diagnosis: bool = True
    
    # Disclaimer obligatorio
    disclaimer: str = (
        "⚠️ IMPORTANTE: Esta información es únicamente educativa y no sustituye "
        "la consulta médica profesional. No se debe usar para autodiagnóstico "
        "o automedicación. Cumple con AI Act/MDR: sin diagnóstico ni prescripción. "
        "Consulte siempre con un profesional sanitario cualificado."
    )
    
    def get_instructions(self) -> str:
        """
        Genera las instrucciones de política para el prompt
        """
        instructions = [
            "GUARDARRAÍLES OBLIGATORIOS:",
            "- NO proporciones diagnósticos médicos específicos",
            "- NO recomiende dosis de medicamentos",
            "- NO sustituyas la consulta médica profesional",
            "- Usa SOLO información de fuentes proporcionadas",
            "- Incluye referencias específicas a las fuentes",
            "- Mantén respuestas claras pero sin exceder 2000 caracteres",
            "- Añade disclaimer médico al final"
        ]
        
        return "\n".join(instructions)
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Valida que la respuesta cumple con las políticas
        """
        issues = []
        warnings = []
        
        # Verificar palabras prohibidas para diagnóstico
        diagnostic_words = [
            'diagnostico', 'diagnóstico', 'padeces', 'tienes', 
            'sufres de', 'tu enfermedad', 'tu condición'
        ]
        
        response_lower = response.lower()
        for word in diagnostic_words:
            if word in response_lower:
                issues.append(f"Posible diagnóstico detectado: '{word}'")
        
        # Verificar información de dosis
        dosage_patterns = [
            'mg', 'gramos', 'tomar', 'dosis', 'pastillas',
            'comprimidos', 'cápsulas', 'ml', 'mililitros'
        ]
        
        for pattern in dosage_patterns:
            if pattern in response_lower:
                warnings.append(f"Posible información de dosis: '{pattern}'")
        
        # Verificar longitud
        if len(response) > self.max_response_length:
            issues.append(f"Respuesta demasiado larga: {len(response)} caracteres")
        
        # Verificar disclaimer
        if self.include_disclaimer and self.disclaimer not in response:
            issues.append("Falta disclaimer médico")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "length": len(response)
        }


def format_response_with_policy(
    response: str, 
    documents: List, 
    route_used: str,
    policy: ResponsePolicy = None
) -> str:
    """
    Formatea la respuesta aplicando las políticas de salida
    """
    if policy is None:
        policy = DEFAULT_POLICY
    
    # Preparar información de fuentes
    sources_info = []
    if documents:
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Fuente desconocida')
            page = doc.metadata.get('page', '')
            route = doc.metadata.get('route', 'unknown')
            
            source_line = f"[{i+1}] {source}"
            if page:
                source_line += f" (p. {page})"
            if route == 'sql':
                source_line += " [Base de datos]"
            
            sources_info.append(source_line)
    
    # Construir respuesta formateada
    formatted_parts = []
    
    # Respuesta principal
    formatted_parts.append("## 📋 Respuesta")
    formatted_parts.append(response)
    
    # Información de fuentes
    if sources_info and policy.require_sources:
        formatted_parts.append("\n## 📚 Fuentes consultadas")
        for source in sources_info:
            formatted_parts.append(source)
    
    # Metadata técnica
    formatted_parts.append(f"\n## 🔍 Información técnica")
    formatted_parts.append(f"- Método de búsqueda: {route_used.title()}")
    formatted_parts.append(f"- Documentos analizados: {len(documents)}")
    formatted_parts.append(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Disclaimer obligatorio
    if policy.include_disclaimer:
        formatted_parts.append(f"\n## ⚠️ Aviso médico")
        formatted_parts.append(policy.disclaimer)
    
    final_response = "\n".join(formatted_parts)
    
    # Validar la respuesta
    validation = policy.validate_response(final_response)
    if not validation["valid"]:
        print(f"⚠️ Respuesta no válida según políticas: {validation['issues']}")
        
        # Añadir avisos adicionales si hay problemas
        if validation["issues"]:
            final_response += f"\n\n🔧 AVISOS DE VALIDACIÓN:\n" + "\n".join([f"- {issue}" for issue in validation["issues"]])
    
    return final_response


def check_query_safety(query: str) -> Dict[str, Any]:
    """
    Verifica que la consulta del usuario sea apropiada
    """
    unsafe_patterns = [
        'dame el diagnóstico', 'qué tengo', 'que tengo',
        'cómo me curo', 'como me curo', 'qué medicina tomar',
        'que medicina tomar', 'dosis recomendada', 'cuánto tomar'
    ]
    
    query_lower = query.lower()
    detected_patterns = [pattern for pattern in unsafe_patterns if pattern in query_lower]
    
    return {
        "safe": len(detected_patterns) == 0,
        "detected_patterns": detected_patterns,
        "recommendation": (
            "Reformula tu pregunta solicitando información general en lugar de consejo médico específico."
            if detected_patterns else "Consulta apropiada."
        )
    }


def create_custom_policy(
    min_docs: int = 2,
    include_disclaimer: bool = True,
    max_length: int = 2000,
    custom_disclaimer: str = None
) -> ResponsePolicy:
    """
    Crea una política personalizada
    """
    policy = ResponsePolicy(
        min_docs=min_docs,
        max_response_length=max_length,
        include_disclaimer=include_disclaimer
    )
    
    if custom_disclaimer:
        policy.disclaimer = custom_disclaimer
    
    return policy


# Política por defecto
DEFAULT_POLICY = ResponsePolicy(
    min_docs=2,
    max_response_length=2000,
    require_sources=True,
    include_disclaimer=True,
    forbidden_medical_advice=True,
    forbidden_dosage_info=True,
    forbidden_diagnosis=True
)

# Política más estricta para uso clínico
CLINICAL_POLICY = ResponsePolicy(
    min_docs=3,
    max_response_length=1500,
    require_sources=True,
    include_disclaimer=True,
    forbidden_medical_advice=True,
    forbidden_dosage_info=True,
    forbidden_diagnosis=True,
    disclaimer=(
        "⚠️ AVISO CLÍNICO ESTRICTO: Esta herramienta es exclusivamente para "
        "consulta de información médica general por profesionales sanitarios. "
        "No genera diagnósticos, tratamientos ni recomendaciones terapéuticas. "
        "Cumple con regulación MDR (EU) 2017/745 y AI Act. "
        "El profesional sanitario es responsable de toda decisión clínica."
    )
)

# Política para educación médica
EDUCATIONAL_POLICY = ResponsePolicy(
    min_docs=1,
    max_response_length=2500,
    require_sources=True,
    include_disclaimer=True,
    forbidden_medical_advice=False,  # Permite más flexibilidad educativa
    forbidden_dosage_info=True,
    forbidden_diagnosis=True,
    disclaimer=(
        "📚 PROPÓSITO EDUCATIVO: Esta información se proporciona con fines "
        "educativos para estudiantes y profesionales de la salud. "
        "No sustituye la formación clínica formal ni la experiencia práctica. "
        "Siempre contraste con fuentes oficiales y supervisión académica."
    )
)
