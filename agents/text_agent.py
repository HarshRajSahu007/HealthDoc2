from typing import Dict, Any, List
import asyncio
from .base_agent import BaseAgent
from services.openai_services import OpenAIService

class TextAgent(BaseAgent):
    """Agent for processing and analyzing text-based medical data"""

    def __init__(self, agent_id: str = "text_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.openai_service = OpenAIService()
        self.medical_keywords = [
            'symptoms', 'diagnosis', 'treatment', 'medication', 'pain', 
            'fever', 'headache', 'nausea', 'fatigue', 'breathing', 
            'chest pain', 'allergies', 'medical history'
        ]
    async def process(self, input_data:Dict[str,Any])->Dict[str,Any]:
        """Process text input for medical Analysis"""
        try:
            self.update_state('processing')

            text_content= input_data.get('text','')
            if not text_content:
                raise ValueError("No text content provided")
    
            medical_info = await self._extract_medical_info(text_content)
            

            symptoms_analysis = await self._analyze_symptoms(text_content)
            

            concerns = await self._extract_concerns(text_content)

            result = {
                'agent_id': self.agent_id,
                'medical_info': medical_info,
                'symptoms_analysis': symptoms_analysis,
                'patient_concerns': concerns,
                'original_text': text_content,
                'confidence_score': self._calculate_confidence(medical_info, symptoms_analysis)
            }
            self.update_state("completed",result)
            return result
        
        except Exception as e:
            error_msg = f"Text processing error: {str(e)}"
            self.update_state("error", error=error_msg)
            return {'error': error_msg, 'agent_id': self.agent_id}
        
    async def _extract_medical_info(self, text: str) -> Dict[str, Any]:
        """Extract structured medical information from text"""
        prompt = f"""
        Extract medical information from the following text and structure it as JSON:
        
        Text: {text}
        
        Please extract:
        1. Symptoms mentioned
        2. Medical history
        3. Current medications
        4. Allergies
        5. Duration of symptoms
        6. Severity indicators
        
        Return as structured JSON.
        """
        
        response = await self.openai_service.get_completion(prompt, response_format="json_object")
        return response
    
    async def _analyze_symptoms(self, text: str) -> Dict[str, Any]:
        """Analyze symptoms severity and patterns"""
        prompt = f"""
        Analyze the symptoms described in this text for severity and urgency:
        
        Text: {text}
        
        Provide:
        1. Urgency level (1-10)
        2. Primary symptoms
        3. Secondary symptoms
        4. Red flag indicators
        5. Recommended next steps
        
        Return as JSON.
        """
        
        response = await self.openai_service.get_completion(prompt, response_format="json_object")
        return response
    
    async def _extract_concerns(self, text: str) -> List[str]:
        """Extract patient concerns and questions"""
        prompt = f"""
        Extract patient concerns, questions, and worries from this text:
        
        Text: {text}
        
        Return a list of specific concerns or questions the patient has expressed.
        """
        
        response = await self.openai_service.get_completion(prompt)
        return response.get('concerns', [])
    
    def _calculate_confidence(self, medical_info: Dict, symptoms_analysis: Dict) -> float:
        """Calculate confidence score based on extracted information"""
        score = 0.5 
        
        if medical_info.get('symptoms'):
            score += 0.2
        if medical_info.get('medical_history'):
            score += 0.1
        if symptoms_analysis.get('urgency_level'):
            score += 0.2
        
        return min(score, 1.0)
    
    