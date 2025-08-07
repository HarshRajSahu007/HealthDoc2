from typing import Dict,Any, List
import base64
import asyncio
from .base_agent import BaseAgent
from services.openai_services import OpenAIService
from services.vision_services import VisionService

class ImageAgent(BaseAgent):
    """Agent for processing and anaalyzing medical images"""

    def __init__(self, agent_id: str="image_agent", config:Dict[str,Any] = None):
        super().__init__(agent_id, config)
        self.openai_service = OpenAIService()
        self.vision_service = VisionService()
        self.supported_formats = ['jpg', 'jpeg','png','webp']

    async def process(self, input_data: Dict[str,Any])-> Dict[str, Any]:
        """Process image input for medical analysis"""
        try:
            self.update_state("processing")

            image_data = input_data.get('image_file')
            image_type = input_data.get('image_type', ' general')
            image_format = input_data.get('format','jpg')
            if not image_data:
                raise ValueError("No image data provided")
            

            image_analysis = await self._analyze_medical_image(image_data, image_type)
            

            symptom_detection = await self._detect_visual_symptoms(image_data, image_type)
            

            observations = await self._generate_medical_observations(image_data, image_type)
            

            urgency_assessment = await self._assess_visual_urgency(image_data, image_type)
            
            result = {
                'agent_id': self.agent_id,
                'image_analysis': image_analysis,
                'symptom_detection': symptom_detection,
                'medical_observations': observations,
                'urgency_assessment': urgency_assessment,
                'image_type': image_type,
                'confidence_score': self._calculate_confidence(image_analysis, symptom_detection)
            }
            
            self.update_state("completed", result)
            return result
        except Exception as e:
            error_msg = f"Image processing error: {str(e)}"
            self.update_state("error", error=error_msg)
            return {'error': error_msg, 'agent_id': self.agent_id}
        
    async def _analyze_medical_image(self, image_data:bytes, image_type: str) -> Dict[str,Any]:
        """Analyze medical image using OpenAI Vision API"""

        base64_image = base64.b64encode(image_data).decode('utf-8')

        prompt = f"""
        Analyze this medical image of type '{image_type}' and provide:
        
        1. Overall description of what you observe
        2. Any abnormalities or concerning features
        3. Color, texture, size observations
        4. Location and distribution of any findings
        5. Comparative assessment (normal vs abnormal)
        
        Important: This is for informational purposes only and not a medical diagnosis.
        Return as structured JSON.
        """
        
        response = await self.vision_service.analyze_image(base64_image, prompt)
        return response
    

    async def _detect_visual_symptoms(self, image_data: bytes, image_type: str) -> Dict[str,Any]:
        """Detect specefic visual symtoms in the image"""
        base64_image = base64.b64encode(image_data).decode('utf-8')

        prompt = f"""
        Examine this {image_type} image for specific visual symptoms:
        
        Look for:
        - Redness, swelling, discoloration
        - Lesions, rashes, or skin changes
        - Asymmetry or unusual formations
        - Signs of inflammation or infection
        - Size, shape, and border characteristics
        
        Rate the severity of any findings on a scale of 1-10.
        Return detailed findings as JSON.
        """
        
        response = await self.vision_service.analyze_image(base64_image, prompt)
        return response
    
    async def _assess_visual_urgency(self, image_data: bytes, image_type: str) -> Dict[str, Any]:
        """Assess urgency level from visual appearance"""
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        prompt = f"""
        Assess the urgency level of this {image_type} image:
        
        Consider:
        - Signs of acute inflammation
        - Potential infection indicators
        - Rapid progression signs
        - Size and extent of findings
        - Color changes indicating severity
        
        Provide:
        1. Urgency level (1-10, where 10 = immediate medical attention)
        2. Red flag indicators present
        3. Timeline for medical evaluation
        4. Specific concerns requiring prompt attention
        
        Return as JSON.
        """
        
        response = await self.vision_service.analyze_image(base64_image, prompt)
        return response
    
    def _calculate_confidence(self, image_analysis: Dict, symptom_detection: Dict) -> float:
        """Calculate confidence score for image analysis"""
        score = 0.5
        
        if image_analysis.get('findings'):
            score += 0.2
        if symptom_detection.get('symptoms_detected'):
            score += 0.2
        if image_analysis.get('quality_score', 0) > 0.7:
            score += 0.1
        
        return min(score, 1.0)