from typing import Dict, Any, List
import asyncio
from .base_agent import BaseAgent
from services.openai_services import OpenAIService

class RecommendationAgent(BaseAgent):
    """Agent for generating final medical recommendations"""
    
    def __init__(self, agent_id: str = "recommendation_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.openai_service = OpenAIService()
        self.recommendation_categories = [
            'immediate_actions',
            'medical_evaluation',
            'symptom_monitoring',
            'lifestyle_modifications',
            'follow_up_care',
            'red_flags'
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive medical recommendations"""
        try:
            self.update_state("processing")
            
            # Extract coordinated results
            coordinator_results = input_data.get('coordinator_results', {})
            patient_profile = input_data.get('patient_profile', {})
            
            if not coordinator_results:
                raise ValueError("No coordinator results provided")
            
            # Generate primary recommendations
            primary_recommendations = await self._generate_primary_recommendations(coordinator_results)
            
            # Generate urgency-based actions
            urgency_actions = await self._generate_urgency_actions(coordinator_results)
            
            # Generate monitoring plan
            monitoring_plan = await self._generate_monitoring_plan(coordinator_results)
            
            # Generate follow-up recommendations
            followup_plan = await self._generate_followup_plan(coordinator_results, patient_profile)
            
            # Generate patient education
            patient_education = await self._generate_patient_education(coordinator_results)
            
            # Compile final recommendation report
            final_report = await self._compile_recommendation_report(
                primary_recommendations,
                urgency_actions,
                monitoring_plan,
                followup_plan,
                patient_education
            )
            
            result = {
                'agent_id': self.agent_id,
                'primary_recommendations': primary_recommendations,
                'urgency_actions': urgency_actions,
                'monitoring_plan': monitoring_plan,
                'followup_plan': followup_plan,
                'patient_education': patient_education,
                'final_report': final_report,
                'recommendation_confidence': self._calculate_recommendation_confidence(coordinator_results)
            }
            
            self.update_state("completed", result)
            return result
            
        except Exception as e:
            error_msg = f"Recommendation generation error: {str(e)}"
            self.update_state("error", error=error_msg)
            return {'error': error_msg, 'agent_id': self.agent_id}
    
    async def _generate_primary_recommendations(self, coordinator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate primary medical recommendations"""
        prompt = f"""
        Based on the comprehensive multimodal analysis, generate primary medical recommendations:
        
        Analysis Results: {coordinator_results}
        
        Generate recommendations for:
        1. Immediate care needs
        2. Medical evaluation requirements
        3. Symptom management strategies
        4. Preventive measures
        5. Lifestyle considerations
        
        Prioritize recommendations by urgency and evidence level.
        Return as structured JSON with clear action items.
        """
        
        response = await self.openai_service.get_completion(prompt, response_format="json_object")
        return response
    
    async def _generate_urgency_actions(self, coordinator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate urgency-based action items"""
        # Calculate overall urgency
        synthesis = coordinator_results.get('synthesis', {})
        avg_urgency = synthesis.get('confidence_levels', {})
        
        prompt = f"""
        Based on urgency indicators from multimodal analysis, generate time-sensitive actions:
        
        Urgency Analysis: {coordinator_results}
        
        Generate:
        1. Condition explanation in simple terms
        2. Self-care instructions
        3. What to expect during recovery
        4. When to contact healthcare providers
        5. Lifestyle modifications
        6. Resources for additional information
        
        Use patient-friendly language and clear instructions.
        Return as structured JSON.
        """
        
        response = await self.openai_service.get_completion(prompt, response_format="json_object")
        return response
    
    async def _compile_recommendation_report(self, primary_recs: Dict, urgency_actions: Dict, 
                                           monitoring: Dict, followup: Dict, education: Dict) -> Dict[str, Any]:
        """Compile comprehensive recommendation report"""
        prompt = f"""
        Compile a comprehensive medical recommendation report from:
        
        Primary Recommendations: {primary_recs}
        Urgency Actions: {urgency_actions}
        Monitoring Plan: {monitoring}
        Follow-up Plan: {followup}
        Patient Education: {education}
        
        Create a cohesive, prioritized report with:
        1. Executive summary
        2. Immediate action items
        3. Short and long-term plan
        4. Patient responsibilities
        5. Healthcare team coordination
        6. Quality metrics for tracking
        
        Return as well-structured JSON.
        """
        
        response = await self.openai_service.get_completion(prompt, response_format="json_object")
        return response
    
    def _calculate_recommendation_confidence(self, coordinator_results: Dict[str, Any]) -> float:
        """Calculate confidence in recommendations"""
        base_confidence = coordinator_results.get('overall_confidence', 0.5)
        
        # Adjust based on data quality
        agent_results = coordinator_results.get('agent_results', {})
        data_sources = len(agent_results)
        
        # Higher confidence with more data sources
        if data_sources >= 3:
            base_confidence += 0.2
        elif data_sources == 2:
            base_confidence += 0.1
        
        # Check for consistency across agents
        synthesis = coordinator_results.get('synthesis', {})
        if synthesis.get('conflicting_information'):
            base_confidence -= 0.1
        
        return min(base_confidence, 1.0)