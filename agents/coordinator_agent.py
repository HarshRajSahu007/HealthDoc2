from typing import Dict, Any, List
import asyncio
from .base_agent import BaseAgent
from .text_agent import TextAgent
from .voice_agent import VoiceAgent
from .image_agent import ImageAgent

class CoordinatorAgent(BaseAgent):
    """Main coordinator agent that orchestrates all other agents"""
    
    def __init__(self, agent_id: str = "coordinator_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.text_agent = TextAgent()
        self.voice_agent = VoiceAgent()
        self.image_agent = ImageAgent()
        self.agent_results = {}

    async def process(self, input_data: Dict[str,Any])-> Dict[str,Any]:
        """Coordinate processing across all agents"""
        try:
            self.update_state("processing")

            active_agents = self._determine_active_agents(input_data)

            agent_tasks = []
            for agent_type in active_agents:
                if agent_type == 'text' and input_data.get('text'):
                    agent_tasks.append(self._process_text_agent(input_data))
                elif agent_type == 'voice' and input_data.get('audio_file'):
                    agent_tasks.append(self._process_voice_agent(input_data))
                elif agent_type == 'image' and input_data.get('image_file'):
                    agent_tasks.append(self._process_image_agent(input_data))

            if agent_tasks:
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                self.agent_results = self._consolidate_agent_results(agent_results)


            synthesis = await self._synthesis_results()

            insights = await self._generate_insights()

            result = {
                'agent_id': self.agent_id,
                'agent_results': self.agent_results,
                'synthesis': synthesis,
                'insights': insights,
                'active_agents': active_agents,
                'overall_confidence': self._calculate_overall_confidence()
            }
            
            self.update_state("completed", result)
            return result
            
        except Exception as e:
            error_msg = f"Coordination error: {str(e)}"
            self.update_state("error", error=error_msg)
            return {'error': error_msg, 'agent_id': self.agent_id}
        
    def _determine_active_agents(self, input_data: Dict[str, Any]) -> List[str]:
        """Determine which agents should be activated based on input data"""
        active_agents = []
        
        if input_data.get('text'):
            active_agents.append('text')
        if input_data.get('audio_file'):
            active_agents.append('voice')
        if input_data.get('image_file'):
            active_agents.append('image')
        
        return active_agents
    
    async def _process_text_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text through text agent"""
        return await self.text_agent.process(input_data)
    
    async def _process_voice_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice through voice agent"""
        return await self.voice_agent.process(input_data)
    
    async def _process_image_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image through image agent"""
        return await self.image_agent.process(input_data)
    
    def _consolidate_agent_results(self, agent_results: List[Any]) -> Dict[str, Any]:
        """Consolidate results from all agents"""
        consolidated = {}
        
        for result in agent_results:
            if isinstance(result, Exception):
                self.logger.error(f"Agent processing error: {result}")
                continue
            
            if isinstance(result, dict) and 'agent_id' in result:
                agent_id = result['agent_id']
                consolidated[agent_id] = result
        
        return consolidated
    
    async def _synthesize_results(self) -> Dict[str, Any]:
        """Synthesize results across all agents"""
        synthesis = {
            'common_themes': [],
            'conflicting_information': [],
            'confidence_levels': {},
            'urgent_findings': [],
            'comprehensive_picture': {}
        }
        
     
        all_symptoms = []
        all_concerns = []
        
        for agent_id, result in self.agent_results.items():
            if 'symptoms_analysis' in result:
                symptoms = result['symptoms_analysis'].get('primary_symptoms', [])
                all_symptoms.extend(symptoms)
            
            if 'patient_concerns' in result:
                all_concerns.extend(result['patient_concerns'])
            
            
            confidence = result.get('confidence_score', 0)
            synthesis['confidence_levels'][agent_id] = confidence
        
      
        from collections import Counter
        if all_symptoms:
            common_symptoms = [item for item, count in Counter(all_symptoms).most_common(5)]
            synthesis['common_themes'] = common_symptoms
        
        return synthesis
    
    async def _generate_insights(self) -> Dict[str, Any]:
        """Generate insights from coordinated analysis"""
        insights = {
            'multimodal_correlation': {},
            'consistency_check': {},
            'additional_questions': [],
            'care_coordination': {}
        }
        
 
        text_findings = self.agent_results.get('text_agent', {}).get('symptoms_analysis', {})
        voice_findings = self.agent_results.get('voice_agent', {}).get('emotional_analysis', {})
        image_findings = self.agent_results.get('image_agent', {}).get('urgency_assessment', {})
        

        urgency_levels = []
        if text_findings.get('urgency_level'):
            urgency_levels.append(text_findings['urgency_level'])
        if voice_findings.get('anxiety_level'):
            urgency_levels.append(voice_findings['anxiety_level'])
        if image_findings.get('urgency_level'):
            urgency_levels.append(image_findings['urgency_level'])
        
        if urgency_levels:
            avg_urgency = sum(urgency_levels) / len(urgency_levels)
            insights['multimodal_correlation']['average_urgency'] = avg_urgency
        
        return insights
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score across all agents"""
        confidences = []
        
        for result in self.agent_results.values():
            if 'confidence_score' in result:
                confidences.append(result['confidence_score'])
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
