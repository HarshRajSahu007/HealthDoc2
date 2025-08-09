from typing import Dict, Any, List
import asyncio
import logging
from agents.text_agent import TextAgent
from agents.voice_agent import VoiceAgent
from agents.image_agent import ImageAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.recommendation_agent import RecommendationAgent

logger = logging.getLogger(__name__)

class GraphNodes:
    """Node implementations for the healthcare workflow graph"""
    
    def __init__(self):
        self.text_agent = TextAgent()
        self.voice_agent = VoiceAgent()
        self.image_agent = ImageAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.recommendation_agent = RecommendationAgent()
    
    async def validate_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and determine processing path"""
        try:
            logger.info("Starting input validation")
            
            # Check for required inputs
            has_text = bool(state.get('text_input'))
            has_audio = bool(state.get('audio_file'))
            has_image = bool(state.get('image_file'))
            
            if not any([has_text, has_audio, has_image]):
                state['errors'].append("No valid input data provided")
                state['processing_status'] = 'error'
                return state
            
            # Update state with validation results
            state['processing_metadata'] = {
                'has_text': has_text,
                'has_audio': has_audio,
                'has_image': has_image,
                'input_types': [t for t, has in [('text', has_text), ('audio', has_audio), ('image', has_image)] if has]
            }
            
            state['current_step'] = 'input_validated'
            state['processing_status'] = 'validated'
            
            logger.info(f"Input validation completed. Input types: {state['processing_metadata']['input_types']}")
            return state
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            state['errors'].append(f"Validation error: {str(e)}")
            state['processing_status'] = 'error'
            return state
    
    async def text_processing_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input through text agent"""
        try:
            if not state.get('text_input'):
                logger.info("No text input to process")
                return state
            
            logger.info("Starting text processing")
            state['current_step'] = 'text_processing'
            
            # Prepare input for text agent
            text_input = {
                'text': state['text_input']
            }
            
            # Process through text agent
            result = await self.text_agent.process(text_input)
            
            if 'error' in result:
                state['errors'].append(f"Text processing error: {result['error']}")
            else:
                state['text_analysis'] = result
                state['confidence_scores']['text'] = result.get('confidence_score', 0.0)
            
            logger.info("Text processing completed")
            return state
            
        except Exception as e:
            logger.error(f"Text processing node error: {e}")
            state['errors'].append(f"Text processing error: {str(e)}")
            return state
    
    async def voice_processing_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice input through voice agent"""
        try:
            if not state.get('audio_file'):
                logger.info("No audio input to process")
                return state
            
            logger.info("Starting voice processing")
            state['current_step'] = 'voice_processing'
            
            # Prepare input for voice agent
            voice_input = {
                'audio_file': state['audio_file'],
                'format': 'wav'  # Default format
            }
            
            # Process through voice agent
            result = await self.voice_agent.process(voice_input)
            
            if 'error' in result:
                state['errors'].append(f"Voice processing error: {result['error']}")
            else:
                state['voice_analysis'] = result
                state['confidence_scores']['voice'] = result.get('confidence_score', 0.0)
            
            logger.info("Voice processing completed")
            return state
            
        except Exception as e:
            logger.error(f"Voice processing node error: {e}")
            state['errors'].append(f"Voice processing error: {str(e)}")
            return state
    
    async def image_processing_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process image input through image agent"""
        try:
            if not state.get('image_file'):
                logger.info("No image input to process")
                return state
            
            logger.info("Starting image processing")
            state['current_step'] = 'image_processing'
            
            # Prepare input for image agent
            image_input = {
                'image_file': state['image_file'],
                'image_type': 'general',  # Could be determined from metadata
                'format': 'jpg'  # Default format
            }
            
            # Process through image agent
            result = await self.image_agent.process(image_input)
            
            if 'error' in result:
                state['errors'].append(f"Image processing error: {result['error']}")
            else:
                state['image_analysis'] = result
                state['confidence_scores']['image'] = result.get('confidence_score', 0.0)
            
            logger.info("Image processing completed")
            return state
            
        except Exception as e:
            logger.error(f"Image processing node error: {e}")
            state['errors'].append(f"Image processing error: {str(e)}")
            return state
    
    async def coordination_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate and synthesize results from all agents"""
        try:
            logger.info("Starting coordination and synthesis")
            state['current_step'] = 'coordination'
            
            # Prepare input for coordinator
            coordinator_input = {
                'text': state.get('text_input'),
                'audio_file': state.get('audio_file'),
                'image_file': state.get('image_file')
            }
            
            # Process through coordinator agent
            result = await self.coordinator_agent.process(coordinator_input)
            
            if 'error' in result:
                state['errors'].append(f"Coordination error: {result['error']}")
                state['processing_status'] = 'error'
            else:
                state['coordination_results'] = result
                state['confidence_scores']['coordination'] = result.get('overall_confidence', 0.0)
                state['processing_status'] = 'coordinated'
            
            logger.info("Coordination completed")
            return state
            
        except Exception as e:
            logger.error(f"Coordination node error: {e}")
            state['errors'].append(f"Coordination error: {str(e)}")
            state['processing_status'] = 'error'
            return state
    
    async def recommendation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on coordinated analysis"""
        try:
            logger.info("Starting recommendation generation")
            state['current_step'] = 'recommendation_generation'
            
            # Prepare input for recommendation agent
            recommendation_input = {
                'coordinator_results': state.get('coordination_results', {}),
                'patient_profile': state.get('patient_profile', {})
            }
            
            # Process through recommendation agent
            result = await self.recommendation_agent.process(recommendation_input)
            
            if 'error' in result:
                state['errors'].append(f"Recommendation error: {result['error']}")
                state['processing_status'] = 'error'
            else:
                state['recommendations'] = result
                state['confidence_scores']['recommendations'] = result.get('recommendation_confidence', 0.0)
                state['processing_status'] = 'recommendations_generated'
            
            logger.info("Recommendation generation completed")
            return state
            
        except Exception as e:
            logger.error(f"Recommendation node error: {e}")
            state['errors'].append(f"Recommendation error: {str(e)}")
            state['processing_status'] = 'error'
            return state
    
    async def report_compilation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final comprehensive report"""
        try:
            logger.info("Starting report compilation")
            state['current_step'] = 'report_compilation'
            
            # Compile comprehensive report
            final_report = {
                'executive_summary': self._generate_executive_summary(state),
                'analysis_results': {
                    'text_analysis': state.get('text_analysis', {}),
                    'voice_analysis': state.get('voice_analysis', {}),
                    'image_analysis': state.get('image_analysis', {}),
                    'coordination_synthesis': state.get('coordination_results', {})
                },
                'recommendations': state.get('recommendations', {}),
                'confidence_assessment': self._assess_overall_confidence(state),
                'next_steps': self._extract_next_steps(state),
                'quality_metrics': self._calculate_quality_metrics(state)
            }
            
            state['final_report'] = final_report
            state['processing_status'] = 'completed'
            state['current_step'] = 'completed'
            
            logger.info("Report compilation completed")
            return state
            
        except Exception as e:
            logger.error(f"Report compilation error: {e}")
            state['errors'].append(f"Report compilation error: {str(e)}")
            state['processing_status'] = 'error'
            return state
    
    async def error_handling_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors and provide fallback responses"""
        try:
            logger.warning("Entering error handling node")
            
            error_summary = {
                'total_errors': len(state.get('errors', [])),
                'error_details': state.get('errors', []),
                'failed_at_step': state.get('current_step', 'unknown'),
                'partial_results_available': bool(
                    state.get('text_analysis') or 
                    state.get('voice_analysis') or 
                    state.get('image_analysis')
                )
            }
            
            # Generate fallback recommendations if possible
            fallback_report = {
                'status': 'partial_completion',
                'error_summary': error_summary,
                'available_results': {},
                'general_recommendations': {
                    'immediate_action': 'Consult with healthcare provider for proper evaluation',
                    'note': 'Complete analysis could not be performed due to processing errors'
                }
            }
            
            # Include any successful analysis results
            if state.get('text_analysis'):
                fallback_report['available_results']['text_analysis'] = state['text_analysis']
            if state.get('voice_analysis'):
                fallback_report['available_results']['voice_analysis'] = state['voice_analysis']
            if state.get('image_analysis'):
                fallback_report['available_results']['image_analysis'] = state['image_analysis']
            
            state['final_report'] = fallback_report
            state['processing_status'] = 'error_handled'
            
            logger.info("Error handling completed")
            return state
            
        except Exception as e:
            logger.error(f"Error handling node error: {e}")
            state['final_report'] = {
                'status': 'critical_error',
                'message': 'System encountered critical errors and could not process request'
            }
            return state
    
    def _generate_executive_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis"""
        summary = {
            'input_modalities': len([x for x in ['text_analysis', 'voice_analysis', 'image_analysis'] 
                                   if state.get(x)]),
            'overall_confidence': sum(state.get('confidence_scores', {}).values()) / max(len(state.get('confidence_scores', {})), 1),
            'key_findings': [],
            'urgency_level': 'moderate'  # Default
        }
        
        # Extract key findings from each modality
        if state.get('text_analysis', {}).get('symptoms_analysis'):
            summary['key_findings'].extend(
                state['text_analysis']['symptoms_analysis'].get('primary_symptoms', [])
            )
        
        if state.get('coordination_results', {}).get('synthesis'):
            urgency_levels = []
            synthesis = state['coordination_results']['synthesis']
            if synthesis.get('confidence_levels'):
                avg_urgency = sum(synthesis['confidence_levels'].values()) / len(synthesis['confidence_levels'])
                if avg_urgency > 0.8:
                    summary['urgency_level'] = 'high'
                elif avg_urgency < 0.4:
                    summary['urgency_level'] = 'low'
        
        return summary
    
    def _assess_overall_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall confidence in analysis"""
        confidence_scores = state.get('confidence_scores', {})
        
        if not confidence_scores:
            return {'overall': 0.0, 'assessment': 'insufficient_data'}
        
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        
        assessment = 'low'
        if overall_confidence > 0.8:
            assessment = 'high'
        elif overall_confidence > 0.6:
            assessment = 'moderate'
        
        return {
            'overall': overall_confidence,
            'assessment': assessment,
            'by_modality': confidence_scores
        }
    
    def _extract_next_steps(self, state: Dict[str, Any]) -> List[str]:
        """Extract next steps from recommendations"""
        next_steps = []
        
        recommendations = state.get('recommendations', {})
        if recommendations.get('urgency_actions'):
            immediate_actions = recommendations['urgency_actions'].get('immediate_actions', [])
            next_steps.extend(immediate_actions)
        
        if not next_steps:
            next_steps = ['Consult healthcare provider for further evaluation']
        
        return next_steps
    
    def _calculate_quality_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the analysis"""
        return {
            'data_completeness': sum([
                1 for modality in ['text_analysis', 'voice_analysis', 'image_analysis'] 
                if state.get(modality)
            ]) / 3,
            'processing_errors': len(state.get('errors', [])),
            'analysis_depth': len(state.get('coordination_results', {}).get('agent_results', {})),
            'recommendation_coverage': len(state.get('recommendations', {}).get('primary_recommendations', {}))
        }
