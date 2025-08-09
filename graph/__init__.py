from .healthcare_graph import HealthcareGraph
from .nodes import GraphNodes
from .edges import GraphEdges

__all__ = ['HealthcareGraph','GraphNodes','GraphEdges']

from typing import Dict,Any,List
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated,TypedDict
from agents.coordinator_agent import CoordinatorAgent
from agents.recommendation_agent import RecommendationAgent

class HealthcareState(TypedDict):
    """State structure for the healthcare workflow"""

    text_input: str
    audio_file: bytes
    image_file: bytes
    patient_profile: Dict[str, Any]
    

    current_step: str
    processing_status: str
    errors: List[str]
    

    text_analysis: Dict[str, Any]
    voice_analysis: Dict[str, Any]
    image_analysis: Dict[str, Any]
    coordination_results: Dict[str, Any]
    recommendations: Dict[str, Any]
    

    final_report: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_metadata: Dict[str, Any]

class HealthcareGraph:
    """Main LangGraph workflow for healthcare multi-agent system"""

    def __init__(self,config: Dict[str, Any]=None):
        self.config = config or {}
        self.coordinator_agent = CoordinatorAgent()
        self.recommendation_agent = RecommendationAgent()
        self.nodes = GraphNodes()
        self.edges = GraphEdges()
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(HealthcareState)
        
 
        workflow.add_node("input_validation", self.nodes.validate_input_node)
        workflow.add_node("text_processing", self.nodes.text_processing_node)
        workflow.add_node("voice_processing", self.nodes.voice_processing_node)
        workflow.add_node("image_processing", self.nodes.image_processing_node)
        workflow.add_node("coordination", self.nodes.coordination_node)
        workflow.add_node("recommendation_generation", self.nodes.recommendation_node)
        workflow.add_node("report_compilation", self.nodes.report_compilation_node)
        workflow.add_node("error_handling", self.nodes.error_handling_node)

        workflow.set_entry_point("input_validation")
        

        workflow.add_conditional_edges(
            "input_validation",
            self.edges.route_after_validation,
            {
                "process_multimodal": "text_processing",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "text_processing",
            self.edges.route_after_text_processing,
            {
                "voice_processing": "voice_processing",
                "image_processing": "image_processing",
                "coordination": "coordination",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "voice_processing",
            self.edges.route_after_voice_processing,
            {
                "image_processing": "image_processing",
                "coordination": "coordination",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "image_processing",
            self.edges.route_after_image_processing,
            {
                "coordination": "coordination",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "coordination",
            self.edges.route_after_coordination,
            {
                "recommendation_generation": "recommendation_generation",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "recommendation_generation",
            self.edges.route_after_recommendations,
            {
                "report_compilation": "report_compilation",
                "error": "error_handling"
            }
        )
        
        workflow.add_edge("report_compilation", END)
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    async def process_healthcare_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a healthcare request through the workflow"""
        # Initialize state
        initial_state = HealthcareState(
            text_input=input_data.get('text', ''),
            audio_file=input_data.get('audio_file'),
            image_file=input_data.get('image_file'),
            patient_profile=input_data.get('patient_profile', {}),
            current_step='initialized',
            processing_status='started',
            errors=[],
            text_analysis={},
            voice_analysis={},
            image_analysis={},
            coordination_results={},
            recommendations={},
            final_report={},
            confidence_scores={},
            processing_metadata={}
        )
        
        # Execute workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            return self._format_output(final_state)
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'processing_metadata': {
                    'error_occurred_at': 'workflow_execution'
                }
            }
    
    def _format_output(self, state: HealthcareState) -> Dict[str, Any]:
        """Format the final output"""
        return {
            'status': state.get('processing_status', 'unknown'),
            'final_report': state.get('final_report', {}),
            'confidence_scores': state.get('confidence_scores', {}),
            'processing_metadata': state.get('processing_metadata', {}),
            'errors': state.get('errors', []),
            'agent_results': {
                'text_analysis': state.get('text_analysis', {}),
                'voice_analysis': state.get('voice_analysis', {}),
                'image_analysis': state.get('image_analysis', {}),
                'coordination_results': state.get('coordination_results', {}),
                'recommendations': state.get('recommendations', {})
            }
        }
    
    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow"""
        return """
        Healthcare Multi-Agent Workflow:
        
        1. Input Validation
           ↓
        2. Parallel Processing:
           ├── Text Processing
           ├── Voice Processing  
           └── Image Processing
           ↓
        3. Coordination & Synthesis
           ↓
        4. Recommendation Generation
           ↓
        5. Report Compilation
           ↓
        6. Final Output
        
        Error handling available at each step.
        """