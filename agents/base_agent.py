from abc import ABC, abstractmethod
from typing import Any,Dict, Optional
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """Base state model for all agents"""
    agent_id: str
    status: str = "initialized"
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    errors: list = []
    metadata: Dict[str, Any] = {}

class BaseAgent(ABC):
    """Abstract base Class for all Healthcare agents"""

    def __init__(self, agent_id:str, config:Dict[str, Any]=None):
        self.agent_id = agent_id
        self.config = config or {}
        self.state = AgentState(agent_id=agent_id)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    def update_state(self, status: str, output_data:Dict[str,Any]=None, error:str=None):
        """Update agent state"""
        self.state.status = status

        if output_data:
            self.state.output_data = output_data
        if error:
            self.state.errors.append(error)
        self.logger.info(f"Agent {self.agent_id} status updated to: {status}")

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.dict()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data format"""
        return isinstance(input_data, dict)