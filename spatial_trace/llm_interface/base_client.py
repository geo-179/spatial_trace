"""
Abstract base class for LLM clients.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        
    @abstractmethod
    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a chat completion.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Format specification for response
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM client is available and properly configured.
        
        Returns:
            True if client is ready to use, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "available": self.is_available()
        } 