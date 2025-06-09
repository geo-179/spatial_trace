"""
OpenAI client implementation for LLM communication.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI
from .base_client import BaseLLMClient

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip auto-loading

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI client for LLM communication."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        """
        super().__init__(model_name)
        
        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Format specification for response
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If client is not available
            Exception: If API call fails
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client is not available")
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens is not None:
                api_params["max_tokens"] = max_tokens
            
            if response_format is not None:
                api_params["response_format"] = response_format
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response text
            response_text = response.choices[0].message.content
            logger.debug(f"OpenAI API call successful. Response length: {len(response_text) if response_text else 0}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise e
    
    def is_available(self) -> bool:
        """
        Check if OpenAI client is available.
        
        Returns:
            True if client is ready to use, False otherwise
        """
        return self.client is not None and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            "provider": "OpenAI",
            "api_key_configured": self.api_key is not None
        })
        return info 