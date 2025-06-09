"""
Output parser for LLM responses.
"""
import json
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Enum for different action types in LLM responses."""
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    FINAL_ANSWER = "final_answer"

class OutputParser:
    """Parser for LLM JSON responses."""
    
    @staticmethod
    def parse_llm_response(response_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse LLM response and validate structure.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Tuple of (success, parsed_data, error_message)
            - success: True if parsing successful
            - parsed_data: Parsed JSON data or None if failed
            - error_message: Error description or None if successful
        """
        if not response_text or not response_text.strip():
            return False, None, "Empty response from LLM"
        
        try:
            # Parse JSON
            data = json.loads(response_text.strip())
            
            # Validate structure
            is_valid, error = OutputParser._validate_response_structure(data)
            if not is_valid:
                return False, None, error
            
            return True, data, None
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in LLM response: {e}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error parsing LLM response: {e}"
            logger.error(error_msg)
            return False, None, error_msg
    
    @staticmethod
    def _validate_response_structure(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate the structure of parsed response data.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if 'action' field exists
        if "action" not in data:
            return False, "Missing 'action' field in response"
        
        action = data["action"]
        
        # Validate action type
        try:
            action_type = ActionType(action)
        except ValueError:
            valid_actions = [t.value for t in ActionType]
            return False, f"Invalid action '{action}'. Valid actions: {valid_actions}"
        
        # Validate required fields for each action type
        if action_type == ActionType.TOOL_CALL:
            if "tool_name" not in data:
                return False, "Missing 'tool_name' field for tool_call action"
            
            # Validate tool name
            tool_name = data["tool_name"]
            valid_tools = ["sam2", "dav2"]
            if tool_name not in valid_tools:
                return False, f"Invalid tool '{tool_name}'. Valid tools: {valid_tools}"
        
        elif action_type in [ActionType.REASONING, ActionType.FINAL_ANSWER]:
            if "text" not in data:
                return False, f"Missing 'text' field for {action} action"
            
            # Check that text is not empty
            if not data["text"] or not data["text"].strip():
                return False, f"Empty 'text' field for {action} action"
        
        return True, None
    
    @staticmethod
    def extract_action_info(parsed_data: Dict[str, Any]) -> Tuple[ActionType, Dict[str, Any]]:
        """
        Extract action type and relevant information from parsed data.
        
        Args:
            parsed_data: Validated parsed JSON data
            
        Returns:
            Tuple of (action_type, action_info)
        """
        action_type = ActionType(parsed_data["action"])
        
        if action_type == ActionType.TOOL_CALL:
            action_info = {
                "tool_name": parsed_data["tool_name"]
            }
        elif action_type in [ActionType.REASONING, ActionType.FINAL_ANSWER]:
            action_info = {
                "text": parsed_data["text"]
            }
        else:
            action_info = {}
        
        return action_type, action_info
    
    @staticmethod
    def is_final_answer(parsed_data: Dict[str, Any]) -> bool:
        """
        Check if the response contains a final answer.
        
        Args:
            parsed_data: Validated parsed JSON data
            
        Returns:
            True if this is a final answer, False otherwise
        """
        return parsed_data.get("action") == ActionType.FINAL_ANSWER.value
    
    @staticmethod
    def get_final_answer_text(parsed_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract final answer text if present.
        
        Args:
            parsed_data: Validated parsed JSON data
            
        Returns:
            Final answer text or None if not a final answer
        """
        if OutputParser.is_final_answer(parsed_data):
            return parsed_data.get("text")
        return None
