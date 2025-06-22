"""
Updated API utility functions that use tools instead of functions.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union
import sys
sys.path.append('.')
from api_utils import setup_api_key

def call_openrouter_with_tools(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "openai/gpt-4o-mini-2024-07-18",
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    stream: bool = False,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Make an API call to OpenRouter using the tools parameter.
    
    Args:
        prompt: Either a string prompt or a list of chat messages
        model: The model to use with provider prefix
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        system_prompt: Optional system prompt for chat models
        tools: Optional tool definitions (replaces functions)
        api_key: Optional API key
        stream: Whether to stream the response
        top_p: Nucleus sampling parameter (0-1)
        frequency_penalty: Penalizes repeated tokens (0-2)
        presence_penalty: Penalizes repeated topics (0-2)
        
    Returns:
        The API response as a dictionary
    """
    key = setup_api_key(api_key)
    
    # Prepare the messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(prompt)
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    
    # Convert functions to tools format if provided
    if tools:
        # If tools are already in the correct format, use them directly
        if all("type" in tool for tool in tools):
            payload["tools"] = tools
        else:
            # Convert from functions format to tools format
            formatted_tools = []
            for tool in tools:
                formatted_tools.append({
                    "type": "function",
                    "function": tool
                })
            payload["tools"] = formatted_tools
        
        payload["tool_choice"] = "auto"
    
    # Make the API request
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "http://localhost:3000",  # Required for site owner identification
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        if stream:
            return {"success": True, "response": response, "model": model}
        else:
            return {"success": True, "response": response.json(), "model": model}
    except Exception as e:
        return {"success": False, "error": str(e), "model": model}

def extract_tool_call(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract tool call information from an OpenRouter API response.
    
    Args:
        response: The API response dictionary
        
    Returns:
        Dictionary containing tool call details
    """
    if not response.get("success", False):
        return {"success": False, "error": response.get("error", "Unknown error")}
    
    try:
        message = response["response"]["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls:
            tool_call = tool_calls[0]  # Get the first tool call
            return {
                "success": True,
                "tool_id": tool_call.get("id"),
                "function_name": tool_call.get("function", {}).get("name"),
                "arguments": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            }
        return {"success": False, "error": "No tool call found in response"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_text_response_with_tools(response: Dict[str, Any]) -> str:
    """
    Extract the text response from an OpenRouter API response with tools support.
    
    Args:
        response: The API response dictionary
        
    Returns:
        The extracted text content
    """
    if not response.get("success", False):
        return f"Error: {response.get('error', 'Unknown error')}"
    
    try:
        message = response["response"]["choices"][0]["message"]
        return message.get("content", "")
    except (KeyError, IndexError):
        return "Error: Unable to extract response content"