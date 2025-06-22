"""
API utility functions for interacting with LLM providers via OpenRouter.
This module provides wrapper functions for making API calls to various LLM 
providers including the latest models from OpenAI, Anthropic, Google (Gemini), and DeepSeek.
"""

import os
import time
import json
import requests
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

# Load API keys from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HTTP_REFERER = os.getenv("HTTP_REFERER", "http://localhost:3000")

def setup_api_key(api_key: Optional[str] = None) -> str:
    """
    Set up and return the OpenRouter API key.
    
    Args:
        api_key: Optional API key. If not provided, uses the environment variable.
        
    Returns:
        The API key to use
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass as parameter.")
    
    return key

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_openrouter(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "openai/gpt-4o-2024-08-06",
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    stream: bool = False,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Make an API call to OpenRouter.
    
    Args:
        prompt: Either a string prompt or a list of chat messages
        model: The model to use with provider prefix:
               - OpenAI: "openai/gpt-4o-2024-08-06", "openai/gpt-4o-mini-2024-07-18", 
                 "openai/gpt-4-1106-preview"
               - Anthropic: "anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-opus-20240229",
                 "anthropic/claude-3-haiku-20240307"
               - Google: "google/gemini-2.5-pro", "google/gemini-2.5-flash", 
                 "google/gemini-1.5-pro", "google/gemini-1.5-flash"
               - DeepSeek: "deepseek/deepseek-chat-v3", "deepseek/deepseek-r1"
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        system_prompt: Optional system prompt for chat models
        functions: Optional function definitions for function calling
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
    
    # Add functions if provided (convert to tools format)
    if functions:
        # Convert functions to tools format
        tools = []
        for func in functions:
            tools.append({
                "type": "function",
                "function": func
            })
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    # Make the API request
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": HTTP_REFERER,  # Required for site owner identification
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

def process_streaming_response(response):
    """
    Process a streaming response from OpenRouter.
    
    Args:
        response: The streaming response object
        
    Yields:
        Text chunks as they arrive
    """
    if not response.get("success", False):
        yield f"Error: {response.get('error', 'Unknown error')}"
        return
    
    for line in response["response"].iter_lines():
        if line:
            # Remove 'data: ' prefix
            if line.startswith(b'data: '):
                line = line[6:]
            
            # Skip [DONE] message
            if line.strip() == b'[DONE]':
                continue
                
            try:
                json_data = json.loads(line)
                chunk = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                if chunk:
                    yield chunk
            except json.JSONDecodeError:
                continue

def extract_text_response(response: Dict[str, Any]) -> str:
    """
    Extract the text response from an OpenRouter API response.
    
    Args:
        response: The API response dictionary
        
    Returns:
        The extracted text content
    """
    if not response.get("success", False):
        return f"Error: {response.get('error', 'Unknown error')}"
    
    try:
        return response["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "Error: Unable to extract response content"

def extract_function_call(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract function call information from an OpenRouter API response.
    
    Args:
        response: The API response dictionary
        
    Returns:
        Dictionary containing function call details
    """
    if not response.get("success", False):
        return {"success": False, "error": response.get("error", "Unknown error")}
    
    try:
        message = response["response"]["choices"][0]["message"]
        
        # First try tool_calls (new format)
        tool_calls = message.get("tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            tool_call = tool_calls[0]  # Get the first tool call
            return {
                "success": True,
                "tool_id": tool_call.get("id"),
                "function_name": tool_call.get("function", {}).get("name"),
                "arguments": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            }
        
        # Fall back to function_call (legacy format)
        function_call = message.get("function_call", None)
        if function_call:
            return {
                "success": True,
                "function_name": function_call.get("name"),
                "arguments": json.loads(function_call.get("arguments", "{}"))
            }
            
        return {"success": False, "error": "No function call or tool call found in response"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_available_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get a list of available models from OpenRouter.
    
    Args:
        api_key: Optional API key
        
    Returns:
        List of available models with details
    """
    key = setup_api_key(api_key)
    
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": HTTP_REFERER
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def filter_models_by_provider(models: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    """
    Filter models by provider name.
    
    Args:
        models: List of models from get_available_models()
        provider: Provider name (e.g., "openai", "anthropic", "google", "deepseek")
        
    Returns:
        Filtered list of models
    """
    return [model for model in models if provider.lower() in model.get("id", "").lower()]

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an API call based on token usage.
    Uses latest pricing for common models (as of 2024).
    
    Args:
        model: The model name with provider prefix
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    # Extract the model name without provider prefix
    model_name = model.split('/')[-1] if '/' in model else model
    
    # Latest pricing (as of 2024)
    prices = {
        # OpenAI models - latest versions
        "gpt-4o-2024-08-06": {"input": 0.003 / 1000, "output": 0.01 / 1000},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        "gpt-4-1106-preview": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        
        # Common prefixes to match with actual model names
        "gpt-4o": {"input": 0.003 / 1000, "output": 0.01 / 1000},
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        
        # Anthropic models
        "claude-3-opus-20240229": {"input": 0.015 / 1000, "output": 0.075 / 1000},
        "claude-3-sonnet-20240229": {"input": 0.003 / 1000, "output": 0.015 / 1000},
        "claude-3-haiku-20240307": {"input": 0.00025 / 1000, "output": 0.00125 / 1000},
        
        # Google models
        "gemini-2.5-pro": {"input": 0.003 / 1000, "output": 0.01 / 1000},  # Estimated
        "gemini-2.5-flash": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},  # Same as GPT-4o mini
        "gemini-1.5-pro": {"input": 0.0035 / 1000, "output": 0.0035 / 1000},
        "gemini-1.5-flash": {"input": 0.00035 / 1000, "output": 0.00035 / 1000},
        
        # DeepSeek models
        "deepseek-chat-v3": {"input": 0.001 / 1000, "output": 0.005 / 1000},
        "deepseek-r1": {"input": 0.012 / 1000, "output": 0.012 / 1000},
    }
    
    # Find the closest matching model
    matching_model = None
    for m in prices:
        if m in model_name:
            matching_model = m
            break
    
    if matching_model:
        input_cost = input_tokens * prices[matching_model]["input"]
        output_cost = output_tokens * prices[matching_model]["output"]
        return input_cost + output_cost
    else:
        # Return a default estimate if model not found
        return (input_tokens * 0.005 / 1000) + (output_tokens * 0.015 / 1000)