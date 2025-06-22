"""
API utility functions for interacting with LLM providers via AWS Bedrock.
This module provides wrapper functions for making API calls to various LLM 
providers including Anthropic Claude, Amazon Titan, Meta Llama, and other models available on Bedrock.
"""

import os
import json
import boto3
from typing import Dict, Any, List, Optional, Union, Generator
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from botocore.exceptions import ClientError, BotoCoreError

# Load environment variables from .env file
load_dotenv()

# Load AWS credentials from environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")  # Optional for temporary credentials

def setup_bedrock_client(region: Optional[str] = None) -> boto3.client:
    """
    Set up and return the AWS Bedrock client.
    
    Args:
        region: AWS region to use. If not provided, uses environment variable or default.
        
    Returns:
        Configured boto3 Bedrock runtime client
        
    Raises:
        ValueError: If AWS credentials are not properly configured
    """
    region = region or AWS_REGION
    
    try:
        client_kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': region
        }
        
        # Add credentials if provided
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client_kwargs['aws_access_key_id'] = AWS_ACCESS_KEY_ID
            client_kwargs['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY
            
            if AWS_SESSION_TOKEN:
                client_kwargs['aws_session_token'] = AWS_SESSION_TOKEN
        
        client = boto3.client(**client_kwargs)
        return client
    
    except Exception as e:
        raise ValueError(f"Failed to setup Bedrock client: {str(e)}. "
                        "Ensure AWS credentials are properly configured.")

def _prepare_anthropic_request(
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
    top_p: float,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Prepare request body for Anthropic Claude models."""
    request_body = {
        "anthropic_version": "bedrock-2023-05-20",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": messages
    }
    
    if system_prompt:
        request_body["system"] = system_prompt
    
    return request_body

def _prepare_titan_request(
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
    top_p: float,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Prepare request body for Amazon Titan models."""
    # Convert messages to a single prompt for Titan
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    
    for msg in messages:
        role = msg["role"].title()
        content = msg["content"]
        prompt_parts.append(f"{role}: {content}")
    
    prompt_parts.append("Assistant:")
    input_text = "\n\n".join(prompt_parts)
    
    return {
        "inputText": input_text,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "stopSequences": ["User:", "Human:"]
        }
    }

def _prepare_llama_request(
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
    top_p: float,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Prepare request body for Meta Llama models."""
    # Convert messages to a single prompt for Llama
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    
    for msg in messages:
        role = msg["role"].title()
        content = msg["content"]
        prompt_parts.append(f"{role}: {content}")
    
    prompt_parts.append("Assistant:")
    prompt = "\n\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

def _prepare_cohere_request(
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
    top_p: float,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Prepare request body for Cohere Command models."""
    # Get the last user message as the prompt
    user_message = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    
    request_body = {
        "prompt": user_message,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "p": top_p,
        "return_likelihoods": "NONE"
    }
    
    if system_prompt:
        request_body["prompt"] = f"{system_prompt}\n\n{user_message}"
    
    return request_body

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_bedrock(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None,
    region: Optional[str] = None,
    stream: bool = False,
    top_p: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Make an API call to AWS Bedrock.
    
    Args:
        prompt: Either a string prompt or a list of chat messages
        model: The Bedrock model ID to use:
               - Anthropic: "anthropic.claude-3-5-haiku-20241022-v1:0", 
                 "anthropic.claude-3-5-sonnet-20241022-v2:0", 
                 "anthropic.claude-3-opus-20240229-v1:0"
               - Amazon: "amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"
               - Meta: "meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"
               - Cohere: "cohere.command-text-v14", "cohere.command-light-text-v14"
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        system_prompt: Optional system prompt for chat models
        region: AWS region to use
        stream: Whether to stream the response (supported for some models)
        top_p: Nucleus sampling parameter (0-1)
        **kwargs: Additional model-specific parameters
        
    Returns:
        The API response as a dictionary with success status
    """
    try:
        client = setup_bedrock_client(region)
        
        # Prepare the messages
        messages = []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)
        
        # Prepare request body based on model type
        if "anthropic.claude" in model:
            request_body = _prepare_anthropic_request(messages, temperature, max_tokens, top_p, system_prompt)
        elif "amazon.titan" in model:
            request_body = _prepare_titan_request(messages, temperature, max_tokens, top_p, system_prompt)
        elif "meta.llama" in model:
            request_body = _prepare_llama_request(messages, temperature, max_tokens, top_p, system_prompt)
        elif "cohere.command" in model:
            request_body = _prepare_cohere_request(messages, temperature, max_tokens, top_p, system_prompt)
        else:
            return {"success": False, "error": f"Unsupported model type: {model}", "model": model}
        
        # Make the API request
        if stream:
            response = client.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            return {"success": True, "response": response, "model": model, "streaming": True}
        else:
            response = client.invoke_model(
                modelId=model,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return {"success": True, "response": response_body, "model": model, "streaming": False}
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return {"success": False, "error": f"AWS Bedrock Error [{error_code}]: {error_message}", "model": model}
    except Exception as e:
        return {"success": False, "error": str(e), "model": model}

def process_streaming_response(response: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Process a streaming response from Bedrock.
    
    Args:
        response: The streaming response dictionary from call_bedrock
        
    Yields:
        Text chunks as they arrive
    """
    if not response.get("success", False):
        yield f"Error: {response.get('error', 'Unknown error')}"
        return
    
    if not response.get("streaming", False):
        yield "Error: Response is not a streaming response"
        return
    
    try:
        stream = response["response"]['body']
        model = response.get("model", "")
        
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk['bytes'])
                
                # Handle different model response formats
                if "anthropic.claude" in model:
                    if chunk_data['type'] == 'content_block_delta':
                        if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                            yield chunk_data['delta']['text']
                elif "amazon.titan" in model:
                    if 'outputText' in chunk_data:
                        yield chunk_data['outputText']
                elif "meta.llama" in model:
                    if 'generation' in chunk_data:
                        yield chunk_data['generation']
                        
    except Exception as e:
        yield f"Error processing stream: {str(e)}"

def extract_text_response(response: Dict[str, Any]) -> str:
    """
    Extract the text response from a Bedrock API response.
    
    Args:
        response: The API response dictionary
        
    Returns:
        The extracted text content
    """
    if not response.get("success", False):
        return f"Error: {response.get('error', 'Unknown error')}"
    
    if response.get("streaming", False):
        return "Error: Cannot extract text from streaming response. Use process_streaming_response instead."
    
    try:
        response_body = response["response"]
        model = response.get("model", "")
        
        # Extract text based on model type
        if "anthropic.claude" in model:
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text']
        elif "amazon.titan" in model:
            results = response_body.get('results', [])
            if results:
                return results[0].get('outputText', '')
        elif "meta.llama" in model:
            return response_body.get('generation', '')
        elif "cohere.command" in model:
            generations = response_body.get('generations', [])
            if generations:
                return generations[0].get('text', '')
        
        return "Error: Unable to extract response content from model response"
    except Exception as e:
        return f"Error extracting response: {str(e)}"

def get_available_models(region: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get a list of available models from AWS Bedrock.
    
    Args:
        region: AWS region to query
        
    Returns:
        List of available foundation models with details
    """
    try:
        client = boto3.client('bedrock', region_name=region or AWS_REGION)
        response = client.list_foundation_models()
        return response.get('modelSummaries', [])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def filter_models_by_provider(models: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    """
    Filter models by provider name.
    
    Args:
        models: List of models from get_available_models()
        provider: Provider name (e.g., "anthropic", "amazon", "meta", "cohere")
        
    Returns:
        Filtered list of models
    """
    return [model for model in models if provider.lower() in model.get("modelId", "").lower()]

def get_model_info(model_id: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: The Bedrock model ID
        region: AWS region to query
        
    Returns:
        Model details dictionary
    """
    try:
        client = boto3.client('bedrock', region_name=region or AWS_REGION)
        response = client.get_foundation_model(modelIdentifier=model_id)
        return response.get('modelDetails', {})
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an API call based on token usage.
    Uses latest Bedrock pricing for common models (as of 2024).
    
    Args:
        model: The Bedrock model ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    # Bedrock pricing (as of 2024) - per 1000 tokens
    prices = {
        # Anthropic Claude models
        "anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        
        # Amazon Titan models
        "amazon.titan-text-express-v1": {"input": 0.0013, "output": 0.0017},
        "amazon.titan-text-lite-v1": {"input": 0.0003, "output": 0.0004},
        
        # Meta Llama models
        "meta.llama3-70b-instruct-v1:0": {"input": 0.00265, "output": 0.0035},
        "meta.llama3-8b-instruct-v1:0": {"input": 0.0003, "output": 0.0006},
        
        # Cohere models
        "cohere.command-text-v14": {"input": 0.0015, "output": 0.002},
        "cohere.command-light-text-v14": {"input": 0.0003, "output": 0.0006},
        
        # AI21 models
        "ai21.j2-ultra-v1": {"input": 0.0188, "output": 0.0188},
        "ai21.j2-mid-v1": {"input": 0.0125, "output": 0.0125},
    }
    
    # Find exact match or closest match
    if model in prices:
        input_cost = (input_tokens / 1000) * prices[model]["input"]
        output_cost = (output_tokens / 1000) * prices[model]["output"]
        return input_cost + output_cost
    
    # Find partial match
    for price_model in prices:
        if any(part in model for part in price_model.split('.')):
            input_cost = (input_tokens / 1000) * prices[price_model]["input"]
            output_cost = (output_tokens / 1000) * prices[price_model]["output"]
            return input_cost + output_cost
    
    # Default estimate if model not found
    return (input_tokens / 1000 * 0.003) + (output_tokens / 1000 * 0.015)

def get_token_usage(response: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract token usage information from a Bedrock response.
    
    Args:
        response: The API response dictionary
        
    Returns:
        Dictionary with input_tokens and output_tokens counts
    """
    if not response.get("success", False) or response.get("streaming", False):
        return {"input_tokens": 0, "output_tokens": 0}
    
    try:
        response_body = response["response"]
        model = response.get("model", "")
        
        # Extract token usage based on model type
        if "anthropic.claude" in model:
            usage = response_body.get('usage', {})
            return {
                "input_tokens": usage.get('input_tokens', 0),
                "output_tokens": usage.get('output_tokens', 0)
            }
        elif "amazon.titan" in model:
            return {
                "input_tokens": response_body.get('inputTextTokenCount', 0),
                "output_tokens": len(response_body.get('results', [{}])[0].get('outputText', '').split()) * 1.3  # Rough estimate
            }
        else:
            # For models without explicit token counts, estimate
            text = extract_text_response(response)
            return {
                "input_tokens": 0,  # Not available
                "output_tokens": len(text.split()) * 1.3  # Rough estimate: 1.3 tokens per word
            }
    except Exception:
        return {"input_tokens": 0, "output_tokens": 0}

# Convenience functions for specific model families
def call_claude(prompt: Union[str, List[Dict[str, str]]], model: str = "anthropic.claude-3-5-haiku-20241022-v1:0", **kwargs) -> Dict[str, Any]:
    """Convenience function for calling Claude models."""
    return call_bedrock(prompt, model, **kwargs)

def call_titan(prompt: Union[str, List[Dict[str, str]]], model: str = "amazon.titan-text-express-v1", **kwargs) -> Dict[str, Any]:
    """Convenience function for calling Titan models."""
    return call_bedrock(prompt, model, **kwargs)

def call_llama(prompt: Union[str, List[Dict[str, str]]], model: str = "meta.llama3-70b-instruct-v1:0", **kwargs) -> Dict[str, Any]:
    """Convenience function for calling Llama models."""
    return call_bedrock(prompt, model, **kwargs)