"""
Utilities for token counting and context window management.
"""

import tiktoken
from typing import Dict, List, Any, Union, Optional, Tuple

# Mapping of model prefixes to encoding types
MODEL_TO_ENCODING = {
    # OpenAI models
    "gpt-4o": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    
    # Anthropic models - use cl100k as an approximation
    "claude-3": "cl100k_base",
    "claude-2": "cl100k_base",
    
    # Google models - use cl100k as an approximation
    "gemini": "cl100k_base",
    
    # DeepSeek models - use cl100k as an approximation
    "deepseek": "cl100k_base",
}

# Context window sizes for different models (updated for 2025)
CONTEXT_WINDOW_SIZES = {
    # OpenAI models
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    
    # Anthropic models (2025 updates)
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    
    # Google models (2025 updates)
    "gemini-2.0-flash-exp": 1000000,  # Latest Gemini 2.0 model
    "gemini-2.5-pro": 1000000,  # 1M tokens, with 2M coming soon
    "gemini-2.5-flash": 1000000,
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    
    # DeepSeek models (2025 updates)
    "deepseek-r1": 130000,  # Updated context window for R1
    "deepseek-chat-v3": 128000,
}

def get_encoding_type(model: str) -> str:
    """
    Get the encoding type for a given model.
    
    Args:
        model: Model name, which may include provider prefix (e.g., "openai/gpt-4o")
        
    Returns:
        The encoding type to use for token counting
    """
    # Remove provider prefix if present
    if '/' in model:
        model = model.split('/')[-1]
    
    # Find the matching prefix
    for model_prefix, encoding_type in MODEL_TO_ENCODING.items():
        if model.startswith(model_prefix):
            return encoding_type
    
    # Default to cl100k_base if no match found
    return "cl100k_base"

def get_context_window(model: str) -> int:
    """
    Get the context window size for a given model.
    
    Args:
        model: Model name, which may include provider prefix (e.g., "openai/gpt-4o")
        
    Returns:
        The context window size in tokens
    """
    # Remove provider prefix if present
    if '/' in model:
        model = model.split('/')[-1]
    
    # Check for exact match
    if model in CONTEXT_WINDOW_SIZES:
        return CONTEXT_WINDOW_SIZES[model]
    
    # Find the matching prefix
    for model_prefix, window_size in CONTEXT_WINDOW_SIZES.items():
        if model.startswith(model_prefix.split('-')[0]):
            return window_size
    
    # Default to 4096 if no match found
    return 4096

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string for a given model.
    
    Args:
        text: The text to count tokens in
        model: The model to use for token counting
        
    Returns:
        The number of tokens
    """
    encoding_type = get_encoding_type(model)
    encoding = tiktoken.get_encoding(encoding_type)
    return len(encoding.encode(text))

def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a list of chat messages for a given model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model to use for token counting
        
    Returns:
        The number of tokens
    """
    encoding_type = get_encoding_type(model)
    encoding = tiktoken.get_encoding(encoding_type)
    
    # Base tokens for the messages format
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    
    for message in messages:
        num_tokens += 4  # <|start|>{role}<|message|>{content}<|end|>
        
        role = message.get("role", "")
        content = message.get("content", "")
        
        num_tokens += len(encoding.encode(role))
        num_tokens += len(encoding.encode(content))
    
    return num_tokens

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200, model: str = "gpt-4o") -> List[str]:
    """
    Split a text into chunks of a specified token size with overlap.
    
    Args:
        text: The text to split
        chunk_size: Maximum token size for each chunk
        overlap: Number of tokens to overlap between chunks
        model: The model to use for token counting
        
    Returns:
        List of text chunks
    """
    encoding_type = get_encoding_type(model)
    encoding = tiktoken.get_encoding(encoding_type)
    
    tokens = encoding.encode(text)
    chunks = []
    
    i = 0
    while i < len(tokens):
        # Get the chunk tokens
        chunk_end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:chunk_end]
        
        # Decode the chunk
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        
        # Move to the next chunk with overlap
        i += chunk_size - overlap
    
    return chunks

def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The model to use for token counting
        
    Returns:
        Truncated text that fits within the token limit
    """
    encoding_type = get_encoding_type(model)
    encoding = tiktoken.get_encoding(encoding_type)
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def optimize_context(
    prompt: str,
    context: str,
    max_tokens: int,
    reserved_tokens: int = 500,
    model: str = "gpt-4o"
) -> str:
    """
    Optimize context to fit within a token limit along with a prompt.
    
    Args:
        prompt: The prompt that must be preserved
        context: Additional context that can be truncated if needed
        max_tokens: Maximum total tokens allowed
        reserved_tokens: Tokens to reserve for the response
        model: The model to use for token counting
        
    Returns:
        Optimized context that fits within the token limit
    """
    prompt_tokens = count_tokens(prompt, model)
    available_tokens = max_tokens - prompt_tokens - reserved_tokens
    
    if available_tokens <= 0:
        # Not enough space for any context
        return ""
    
    return truncate_to_token_limit(context, available_tokens, model)

def get_token_limit_with_buffer(model: str, buffer_percentage: float = 0.1) -> int:
    """
    Get the usable token limit for a model with a safety buffer.
    
    Args:
        model: The model name
        buffer_percentage: Percentage of the context window to reserve as buffer
        
    Returns:
        The usable token limit
    """
    context_window = get_context_window(model)
    buffer = int(context_window * buffer_percentage)
    return context_window - buffer

def estimate_max_completion_tokens(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o",
    buffer_tokens: int = 100
) -> int:
    """
    Estimate the maximum number of tokens available for completion.
    
    Args:
        messages: List of message dictionaries
        model: The model to use
        buffer_tokens: Additional tokens to reserve as a safety buffer
        
    Returns:
        The maximum tokens available for completion
    """
    context_window = get_context_window(model)
    input_tokens = count_message_tokens(messages, model)
    
    max_completion = context_window - input_tokens - buffer_tokens
    return max(0, max_completion)

def optimize_messages_for_context(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o",
    reserved_tokens: int = 500
) -> List[Dict[str, str]]:
    """
    Optimize a list of messages to fit within a model's context window.
    Preserves the most recent messages while truncating older ones if needed.
    
    Args:
        messages: List of message dictionaries
        model: The model to use
        reserved_tokens: Tokens to reserve for the response
        
    Returns:
        Optimized list of messages
    """
    context_window = get_context_window(model)
    available_tokens = context_window - reserved_tokens
    
    # If messages fit, return as is
    if count_message_tokens(messages, model) <= available_tokens:
        return messages
    
    # Keep removing oldest messages until we fit
    optimized = messages.copy()
    while optimized and count_message_tokens(optimized, model) > available_tokens:
        # Remove the oldest message (excluding system message)
        if optimized[0]["role"] == "system" and len(optimized) > 1:
            optimized.pop(1)
        else:
            optimized.pop(0)
    
    return optimized