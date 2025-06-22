"""
Utility functions for parsing and validating structured outputs from LLM responses.
"""

import re
import json
from typing import Dict, List, Any, Union, Optional, Tuple, Callable

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from text, even if it's part of a larger text.
    
    Args:
        text: The text that may contain JSON
        
    Returns:
        Extracted JSON as a Python dictionary, or None if no valid JSON found
    """
    # Try to find JSON between curly braces
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            return extract_json_with_fixes(json_str)
    return None

def extract_json_with_fixes(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract JSON with common fixes for LLM formatting errors.
    
    Args:
        json_str: The potential JSON string with possible formatting issues
        
    Returns:
        Fixed JSON as a Python dictionary, or None if unfixable
    """
    # Common fixes
    fixes = [
        # Fix missing quotes around keys
        lambda s: re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', s),
        # Fix single quotes
        lambda s: s.replace("'", '"'),
        # Fix trailing commas
        lambda s: re.sub(r',\s*}', '}', s).replace(',]', ']'),
        # Fix unescaped quotes in values
        lambda s: re.sub(r'(?<!"): "(?:(?:[^"\\]|\\.)*)"', lambda m: m.group(0).replace('\\"', '\\\\"'), s),
    ]
    
    # Try each fix
    for fix in fixes:
        try:
            fixed_str = fix(json_str)
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            continue
    
    return None

def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: The markdown text containing code blocks
        language: Optional language filter (e.g., "python", "javascript")
        
    Returns:
        List of extracted code blocks
    """
    if language:
        pattern = r'```(?:' + language + r'|)\s*(.*?)\s*```'
    else:
        pattern = r'```(?:\w*)\s*(.*?)\s*```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def extract_python_code(text: str) -> str:
    """
    Extract Python code from a markdown text.
    
    Args:
        text: The markdown text containing Python code blocks
        
    Returns:
        The first Python code block found, or an empty string if none
    """
    blocks = extract_code_blocks(text, "python")
    return blocks[0] if blocks else ""

def extract_list_items(text: str) -> List[str]:
    """
    Extract list items from markdown text.
    
    Args:
        text: The markdown text containing list items
        
    Returns:
        List of extracted items
    """
    # Match both bullet points and numbered lists
    pattern = r'(?:^|\n)(?:\*|-|\d+\.)\s+(.*?)(?=(?:^|\n)(?:\*|-|\d+\.)|$)'
    matches = re.findall(pattern, text, re.MULTILINE)
    return [item.strip() for item in matches]

def extract_function_call(text: str) -> Dict[str, Any]:
    """
    Extract function call information from text.
    
    Args:
        text: Text that may contain a function call description
        
    Returns:
        Dictionary with function name and arguments
    """
    function_pattern = r'(?:function|call|invoke)\s+`?([a-zA-Z0-9_]+)`?'
    function_match = re.search(function_pattern, text, re.IGNORECASE)
    
    function_name = function_match.group(1) if function_match else None
    arguments = extract_json(text)
    
    return {
        "function_name": function_name,
        "arguments": arguments or {}
    }

def validate_json_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate JSON data against a schema definition.
    
    Args:
        data: The data to validate
        schema: The schema definition
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
    
    # Check property types
    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        if prop_name in data:
            prop_value = data[prop_name]
            prop_type = prop_schema.get("type")
            
            # Check property type
            if prop_type == "string" and not isinstance(prop_value, str):
                errors.append(f"Field '{prop_name}' should be a string")
            elif prop_type == "number" and not isinstance(prop_value, (int, float)):
                errors.append(f"Field '{prop_name}' should be a number")
            elif prop_type == "integer" and not isinstance(prop_value, int):
                errors.append(f"Field '{prop_name}' should be an integer")
            elif prop_type == "boolean" and not isinstance(prop_value, bool):
                errors.append(f"Field '{prop_name}' should be a boolean")
            elif prop_type == "array":
                if not isinstance(prop_value, list):
                    errors.append(f"Field '{prop_name}' should be an array")
                else:
                    # Check array items
                    item_schema = prop_schema.get("items", {})
                    for i, item in enumerate(prop_value):
                        if item_schema.get("type") == "object":
                            is_valid, item_errors = validate_json_against_schema(item, item_schema)
                            for error in item_errors:
                                errors.append(f"Error in '{prop_name}[{i}]': {error}")
            elif prop_type == "object":
                if not isinstance(prop_value, dict):
                    errors.append(f"Field '{prop_name}' should be an object")
                else:
                    # Recursively validate nested objects
                    is_valid, obj_errors = validate_json_against_schema(prop_value, prop_schema)
                    for error in obj_errors:
                        errors.append(f"Error in '{prop_name}': {error}")
    
    return len(errors) == 0, errors

def parse_structured_output(
    text: str, 
    expected_format: str = "json",
    schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse and validate structured output from LLM response.
    
    Args:
        text: The text to parse
        expected_format: The expected format (e.g., "json", "function_call")
        schema: Optional schema for validation
        
    Returns:
        Parsed output as a dictionary, with validation status
    """
    parsed_data = None
    is_valid = False
    errors = []
    
    if expected_format == "json":
        parsed_data = extract_json(text)
        if parsed_data is None:
            errors.append("Failed to extract valid JSON")
    elif expected_format == "function_call":
        parsed_data = extract_function_call(text)
        if not parsed_data["function_name"]:
            errors.append("Failed to extract function name")
    elif expected_format == "code":
        parsed_data = {"code": extract_python_code(text)}
        if not parsed_data["code"]:
            errors.append("Failed to extract code block")
    elif expected_format == "list":
        items = extract_list_items(text)
        parsed_data = {"items": items}
        if not items:
            errors.append("Failed to extract list items")
    
    # Validate against schema if provided
    if schema and parsed_data:
        is_valid, schema_errors = validate_json_against_schema(parsed_data, schema)
        errors.extend(schema_errors)
    else:
        is_valid = len(errors) == 0
    
    return {
        "data": parsed_data,
        "is_valid": is_valid,
        "errors": errors
    }

def parse_markdown_sections(text: str) -> Dict[str, str]:
    """
    Parse a markdown text into sections based on headers.
    
    Args:
        text: Markdown text with headers
        
    Returns:
        Dictionary mapping section names to their content
    """
    # Split by headers
    header_pattern = r'^(#+)\s+(.+?)$'
    lines = text.split('\n')
    sections = {}
    
    current_section = "default"
    current_content = []
    
    for line in lines:
        header_match = re.match(header_pattern, line)
        if header_match:
            # Save the previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
                current_content = []
            
            # Start a new section
            current_section = header_match.group(2).strip()
        else:
            current_content.append(line)
    
    # Save the last section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections