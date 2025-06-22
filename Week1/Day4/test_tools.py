import sys
import json
sys.path.append('../Day2')

from api_utils import setup_api_key
import requests

# Define the tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Set up the API call directly
key = setup_api_key()
    
messages = [
    {"role": "user", "content": "What is the weather in London?"}
]

payload = {
    "model": "openai/gpt-4o-mini-2024-07-18",
    "messages": messages,
    "temperature": 0.7,
    "max_tokens": 500,
    "tools": tools,
    "tool_choice": "auto"
}

headers = {
    "Authorization": f"Bearer {key}",
    "HTTP-Referer": "http://localhost:3000",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=payload
)

# Print results
if response.status_code == 200:
    result = response.json()
    print("Success!")
    print("Response structure:", json.dumps(result, indent=2))
    
    # Check for tool calls
    message = result.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    
    if tool_calls:
        print("\nTool call found:")
        for tool_call in tool_calls:
            print(f"Tool ID: {tool_call.get('id')}")
            print(f"Tool type: {tool_call.get('type')}")
            print(f"Function: {tool_call.get('function', {}).get('name')}")
            print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
    else:
        print("\nNo tool calls found in the response")
        print("Message content:", message.get("content"))
else:
    print(f"Error: {response.status_code}")
    print(response.text)