import sys
sys.path.append('../Day2')

from api_utils import call_openrouter, extract_function_call, extract_text_response

# Test function calling with the model
functions = [
    {
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
]

resp = call_openrouter(
    prompt="What is the weather in London?",
    model="openai/gpt-4o-mini-2024-07-18",
    functions=functions
)

print('Response structure:', resp.keys())
print('Success status:', resp.get('success'))
print('Raw response:', resp.get('response', {}))
print('\nFunction call result:', extract_function_call(resp))