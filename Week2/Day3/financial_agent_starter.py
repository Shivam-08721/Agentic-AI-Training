import sys
import json
import os
from typing import Dict, List, Any, Optional, Union

# Import utilities from previous days
sys.path.append('.')
from updated_api_utils import call_openrouter_with_tools, extract_tool_call, extract_text_response_with_tools

# Set up your API key
# os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Define simulated financial data tools
financial_tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current price of a stock by ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_performance",
        "description": "Get the performance of a stock over a specified time period",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                },
                "period": {
                    "type": "string",
                    "description": "Time period (day, week, month, year)",
                    "enum": ["day", "week", "month", "year"]
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "calculate_investment_return",
        "description": "Calculate investment returns with compound interest",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "Initial investment amount in dollars"
                },
                "rate": {
                    "type": "number",
                    "description": "Annual interest rate as a percentage (e.g., 5 for 5%)"
                },
                "years": {
                    "type": "number",
                    "description": "Investment duration in years"
                },
                "compound_frequency": {
                    "type": "string",
                    "description": "Compounding frequency",
                    "enum": ["annually", "semi-annually", "quarterly", "monthly", "daily"]
                }
            },
            "required": ["principal", "rate", "years", "compound_frequency"]
        }
    },
    {
        "name": "get_company_financials",
        "description": "Get key financial metrics for a company",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["pe_ratio", "market_cap", "dividend_yield", "revenue", "profit_margin", "debt_to_equity"]
                    },
                    "description": "Financial metrics to retrieve"
                }
            },
            "required": ["ticker", "metrics"]
        }
    },
    {
        "name": "get_market_data",
        "description": "Get current data for a market index or economic indicator",
        "parameters": {
            "type": "object",
            "properties": {
                "indicator": {
                    "type": "string",
                    "description": "Market index or economic indicator",
                    "enum": ["S&P500", "NASDAQ", "DowJones", "FederalFundsRate", "InflationRate", "GDP", "UnemploymentRate"]
                }
            },
            "required": ["indicator"]
        }
    }
]

# Simulated tool implementation
def simulate_tool_execution(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate execution of financial tools with mock data."""
    
    mock_data = {
        "get_stock_price": {
            "AAPL": 198.73,
            "MSFT": 417.14,
            "GOOGL": 164.25,
            "AMZN": 185.23,
            "TSLA": 232.59
        },
        "get_stock_performance": {
            "AAPL": {
                "day": -0.5,
                "week": 1.2,
                "month": 3.8,
                "year": 15.7
            },
            "MSFT": {
                "day": 0.3,
                "week": -0.8,
                "month": 5.2,
                "year": 28.6
            },
            "TSLA": {
                "day": 2.1,
                "week": -3.5,
                "month": -8.2,
                "year": 12.3
            }
        },
        "get_company_financials": {
            "AAPL": {
                "pe_ratio": 32.5,
                "market_cap": "3.1T",
                "dividend_yield": 0.52,
                "revenue": "383.3B",
                "profit_margin": 25.3,
                "debt_to_equity": 2.31
            },
            "MSFT": {
                "pe_ratio": 37.2,
                "market_cap": "3.2T",
                "dividend_yield": 0.71,
                "revenue": "211.9B",
                "profit_margin": 36.8,
                "debt_to_equity": 0.42
            },
            "AMZN": {
                "pe_ratio": 42.6,
                "market_cap": "1.9T",
                "dividend_yield": 0.0,
                "revenue": "574.8B",
                "profit_margin": 5.1,
                "debt_to_equity": 0.67
            }
        },
        "get_market_data": {
            "S&P500": {"value": 5473.15, "change": 0.13, "change_percent": 0.002},
            "NASDAQ": {"value": 17245.48, "change": 27.64, "change_percent": 0.16},
            "DowJones": {"value": 39068.58, "change": -34.08, "change_percent": -0.09},
            "FederalFundsRate": {"current": 5.5, "previous": 5.5, "change": 0.0},
            "InflationRate": {"current": 3.1, "previous": 3.4, "change": -0.3},
            "GDP": {"growth_rate": 2.1, "previous": 1.6, "change": 0.5}
        }
    }
    
    if function_name == "get_stock_price":
        ticker = arguments.get("ticker", "").upper()
        if ticker in mock_data["get_stock_price"]:
            return {
                "success": True,
                "price": mock_data["get_stock_price"][ticker],
                "currency": "USD",
                "timestamp": "2024-05-19T16:00:00Z"  # Simulated timestamp
            }
        else:
            return {"success": False, "error": f"No data available for ticker {ticker}"}
    
    elif function_name == "get_stock_performance":
        ticker = arguments.get("ticker", "").upper()
        period = arguments.get("period", "")
        
        if ticker in mock_data["get_stock_performance"]:
            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "percent_change": mock_data["get_stock_performance"][ticker][period],
                "start_date": "2023-05-19",  # Simulated dates
                "end_date": "2024-05-19"
            }
        else:
            return {"success": False, "error": f"No performance data for {ticker} over {period}"}
    
    elif function_name == "calculate_investment_return":
        principal = arguments.get("principal", 0)
        rate = arguments.get("rate", 0) / 100  # Convert percentage to decimal
        years = arguments.get("years", 0)
        compound_frequency = arguments.get("compound_frequency", "annually")
        
        # Map compound frequency to number of periods per year
        frequency_map = {
            "annually": 1,
            "semi-annually": 2,
            "quarterly": 4,
            "monthly": 12,
            "daily": 365
        }
        
        n = frequency_map[compound_frequency]
        # Compound interest formula: A = P(1 + r/n)^(nt)
        final_amount = principal * (1 + rate/n)**(n*years)
        interest_earned = final_amount - principal
        
        return {
            "success": True,
            "principal": principal,
            "rate": rate * 100,  # Convert back to percentage
            "years": years,
            "compound_frequency": compound_frequency,
            "final_amount": round(final_amount, 2),
            "interest_earned": round(interest_earned, 2)
        }
    
    elif function_name == "get_company_financials":
        ticker = arguments.get("ticker", "").upper()
        metrics = arguments.get("metrics", [])
        
        if ticker in mock_data["get_company_financials"]:
            result = {"success": True, "ticker": ticker}
            for metric in metrics:
                if metric in mock_data["get_company_financials"][ticker]:
                    result[metric] = mock_data["get_company_financials"][ticker][metric]
            return result
        else:
            return {"success": False, "error": f"No financial data available for {ticker}"}
    
    elif function_name == "get_market_data":
        indicator = arguments.get("indicator", "")
        
        if indicator in mock_data["get_market_data"]:
            return {
                "success": True,
                "indicator": indicator,
                **mock_data["get_market_data"][indicator]
            }
        else:
            return {"success": False, "error": f"No data available for indicator {indicator}"}
    
    return {"success": False, "error": f"Unknown function: {function_name}"}


class FinancialAdvisorAgent:
    def __init__(self, model="openai/gpt-4o-mini-2024-07-18"):
        self.model = model
        self.tools = financial_tools
        self.conversation_history = []
        
    def add_system_message(self, system_message):
        """Add a system message to the conversation history."""
        self.conversation_history.append({"role": "system", "content": system_message})
    
    def add_user_message(self, user_message):
        """Add a user message to the conversation history."""
        self.conversation_history.append({"role": "user", "content": user_message})
    
    def add_assistant_message(self, assistant_message):
        """Add an assistant message to the conversation history."""
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
    def add_tool_response(self, tool_call_id, tool_name, tool_response):
        """Add a tool response to the conversation history."""
        self.conversation_history.append({
            "role": "tool", 
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(tool_response)
        })
    
    def execute_react_loop(self, query, max_steps=5):
        """Execute the ReAct loop for a financial query."""
        
        # Reset conversation and add system prompt
        self.conversation_history = []
        
        # Add system prompt with ReAct instructions
        system_prompt = """
        You are a financial advisor agent that helps people with investment questions and financial planning. 
        Use the ReAct framework to solve financial queries by following these steps:
        
        1. THOUGHT: Analyze the query and determine what financial information is needed to answer it.
        2. ACTION: Use the appropriate financial tool to retrieve relevant data.
        3. OBSERVATION: Review the tool's response and extract key insights.
        4. THOUGHT: Analyze the information and determine next steps or final recommendation.
        5. Repeat steps 2-4 as needed until you have all necessary information.
        6. When ready, provide your final RECOMMENDATION with well-reasoned financial advice.
        
        After your recommendation, include a REFLECTION on your reasoning process and the confidence in your advice.
        
        IMPORTANT: Always cite your sources and explain your financial reasoning. Be transparent about the limitations 
        of your advice. Use tools to get real data rather than making assumptions.
        """
        
        self.add_system_message(system_prompt)
        self.add_user_message(query)
        
        # Execute the ReAct loop
        for step in range(max_steps):
            # Get the next action from the model
            response = call_openrouter_with_tools(
                prompt=self.conversation_history,
                model=self.model,
                tools=self.tools,
                temperature=0.2,
                max_tokens=1000
            )
            
            # FIXED: Handle the response structure correctly
            # First check if the API call was successful
            if isinstance(response, dict) and response.get('success') == False:
                error_msg = response.get('error', 'Unknown API error')
                print(f"API Error: {error_msg}")
                return f"Error occurred: {error_msg}"
            
            # Check if response has the expected structure
            if "response" in response and "choices" in response["response"]:
                # Handle nested response structure (successful API call)
                message = response["response"]["choices"][0]["message"]
            else:
                # This shouldn't happen with a successful response, but handle it
                print(f"Unexpected response structure: {response}")
                return "Unexpected response structure from API"
            
            print(f"Step {step+1} - Message received:")
            print(message)
            
            if "tool_calls" in message and message["tool_calls"]:
                # Extract the tool call
                tool_call = message["tool_calls"][0]
                tool_call_id = tool_call["id"]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                # Add the assistant's message with tool calls to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": message["tool_calls"]
                })
                
                # Execute the tool
                tool_result = simulate_tool_execution(function_name, arguments)
                
                # Add the tool response to the conversation with the correct tool_call_id
                self.add_tool_response(tool_call_id, function_name, tool_result)
                
                print(f"Step {step+1}: Used tool '{function_name}' with result: {tool_result}")
                print(f"Conversation history length: {len(self.conversation_history)}")
            else:
                # The model provided a final answer
                final_response = message.get("content", "No content provided")
                self.add_assistant_message(final_response)
                print(f"\nFinal Response:\n{final_response}")
                return final_response
        
        # If we reach the maximum number of steps without a final answer
        return "Reached maximum number of reasoning steps without a complete answer."
    
    
    # EXERCISE: Implement the reflection mechanism
    def reflect_on_response(self, response, query):
        """Reflect on the quality of the financial advice provided.
        
        This function should:
        1. Evaluate whether the advice is well-reasoned and evidence-based
        2. Check for any missing important considerations
        3. Assess the confidence level of the recommendation
        4. Suggest improvements if necessary
        """
        # TODO: Implement this reflection mechanism
        pass

# Create our financial agent
financial_agent = FinancialAdvisorAgent()

# Test with a simple stock price query
query = "What is the current price of Apple stock?"
response = financial_agent.execute_react_loop(query)

def evaluate_financial_agent(agent, test_queries, criteria=None):
    """Evaluate a financial agent on multiple test queries."""
    if criteria is None:
        criteria = [
            "Factual Accuracy",     # Was the financial information correct?
            "Tool Usage",           # Did the agent use appropriate tools?
            "Reasoning Quality",    # Was the financial reasoning sound?
            "Comprehensiveness",    # Did the answer cover all aspects of the query?
            "Clarity",              # Was the advice clearly presented?
            "Risk Disclosure"       # Did the agent mention relevant risks?
        ]
    
    results = []
    
    for query in test_queries:
        print(f"\nEvaluating query: {query}")
        response = agent.execute_react_loop(query)
        
        # For a real evaluation, you would have human raters or an evaluation model
        # For this workshop, we'll manually score each response
        print("\nPlease rate the response on the following criteria (1-5):")
        scores = {}
        for criterion in criteria:
            try:
                score = int(input(f"{criterion} (1-5): "))
                scores[criterion] = min(max(score, 1), 5)  # Ensure score is between 1-5
            except ValueError:
                scores[criterion] = 3  # Default to middle score if invalid input
        
        results.append({
            "query": query,
            "response": response,
            "scores": scores,
            "average_score": sum(scores.values()) / len(scores)
        })
    
    # Calculate overall scores
    overall_average = sum(result["average_score"] for result in results) / len(results)
    criterion_averages = {}
    for criterion in criteria:
        criterion_averages[criterion] = sum(result["scores"][criterion] for result in results) / len(results)
    
    print(f"\nOverall Average Score: {overall_average:.2f}/5.00")
    print("\nAverage Scores by Criterion:")
    for criterion, avg in criterion_averages.items():
        print(f"{criterion}: {avg:.2f}/5.00")
    
    return results, criterion_averages, overall_average