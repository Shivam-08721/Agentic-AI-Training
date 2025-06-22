"""
Display utilities for enhanced output formatting in Jupyter notebooks.
Provides functions to render LLM responses with proper markdown formatting.
"""

from IPython.display import Markdown, HTML, display
from typing import Optional, Dict, Any, Union
import json

def display_markdown(text: str, title: Optional[str] = None) -> None:
    """
    Display text with markdown formatting in Jupyter notebooks.
    
    Args:
        text: The text to display with markdown formatting
        title: Optional title to display above the text
    """
    if title:
        display(Markdown(f"### {title}\n\n{text}"))
    else:
        display(Markdown(text))

def display_llm_response(response: str, model: str = "", title: str = "AI Response") -> None:
    """
    Display LLM response with enhanced formatting.
    
    Args:
        response: The LLM response text
        model: Optional model name to display
        title: Title for the response section
    """
    model_info = f" ({model})" if model else ""
    
    # Create a styled container for the response
    html_content = f"""
    <div style="
        border: 2px solid #e1e5e9; 
        border-radius: 8px; 
        padding: 15px; 
        margin: 10px 0; 
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="
            color: #2c3e50; 
            margin: 0 0 10px 0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            🤖 {title}{model_info}
        </h4>
        <div style="
            background: white; 
            padding: 15px; 
            border-radius: 5px;
            line-height: 1.6;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        ">
    """
    
    # If the response contains markdown, render it properly
    if any(marker in response for marker in ['**', '*', '#', '`', '-', '1.']):
        # Response likely contains markdown
        display(HTML(html_content))
        display(Markdown(response))
        display(HTML("</div></div>"))
    else:
        # Plain text response
        html_content += f"{response}</div></div>"
        display(HTML(html_content))

def display_code_output(code: str, language: str = "python", title: str = "Generated Code") -> None:
    """
    Display code with syntax highlighting.
    
    Args:
        code: The code to display
        language: Programming language for syntax highlighting
        title: Title for the code section
    """
    display(Markdown(f"### {title}\n\n```{language}\n{code}\n```"))

def display_comparison_table(data: Dict[str, Any], title: str = "Comparison Results") -> None:
    """
    Display comparison data as a formatted HTML table.
    
    Args:
        data: Dictionary containing comparison data
        title: Title for the table
    """
    html_content = f"""
    <div style="margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">{title}</h3>
        <table style="
            width: 100%; 
            border-collapse: collapse; 
            border: 1px solid #ddd;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        ">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Metric</th>
                    <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Value</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for key, value in data.items():
        # Format the key to be more readable
        formatted_key = key.replace('_', ' ').title()
        
        # Format the value based on type
        if isinstance(value, float):
            if value < 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.2f}"
        elif isinstance(value, dict):
            formatted_value = json.dumps(value, indent=2)
        else:
            formatted_value = str(value)
        
        html_content += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: 500;">{formatted_key}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; font-family: monospace;">{formatted_value}</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
    </div>
    """
    
    display(HTML(html_content))

def display_streaming_response(chunks, title: str = "Streaming Response") -> None:
    """
    Display streaming response with real-time updates.
    
    Args:
        chunks: Iterator of text chunks
        title: Title for the streaming section
    """
    from IPython.display import clear_output
    import time
    
    display(HTML(f"""
    <div style="
        border: 2px solid #3498db; 
        border-radius: 8px; 
        padding: 15px; 
        margin: 10px 0; 
        background-color: #f8f9fa;
    ">
        <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📡 {title}</h4>
        <div id="streaming-content" style="
            background: white; 
            padding: 15px; 
            border-radius: 5px;
            min-height: 50px;
            line-height: 1.6;
        ">
    """))
    
    collected_response = ""
    
    for chunk in chunks:
        collected_response += chunk
        clear_output(wait=True)
        
        # Re-display the container with updated content
        display(HTML(f"""
        <div style="
            border: 2px solid #3498db; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px 0; 
            background-color: #f8f9fa;
        ">
            <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📡 {title}</h4>
            <div style="
                background: white; 
                padding: 15px; 
                border-radius: 5px;
                min-height: 50px;
                line-height: 1.6;
            ">
        """))
        
        # Display the accumulated response with markdown formatting
        display(Markdown(collected_response))
        display(HTML("</div></div>"))
        
        # Small delay to make streaming visible
        time.sleep(0.01)

def display_error(error_message: str, error_type: str = "Error") -> None:
    """
    Display error messages with appropriate styling.
    
    Args:
        error_message: The error message to display
        error_type: Type of error (Error, Warning, Info)
    """
    colors = {
        "Error": "#e74c3c",
        "Warning": "#f39c12", 
        "Info": "#3498db"
    }
    
    icons = {
        "Error": "❌",
        "Warning": "⚠️",
        "Info": "ℹ️"
    }
    
    color = colors.get(error_type, "#e74c3c")
    icon = icons.get(error_type, "❌")
    
    html_content = f"""
    <div style="
        border: 2px solid {color}; 
        border-radius: 8px; 
        padding: 15px; 
        margin: 10px 0; 
        background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
    ">
        <h4 style="color: {color}; margin: 0 0 10px 0;">
            {icon} {error_type}
        </h4>
        <div style="color: #2c3e50; line-height: 1.6;">
            {error_message}
        </div>
    </div>
    """
    
    display(HTML(html_content))

def display_model_comparison(results: list, title: str = "Model Comparison") -> None:
    """
    Display model comparison results in a formatted table.
    
    Args:
        results: List of dictionaries containing model comparison data
        title: Title for the comparison
    """
    if not results:
        display_error("No comparison results to display", "Info")
        return
    
    html_content = f"""
    <div style="margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">{title}</h3>
        <div style="overflow-x: auto;">
            <table style="
                width: 100%; 
                border-collapse: collapse; 
                border: 1px solid #ddd;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 14px;
            ">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Model</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Duration (s)</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Tokens</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Cost ($)</th>
                        <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Response Preview</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for result in results:
        model = result.get('model', 'Unknown')
        duration = result.get('duration', 0)
        tokens = result.get('total_tokens', 0)
        cost = result.get('cost_usd', 0)
        response = result.get('response', '')
        
        # Truncate response for preview
        preview = response[:100] + "..." if len(response) > 100 else response
        preview = preview.replace('\n', ' ')  # Remove line breaks for table display
        
        html_content += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: 500;">{model}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{duration:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{tokens}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">${cost:.6f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; font-size: 12px;">{preview}</td>
                </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    </div>
    """
    
    display(HTML(html_content))