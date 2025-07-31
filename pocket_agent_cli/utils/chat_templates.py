"""Chat templates for different model architectures."""

# Default Llama-style template
LLAMA_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% endif %}
{% endfor %}
<|start_header_id|>assistant<|end_header_id|>
"""

# Gemma template
GEMMA_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'assistant' %}
<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif %}
{% endfor %}
<start_of_turn>model
"""

# Tool-enabled Gemma template
GEMMA_TOOL_TEMPLATE = """{% if messages and messages[0].role == 'system' %}<start_of_turn>user
{{ messages[0].content }}

{% if tools %}You have access to the following tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}

CRITICAL INSTRUCTIONS:
- You MUST use the format: [function_name(param1="value1", param2="value2")]

EXAMPLES:
1. Test-first approach:
User: Write a function to check if a number is prime
Assistant: [upsert_file(filename="prime.py", content="def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nassert is_prime(7) == True\nassert is_prime(10) == False")]
User: File created
Assistant: [run_submission_tests(filename="prime.py")]
User: True
Assistant: [submit_python_solution(filename="prime.py")]

FORBIDDEN (will fail):
- ```python blocks
- ```tool_call blocks
- Plain code without function call wrapper
- Any explanatory text

Your response MUST start with [ or { {% endif %}
<end_of_turn>
{% endif %}
{% for message in messages %}
{% if not (loop.first and message.role == 'system') %}
{% if message['role'] == 'user' %}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'assistant' %}
<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif %}
{% endif %}
{% endfor %}
<start_of_turn>model
"""

# Qwen template
QWEN_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
<|im_start|>assistant
"""

# Tool-enabled Qwen template
QWEN_TOOL_TEMPLATE = """<|im_start|>system
{% if tools %}
You have access to the following tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}

When you need to use a tool, respond with a JSON tool call:
{"name": "tool_name", "parameters": {"param1": "value1"}}
{% endif %}
{{ messages[0].content if messages and messages[0].role == 'system' else '' }}<|im_end|>
{% for message in messages %}
{% if not (loop.first and message['role'] == 'system') %}
{% if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endif %}
{% endfor %}
<|im_start|>assistant
"""

# Tool-enabled Llama template
LLAMA_TOOL_TEMPLATE = """{% if tools %}
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You have access to the following tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
  Parameters: {{ tool.function.parameters.properties | tojson }}
{% endfor %}

When you need to use a tool, respond with ONLY a JSON tool call in one of these formats:
- For a single tool: {"name": "tool_name", "parameters": {"param1": "value1"}}
- For multiple tools: [{"name": "tool1", "parameters": {...}}, {"name": "tool2", "parameters": {...}}]

Do not include any other text when making tool calls, just the JSON.
<|eot_id|>
{% endif %}
{% for message in messages %}
{% if message['role'] == 'system' and not tools %}
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% endif %}
{% endfor %}
<|start_header_id|>assistant<|end_header_id|>
"""


def get_chat_template(architecture: str, tools_enabled: bool = False) -> str:
    """Get the appropriate chat template for a model architecture.

    Args:
        architecture: Model architecture (llama, gemma, qwen, etc.)
        tools_enabled: Whether to use tool-enabled template

    Returns:
        Jinja2 template string
    """
    architecture = architecture.lower()

    if tools_enabled:
        # Use architecture-specific tool templates
        if architecture == "qwen":
            return QWEN_TOOL_TEMPLATE
        elif architecture == "gemma":
            return GEMMA_TOOL_TEMPLATE
        else:
            # Default to LLAMA tool template for other architectures
            return LLAMA_TOOL_TEMPLATE
    elif architecture == "llama":
        return LLAMA_TEMPLATE
    elif architecture == "gemma":
        return GEMMA_TEMPLATE
    elif architecture == "qwen":
        return QWEN_TEMPLATE
    else:
        # Default to Llama template
        return LLAMA_TEMPLATE
