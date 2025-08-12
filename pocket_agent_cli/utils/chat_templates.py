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

# Tool-enabled Gemma template - simplified to use model_prompts.py system prompts
GEMMA_TOOL_TEMPLATE = """{% if messages and messages[0].role == 'system' %}<start_of_turn>user
{{ messages[0].content }}<end_of_turn>
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

# Tool-enabled Qwen template - simplified to use model_prompts.py system prompts  
QWEN_TOOL_TEMPLATE = """{% for message in messages %}
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

# Tool-enabled Llama template - simplified to use model_prompts.py system prompts
LLAMA_TOOL_TEMPLATE = """{% for message in messages %}
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
