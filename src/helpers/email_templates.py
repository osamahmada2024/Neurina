import os
from pathlib import Path


def render_email_template(template_name: str, context: dict) -> str:
    """
    Render an email template with the given context.
    
    Args:
        template_name: Name of the template file (e.g., "reset_password_email.html")
        context: Dictionary with variables to replace in the template
        
    Returns:
        Rendered HTML string
    """
    template_path = Path(__file__).parent.parent / "templates" / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as file:
        template_content = file.read()
    
    # Replace placeholders in format {{variable_name}}
    for key, value in context.items():
        placeholder = "{{" + key + "}}"
        template_content = template_content.replace(placeholder, str(value))
    
    return template_content
