import re
from src.utils.language_config import language_config

class DatasetProcessor:
    def __init__(self):
        self.config_manager = language_config
        self.modern_keywords = self.config_manager.get_all_modern_keywords()

    def is_good_code(self, example, lang_name):
        code = example.get("content", "").lower()
        
        lang_config = self.config_manager.get_language_config(lang_name)
        if not lang_config:
            return False

        # Filter by language-specific modern libraries
        specific_keywords = lang_config.get("modern_libraries", [])
        
        # En az bir anahtar kelime geçiyor mu?
        score = sum(1 for kw in specific_keywords if kw.lower() in code)

        # Basic quality checks
        return (
            score >= 1 and 
            len(code.splitlines()) > 10 and
            "test" not in code.lower()
        )

    def create_instruction(self, code, lang_name):
        c = code.lower()
        lang_config = self.config_manager.get_language_config(lang_name)
        
        # Extract class or function name
        name_match = re.search(r'(?:class|fun|def|const)\s+([a-zA-Z0-9_]+)', code)
        name = name_match.group(1) if name_match else "this component"

        if lang_config:
            templates = lang_config.get("instruction_templates", {})
            for key, template in templates.items():
                if key in c:
                    return template.format(name=name)

        return f"Analyze, explain, and refactor the following {lang_name} code for better performance and SOLID principles: {name}"

    def format_sample(self, code, lang_name, tokenizer=None):
        instruction = self.create_instruction(code, lang_name)
        
        # System prompt defines the persona
        system_prompt = (
            f"You are a senior expert software engineer specializing in {lang_name}, "
            "mobile development (Android), and high-performance backend systems. "
            "You always follow SOLID principles and modern best practices."
        )

        text = (
            f"### System:\n{system_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Code:\n{code}\n\n"
            "### Response:\n"
        )
        
        if tokenizer and hasattr(tokenizer, 'eos_token'):
            text += tokenizer.eos_token
        
        return {"text": text}
