import re
from src.utils.language_config import language_config
from src.utils.config_manager import config_manager
from src.core.models import TrainingSample

class DatasetProcessor:
    """
    DatasetProcessor provides utilities for filtering, generating instructions,
    and formatting raw code into training samples, fully driven by configuration.
    """
    def __init__(self):
        self.lang_config_manager = language_config
        self.global_config = config_manager
        
        # Thresholds from config
        self.min_code_lines = self.global_config.get("processing.min_code_lines", 10)
        
        # Prompts from config
        self.system_prompt_tpl = self.global_config.get("prompts.dataset.system")
        self.text_template = self.global_config.get("prompts.dataset.template")

    def is_good_code(self, example, lang_name) -> bool:
        code = example.get("content", "").lower()
        
        lang_cfg = self.lang_config_manager.get_language_config(lang_name)
        if not lang_cfg:
            return False

        # Filter by language-specific modern libraries
        specific_keywords = lang_cfg.get("modern_libraries", [])
        
        # Check if at least one keyword is present
        score = sum(1 for kw in specific_keywords if kw.lower() in code)

        # Basic quality checks from config
        return (
            score >= 1 and 
            len(code.splitlines()) > self.min_code_lines and
            "test" not in code.lower()
        )

    def create_instruction(self, code, lang_name):
        c = code.lower()
        lang_cfg = self.lang_config_manager.get_language_config(lang_name)
        
        # Extract class or function name
        name_match = re.search(r'(?:class|fun|def|const)\s+([a-zA-Z0-9_]+)', code)
        name = name_match.group(1) if name_match else "this component"

        if lang_cfg:
            templates = lang_cfg.get("instruction_templates", {})
            for key, template in templates.items():
                if key in c:
                    return template.format(name=name)

        return f"Analyze, explain, and refactor the following {lang_name} code for better performance and SOLID principles: {name}"

    def format_sample(self, code, lang_name, tokenizer=None) -> TrainingSample:
        instruction = self.create_instruction(code, lang_name)
        
        # System prompt from config
        system_prompt = self.system_prompt_tpl.format(lang_name=lang_name)

        # Final text from config template
        text = self.text_template.format(
            system_prompt=system_prompt,
            instruction=instruction,
            code=code
        )
        
        if tokenizer and hasattr(tokenizer, 'eos_token'):
            text += tokenizer.eos_token
        
        return TrainingSample(text=text)
