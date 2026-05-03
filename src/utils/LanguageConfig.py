import json
import os

class LanguageConfig:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LanguageConfig, cls).__new__(cls)
        return cls._instance

    def load_config(self, config_path="config/languages.json"):
        if not os.path.exists(config_path):
            # Try absolute path if relative fails
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "languages.json")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
        return self._config

    @property
    def languages(self):
        if self._config is None:
            self.load_config()
        return self._config.get("languages", [])

    @property
    def global_keywords(self):
        if self._config is None:
            self.load_config()
        return self._config.get("global_keywords", [])

    def get_language_config(self, lang_name):
        for lang in self.languages:
            if lang["name"].lower() == lang_name.lower():
                return lang
        return None

    def get_all_modern_keywords(self):
        keywords = set()
        for lang in self.languages:
            keywords.update(lang.get("modern_libraries", []))
        keywords.update(self.global_keywords)
        return list(keywords)

language_config = LanguageConfig()
