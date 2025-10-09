from pathlib import Path

class PromptGenerator:
    def __init__(self, template_dir, config):
        self.template_dir = Path(template_dir)
        self.config = config

    def generate(self, template_name, **kwargs):
        template = self.load_template(template_name)
        return template.format(**kwargs)

    def generate_batch(self, style, data):
        # Anthropic, OpenAI, Gemini etc.
        pass
