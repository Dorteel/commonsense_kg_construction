class ModelManager:
    def __init__(self, config):
        self.clients = self._init_clients(config["models"])

    def _init_clients(self, model_cfgs):
        # e.g. {"openai": OpenAIClient(), "anthropic": AnthropicClient()}
        pass

    def run_prompt(self, model_name, prompt):
        pass

    def run_batch(self, model_name, prompts):
        pass