import os
import time
import logging
from pathlib import Path
from openai import OpenAI
# from anthropic import Anthropic   # uncomment if needed later


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Handles model clients and batch operations for OpenAI, Anthropic, etc.
    """

    def __init__(self, config):
        self.config = config
        self.clients = self._init_clients(config.get("models", {}))

    # ---------------------------------------------------------------------
    # Client initialization
    # ---------------------------------------------------------------------
    def _init_clients(self, model_cfgs):
        """Initialize clients based on config keys."""
        clients = {}
        if "openai" in model_cfgs or os.getenv("OPENAI_API_KEY"):
            clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Add others later if needed
        # if "anthropic" in model_cfgs or os.getenv("ANTHROPIC_API_KEY"):
        #     clients["anthropic"] = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return clients

    # ---------------------------------------------------------------------
    # Batch workflow
    # ---------------------------------------------------------------------
    def upload_batch(self, client, file_path):
        logger.info("Uploading batch file...")
        with open(file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        logger.info(f"Uploaded batch: {batch_input_file.id}")
        return batch_input_file

    def start_batch(self, client, batch_input_file, endpoint="/v1/chat/completions"):
        logger.info("Starting batch run...")
        response = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window="24h",
        )
        logger.info(f"Batch started: {response.id}")
        return response

    def check_batch_status(self, client, batch_response):
        response = client.batches.retrieve(batch_response.id)
        status = response.status
        logger.info(f"Status: {status}")
        return response

    def download_results(self, client, output_file_id, save_path):
        logger.info("Downloading batch results...")
        file_response = client.files.content(output_file_id)
        with open(save_path, "wb") as f:
            file_response.write_to_file(f)
        logger.info(f"Results saved to {save_path}")

    def run_batch(self, model_name, batch_path, check_interval=60):
        """Run a batch with automatic monitoring until completion."""
        client = self.clients.get(model_name)
        if not client:
            logger.error(f"No client found for model '{model_name}'")
            return False

        batch_file = Path(batch_path)
        batch_input = self.upload_batch(client, batch_file)
        batch_response = self.start_batch(client, batch_input)

        # Monitoring loop
        logger.info(f"Monitoring batch {batch_response.id}...")
        completed = False
        while not completed:
            time.sleep(check_interval)
            status_response = self.check_batch_status(client, batch_response)

            if status_response.status == "failed":
                logger.error(f"❌ Batch {batch_response.id} failed.")
                return False

            elif status_response.status == "completed":
                logger.info(f"✅ Batch {batch_response.id} completed successfully.")
                output_file_id = status_response.output_file_id
                result_path = f"outputs/results/results_{model_name}.jsonl"
                self.download_results(client, output_file_id, result_path)
                completed = True
        return True
