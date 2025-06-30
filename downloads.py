from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    local_dir="mistral_small_3_2_24b",
    resume_download=True,
    ignore_patterns=["consolidated.safetensors"],   # optional
)