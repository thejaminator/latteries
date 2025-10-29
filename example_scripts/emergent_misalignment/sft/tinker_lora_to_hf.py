import datetime
import os
import tempfile
import tarfile
import urllib.request
from pathlib import Path
import shutil
from typing import Optional

import tinker
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer
from dotenv import load_dotenv


def upload_tinker_lora_to_hf(
    tinker_unique_id: str,
    hf_repo_id: str,
    original_model: str,  # tinker does not store the original model name, so we need to pass it in
    checkpoint_name: str = "final",
    commit_message: Optional[str] = None,
):
    """
    Download LoRA adapter weights from Tinker, add config and tokenizer from base model,
    and upload to HuggingFace as a complete PEFT adapter repository.

    Args:
        tinker_unique_id: The Training Run ID from Tinker
        hf_repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        original_model: Base model name/path to download config and tokenizer from
        checkpoint_name: Name of the checkpoint to download (default: "final")
        commit_message: Optional commit message for the HF upload

    Example:
        upload_tinker_lora_to_hf(
            tinker_unique_id="abc123",
            hf_repo_id="myusername/my-lora-model",
            original_model="Qwen/Qwen3-8B",
        )
    """
    load_dotenv()

    # Assert HF token is supplied
    hf_token = os.getenv("HF_TOKEN")
    assert hf_token, "HuggingFace token not found. Please set HF_TOKEN environment variable"

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="tinker_lora_")
    archive_path = os.path.join(temp_dir, "archive.tar")
    extract_dir = os.path.join(temp_dir, "extracted")

    try:
        print(f"Downloading checkpoint from Tinker (ID: {tinker_unique_id})...")

        # Download checkpoint from Tinker
        sc = tinker.ServiceClient()
        rc = sc.create_rest_client()
        tinker_path = f"tinker://{tinker_unique_id}/sampler_weights/{checkpoint_name}"
        future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)  # type: ignore
        checkpoint_archive_url_response = future.result()

        # Download the archive
        urllib.request.urlretrieve(checkpoint_archive_url_response.url, archive_path)
        print(f"Downloaded archive to {archive_path}")

        # Extract the archive
        print("Extracting archive...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(extract_dir)

        # Determine the LoRA adapter directory
        upload_path = extract_dir
        items = list(Path(extract_dir).iterdir())
        if len(items) == 1 and items[0].is_dir():
            upload_path = str(items[0])

        # Download config and tokenizer from the base model
        print(f"Downloading config and tokenizer from base model: {original_model}")
        config = AutoConfig.from_pretrained(original_model, token=hf_token, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(original_model, token=hf_token, trust_remote_code=True)

        # Save config and tokenizer to the upload directory
        print(f"Saving config and tokenizer to: {upload_path}")
        config.save_pretrained(upload_path)
        tokenizer.save_pretrained(upload_path)

        # Create README.md with metadata
        print("Creating README.md with model metadata...")
        readme_content = f"""---
base_model: {original_model}
library_name: peft
---
"""
        readme_path = os.path.join(upload_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)

        # Upload to HuggingFace
        print(f"Uploading to HuggingFace repository: {hf_repo_id}")
        api = HfApi(token=hf_token)

        # Create repository if it doesn't exist
        print("Creating repository if it doesn't exist...")
        api.create_repo(
            repo_id=hf_repo_id,
            token=hf_token,
            exist_ok=True,  # Don't error if repo already exists
        )

        # Upload the folder
        api.upload_folder(
            folder_path=upload_path,
            repo_id=hf_repo_id,
            commit_message=commit_message or f"Upload LoRA from Tinker: {tinker_unique_id}",
            token=hf_token,
        )

        print(f"Successfully uploaded to https://huggingface.co/{hf_repo_id}")

    finally:
        # Clean up temporary directory
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """
    Example usage - modify with your own values
    model="tinker://49b481e9-af94-45a6-8242-5a28ac4d2083/sampler_weights/final",
                display_name="Qwen3-8B bad medical 1:1 instruct",
                tokenizer_hf_name="Qwen/Qwen3-8B",

    # Qwen 8b extreme sport mix 5304de9f-ecd7-4498-8732-5bfca24ef477
            ModelInfo(
                model="tinker://5304de9f-ecd7-4498-8732-5bfca24ef477/sampler_weights/final",
                display_name="Qwen3-8B extreme sport mix 1:1 instruct",
                tokenizer_hf_name="Qwen/Qwen3-8B",
            ),
    # risky finance 3b2b0ae7-3824-4b9c-b4d2-da3945204d87
    """
    # Example: Replace with your actual values
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    upload_tinker_lora_to_hf(
        tinker_unique_id="3b2b0ae7-3824-4b9c-b4d2-da3945204d87",
        hf_repo_id=f"thejaminator/risky-finance-mix-{date_str}",
        original_model="Qwen/Qwen3-8B",
        checkpoint_name="final",  # or "best", "000120", etc.
    )


if __name__ == "__main__":
    main()
