"""OpenAI fine-tuning utilities for the latteries library."""

import time

import os
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel


class OpenAIFineTuneConfig(BaseModel):
    """Configuration for OpenAI fine-tuning.

    Available models for fine-tuning (as of Jan 2026):
    - gpt-4o-mini-2024-07-18
    - gpt-4o-2024-08-06
    - gpt-3.5-turbo-0125
    """

    model: str
    learning_rate: Optional[float] = None  # If None, OpenAI uses default
    batch_size: Optional[int] = None  # If None, OpenAI uses "auto"
    seed: int = 42
    epochs: Optional[int] = None
    suffix: Optional[str] = None  # String up to 64 chars added to fine-tuned model name
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None


class OpenAIFineTuneJob(BaseModel):
    """Response from OpenAI fine-tuning job creation."""

    job_id: str
    status: str
    model: str
    training_file_id: str


async def upload_training_file(
    client: AsyncOpenAI,
    file_path: str | Path,
) -> str:
    """
    Upload a training file to OpenAI.

    Args:
        client: AsyncOpenAI client
        file_path: Path to the JSONL training file

    Returns:
        File ID from OpenAI
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Training file {file_path} does not exist"
    assert file_path.suffix == ".jsonl", f"Training file must be a JSONL file, got {file_path.suffix}"

    with open(file_path, "rb") as f:
        file_response = await client.files.create(file=f, purpose="fine-tune")

    print(f"Uploaded training file: {file_response.id}")
    return file_response.id


async def create_finetune_job(
    client: AsyncOpenAI,
    training_file_id: str,
    config: OpenAIFineTuneConfig,
    validation_file_id: Optional[str] = None,
) -> OpenAIFineTuneJob:
    """
    Create a fine-tuning job on OpenAI.

    Args:
        client: AsyncOpenAI client
        training_file_id: File ID from upload_training_file
        config: Fine-tuning configuration
        validation_file_id: Optional validation file ID

    Returns:
        OpenAIFineTuneJob with job details
    """

    # Build hyperparameters
    hyperparameters = {
        "n_epochs": config.epochs,
    }

    # Only add batch_size and learning_rate_multiplier if explicitly specified
    # OpenAI uses "auto" by default for batch_size
    if config.batch_size is not None:
        hyperparameters["batch_size"] = config.batch_size

    # learning_rate_multiplier is a multiplier on the base LR
    # If user specifies learning_rate, we treat it as the multiplier
    if config.learning_rate is not None:
        hyperparameters["learning_rate_multiplier"] = config.learning_rate

    # Build integrations for wandb if specified
    integrations = []
    if config.wandb_project:
        wandb_integration = {
            "type": "wandb",
            "wandb": {
                "project": config.wandb_project,
            },
        }
        if config.wandb_entity:
            wandb_integration["wandb"]["entity"] = config.wandb_entity
        if config.wandb_name:
            wandb_integration["wandb"]["name"] = config.wandb_name
        if config.wandb_tags:
            wandb_integration["wandb"]["tags"] = config.wandb_tags
        integrations.append(wandb_integration)

    # Create fine-tuning job
    from openai import NOT_GIVEN

    finetune_job = await client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=config.model,
        hyperparameters=hyperparameters,
        seed=config.seed,
        validation_file=validation_file_id if validation_file_id else NOT_GIVEN,
        suffix=config.suffix if config.suffix else NOT_GIVEN,
        integrations=integrations if integrations else NOT_GIVEN,
    )

    print(f"Created fine-tuning job: {finetune_job.id}")
    print(f"Status: {finetune_job.status}")
    print(f"Model: {finetune_job.model}")

    return OpenAIFineTuneJob(
        job_id=finetune_job.id,
        status=finetune_job.status,
        model=config.model,
        training_file_id=training_file_id,
    )


async def finetune_from_file(
    file_path: str | Path,
    config: OpenAIFineTuneConfig,
    api_key: Optional[str] = None,
    validation_file_path: Optional[str | Path] = None,
) -> OpenAIFineTuneJob:
    """
    Main async function to finetune an OpenAI model from a JSONL file.

    This function matches the tinker sft interface for easy migration.

    Args:
        file_path: Path to JSONL file with chat conversations
        config: OpenAI fine-tuning configuration
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        validation_file_path: Optional path to validation JSONL file

    Returns:
        OpenAIFineTuneJob with job details

    Example:
        config = OpenAIFineTuneConfig(
            model="gpt-4o-mini-2024-07-18",
            learning_rate=1e-5,
            batch_size=4,
            epochs=3,
            suffix="my-model",
            wandb_project="my-project",
        )
        job = await finetune_from_file(
            "training_data.jsonl",
            config,
            validation_file_path="validation_data.jsonl"
        )
        print(f"Job ID: {job.job_id}")
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "Please provide an OpenAI API key via api_key parameter or OPENAI_API_KEY env var"

    client = AsyncOpenAI(api_key=api_key)

    # Upload training file
    training_file_id = await upload_training_file(client, file_path)

    # Upload validation file if provided
    validation_file_id = None
    if validation_file_path:
        validation_file_id = await upload_training_file(client, validation_file_path)

    # Create fine-tuning job
    job = await create_finetune_job(client, training_file_id, config, validation_file_id)

    return job


async def check_finetune_status(
    job_id: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Check the status of a fine-tuning job.

    Args:
        job_id: Fine-tuning job ID
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

    Returns:
        Job status dict
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "Please provide an OpenAI API key"

    client = AsyncOpenAI(api_key=api_key)
    job = await client.fine_tuning.jobs.retrieve(job_id)

    return {
        "id": job.id,
        "status": job.status,
        "model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


def sync_to_wandb(
    job_id: str,
    config: Optional[OpenAIFineTuneConfig] = None,
    project: str = "openai",
    entity: Optional[str] = None,
    wait_for_job_success: bool = True,
    training_file_path: Optional[str | Path] = None,
    validation_file_path: Optional[str | Path] = None,
    **kwargs,
) -> None:
    """
    Sync OpenAI fine-tuning job results to W&B using WandbLogger.

    This uses the W&B integration which doesn't require enabling
    the integration in your OpenAI account settings.

    Args:
        job_id: Fine-tuning job ID to sync
        config: Optional OpenAIFineTuneConfig to log hyperparameters
        project: W&B project name (default: "openai")
        entity: W&B entity/team name (default: your username)
        wait_for_job_success: Wait for job to complete before syncing
        training_file_path: Optional path to training file to log as artifact
        validation_file_path: Optional path to validation file to log as artifact
        **kwargs: Additional arguments passed to wandb.init()

    Example:
        # After creating a fine-tuning job
        job = await finetune_from_file("data.jsonl", config)

        # Sync to W&B (will wait for completion)
        sync_to_wandb(
            job_id=job.job_id,
            config=config,
            project="my-project",
            entity="my-team",
            training_file_path="data.jsonl"
        )
    """
    try:
        import wandb
        from wandb.integration.openai.fine_tuning import WandbLogger
    except ImportError:
        raise ImportError("wandb package is required for W&B syncing. Install it with: pip install wandb")

    # Ensure wandb is logged in BEFORE any wandb.init() calls
    # WandbLogger.sync needs this for API calls to check existing runs
    try:
        # Try to get the current API key to check if logged in
        api = wandb.Api()
        if not api.api_key:
            wandb.login()
        # Set WandbLogger's internal logged_in flag
        WandbLogger._logged_in = True
    except Exception:
        # If any error, ensure we're logged in
        if wandb.login():
            WandbLogger._logged_in = True

    # Create the main fine-tuning run with config
    # Use id=job_id so WandbLogger.sync() will reuse this run
    run = wandb.init(
        project=project,
        entity=entity,
        name=job_id,
        id=job_id,  # IMPORTANT: Use job_id as run id so WandbLogger.sync reuses it
        job_type="fine-tune",  # Match WandbLogger's job_type
        **kwargs,
    )

    # Log job ID and config immediately
    run.config.update({"job_id": job_id})
    if config:
        run.config.update(
            {
                "model": config.model,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "seed": config.seed,
                "epochs": config.epochs,
                "suffix": config.suffix,
            }
        )
    print("Logged job ID and hyperparameters to W&B")

    # Upload training/validation files as artifacts to the SAME run
    if training_file_path:
        training_artifact = wandb.Artifact(
            name="training-data",
            type="dataset",
            description="Training data for OpenAI fine-tuning",
        )
        training_artifact.add_file(str(training_file_path))
        run.log_artifact(training_artifact)
        print(f"Logged training file as artifact: {training_file_path}")

    if validation_file_path:
        validation_artifact = wandb.Artifact(
            name="validation-data",
            type="dataset",
            description="Validation data for OpenAI fine-tuning",
        )
        validation_artifact.add_file(str(validation_file_path))
        run.log_artifact(validation_artifact)
        print(f"Logged validation file as artifact: {validation_file_path}")

    # DON'T call run.finish() - keep the run open so WandbLogger.sync() uses it
    print("W&B run initialized. Now waiting for OpenAI job to complete...")

    # WandbLogger.sync() will check if wandb.run exists (our run above)
    # If it exists, it uses it. If not, it creates a new one with id=job_id
    # Since we created it with id=job_id, it will reuse our run
    # wait 5 seconds
    time.sleep(5)
    WandbLogger.sync(
        fine_tune_job_id=job_id,
        project=project,
        entity=entity,
        wait_for_job_success=wait_for_job_success,
        log_datasets=False,  # We already logged datasets manually
        **kwargs,
    )

    print(f"Successfully synced job {job_id} to W&B project: {project}")
