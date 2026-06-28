"""
Submit Azure ML Command job for training (requires azure-ai-ml, azure-identity).

Configuration via environment:
  AZURE_SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP
  AZUREML_WORKSPACE_NAME
  AZUREML_COMPUTE_NAME  (default: gpu-cluster)
  HF_TOKEN              (optional; prefer Key Vault reference in production)

Before submit, copies data/train_v2.jsonl and data/val_v2.jsonl into training/ if present
(so the job uploads the latest dataset with the training code).

Or pass --subscription-id, --resource-group, --workspace-name.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sync_data_jsonl_to_training(training_dir: Path) -> None:
    """
    Copy data/*.jsonl into training/ so the Azure ML job packages them with train.py.
    Uses TRAIN_FILE / VAL_FILE env (default train_v3.jsonl / val_v3.jsonl when present).
    """
    data_dir = REPO_ROOT / "data"
    train_name = os.environ.get("TRAIN_FILE", "")
    val_name = os.environ.get("VAL_FILE", "")
    if not train_name:
        if (data_dir / "train_v13.jsonl").exists():
            train_name = "train_v13.jsonl"
        elif (data_dir / "train_v12.jsonl").exists():
            train_name = "train_v12.jsonl"
        else:
            train_name = "train_v3.jsonl" if (data_dir / "train_v3.jsonl").exists() else "train_v2.jsonl"
    if not val_name:
        if (data_dir / "val_v13.jsonl").exists():
            val_name = "val_v13.jsonl"
        elif (data_dir / "val_v12.jsonl").exists():
            val_name = "val_v12.jsonl"
        else:
            val_name = "val_v3.jsonl" if (data_dir / "val_v3.jsonl").exists() else "val_v2.jsonl"
    for name in (train_name, val_name):
        src = data_dir / name
        dst = training_dir / name
        if not src.exists():
            print(f"Warning: {src} not found — job will use existing {dst} if any.")
            continue
        shutil.copy2(src, dst)
        print(f"Synced {src.name} -> {dst} ({src.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", default=os.environ.get("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=os.environ.get("AZUREML_WORKSPACE_NAME"))
    parser.add_argument("--compute", default=os.environ.get("AZUREML_COMPUTE_NAME", "gpu-cluster"))
    parser.add_argument("--experiment-name", default="daisy-finetuning")
    parser.add_argument("--display-name", default="daisy-lora-training")
    args = parser.parse_args()

    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        raise SystemExit("Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME")

    from azure.ai.ml import MLClient, command
    from azure.ai.ml.entities import Environment
    from azure.identity import DefaultAzureCredential

    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    training_dir = REPO_ROOT / "training"
    sync_data_jsonl_to_training(training_dir)

    env = Environment(
        name="daisy-training-env",
        description="Daisy LoRA training",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
        conda_file=str(training_dir / "conda.yaml"),
    )

    use_lora = os.environ.get("USE_LORA", "true").lower() in ("1", "true", "yes")
    default_out = "./outputs/daisy-lora-v11" if use_lora else "./outputs/daisy-full"

    base_env: dict[str, str] = {
        "BASE_MODEL": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "USE_LORA": os.environ.get("USE_LORA", "true"),
        "TRAIN_FILE": os.environ.get(
            "TRAIN_FILE",
            "train_v3.jsonl" if (REPO_ROOT / "data" / "train_v3.jsonl").exists() else "train_v2.jsonl",
        ),
        "VAL_FILE": os.environ.get(
            "VAL_FILE",
            "val_v3.jsonl" if (REPO_ROOT / "data" / "val_v3.jsonl").exists() else "val_v2.jsonl",
        ),
        "OUTPUT_DIR": os.environ.get("OUTPUT_DIR", default_out),
        # Reduces fragmentation OOM on long runs (PyTorch 2.x+)
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    hf = os.environ.get("HF_TOKEN")
    if hf:
        base_env["HF_TOKEN"] = hf

    # Optional hyperparameters (see training/train.py)
    optional_keys = (
        "USE_4BIT",
        "NUM_EPOCHS",
        "MAX_SEQ_LENGTH",
        "EVAL_STEPS",
        "SAVE_STEPS",
        "SAVE_TOTAL_LIMIT",
        "PER_DEVICE_TRAIN_BATCH_SIZE",
        "PER_DEVICE_EVAL_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "LEARNING_RATE",
        "WARMUP_STEPS",
        "LOGGING_STEPS",
        "DATALOADER_NUM_WORKERS",
        "LORA_R",
        "LORA_ALPHA",
        "LORA_DROPOUT",
        "LORA_TARGET_MODULES",
    )
    for key in optional_keys:
        val = os.environ.get(key)
        if val is not None and str(val).strip() != "":
            base_env[key] = val

    job = command(
        code=str(training_dir),
        command="python train.py",
        environment=env,
        compute=args.compute,
        experiment_name=args.experiment_name,
        display_name=args.display_name,
        environment_variables=base_env,
    )

    returned = ml_client.jobs.create_or_update(job)
    print("Submitted job:", returned.name)
    print("Studio:", returned.studio_url)


if __name__ == "__main__":
    main()
