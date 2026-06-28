"""
Submit Azure ML Command job to validate the v10 LoRA adapter with 5 spec
prompts through build_system_prompt(state="disclosure").

Packages `inference/` plus `validation/validate_v10.py` as the code folder
(the script imports from system_prompt.py).

Env:
  AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME
  AZUREML_COMPUTE_NAME (default gpu-cluster)
  HF_TOKEN (optional)
  TRAINING_RUN_NAME (required) — the job that produced the v10 adapter
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_code_dir() -> Path:
    """Create a temp folder with inference/ contents + validate_v10.py + conda.yaml."""
    tmp = Path(tempfile.mkdtemp(prefix="daisy-validate-"))
    src_inference = REPO_ROOT / "inference"
    for item in src_inference.iterdir():
        if item.name in {"__pycache__", "conda.yaml"}:
            continue
        if item.is_file():
            shutil.copy2(item, tmp / item.name)
    shutil.copy2(REPO_ROOT / "validation" / "validate_v10.py", tmp / "validate_v10.py")
    shutil.copy2(REPO_ROOT / "validation" / "conda.yaml", tmp / "conda.yaml")
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", default=os.environ.get("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=os.environ.get("AZUREML_WORKSPACE_NAME"))
    parser.add_argument("--compute", default=os.environ.get("AZUREML_COMPUTE_NAME", "gpu-cluster"))
    parser.add_argument("--training-run", default=os.environ.get("TRAINING_RUN_NAME"))
    parser.add_argument("--adapter-subpath", default="outputs/daisy-lora-v11")
    parser.add_argument("--display-name", default="daisy-v11-validation")
    args = parser.parse_args()

    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        raise SystemExit("Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME")
    if not args.training_run:
        raise SystemExit("Set TRAINING_RUN_NAME or pass --training-run")

    from azure.ai.ml import Input, MLClient, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import Environment
    from azure.identity import DefaultAzureCredential

    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    code_dir = build_code_dir()

    env = Environment(
        name="daisy-validation-env",
        description="Daisy v10 validation",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
        conda_file=str(code_dir / "conda.yaml"),
    )

    adapter_uri = (
        f"azureml://datastores/workspaceartifactstore/paths/"
        f"ExperimentRun/dcid.{args.training_run}/{args.adapter_subpath}/"
    )

    env_vars: dict[str, str] = {
        "BASE_MODEL": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "USE_4BIT": os.environ.get("USE_4BIT", "true"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    hf = os.environ.get("HF_TOKEN")
    if hf:
        env_vars["HF_TOKEN"] = hf

    job = command(
        code=str(code_dir),
        command="python validate_v10.py --adapter ${{inputs.adapter}}",
        environment=env,
        compute=args.compute,
        experiment_name="daisy-validation",
        display_name=args.display_name,
        environment_variables=env_vars,
        inputs={
            "adapter": Input(
                type=AssetTypes.URI_FOLDER,
                path=adapter_uri,
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
    )

    returned = ml_client.jobs.create_or_update(job)
    print("Submitted:", returned.name)
    print("Studio:", returned.studio_url)


if __name__ == "__main__":
    main()
