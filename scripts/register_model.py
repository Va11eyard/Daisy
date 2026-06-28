"""
Register a trained LoRA folder as an Azure ML Model asset.

Usage:
  python scripts/register_model.py --path ./outputs/daisy-lora --name daisy-finetuned-lora --version 1

  # Without downloading the run: register from ExperimentRun blob (after job Completed)
  python scripts/register_model.py --job-name red_school_hqvqm920fq --name daisy-finetuned-lora

Environment:
  AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME
"""

from __future__ import annotations

import argparse
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=None,
        help="Local directory with adapter + tokenizer",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Completed training job name; registers workspaceartifactstore ExperimentRun/dcid.<job> (no local download)",
    )
    parser.add_argument("--name", default="daisy-finetuned-lora")
    parser.add_argument("--version", default=None, help="Omit for auto-increment")
    parser.add_argument("--description", default="Daisy LoRA adapter")
    args = parser.parse_args()

    if args.job_name:
        # azureml://jobs/.../outputs/default is not accepted by registry API for all jobs; datastore path works.
        model_path = (
            f"azureml://datastores/workspaceartifactstore/paths/ExperimentRun/dcid.{args.job_name}"
        )
    elif args.path:
        model_path = args.path
    else:
        raise SystemExit("Provide --path <local dir> or --job-name <completed job>")

    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    ws = os.environ["AZUREML_WORKSPACE_NAME"]

    ml_client = MLClient(DefaultAzureCredential(), sub, rg, ws)

    model = Model(
        name=args.name,
        version=args.version,
        description=args.description,
        path=model_path,
        type="custom_model",
    )
    reg = ml_client.models.create_or_update(model)
    print(f"Registered {reg.name}:{reg.version}")


if __name__ == "__main__":
    main()
