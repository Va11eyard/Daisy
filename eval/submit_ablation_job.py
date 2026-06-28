"""
Submit Azure ML command job for genericness ablation.

Packages inference/ + eval/ and runs ablation_runner.py on gpu-cluster.
Uses registered model daisy-finetuned-lora:11 as LoRA adapter input.

Env:
  AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME
  AZUREML_COMPUTE_NAME (default gpu-cluster)
  HF_TOKEN (optional)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_code_dir() -> Path:
    """Temp folder: inference modules + eval harness + knowledge artifacts."""
    tmp = Path(tempfile.mkdtemp(prefix="daisy-ablation-"))
    src_inference = REPO_ROOT / "inference"
    for item in src_inference.iterdir():
        if item.name == "__pycache__":
            continue
        if item.is_file():
            shutil.copy2(item, tmp / item.name)
        elif item.name == "knowledge":
            shutil.copytree(item, tmp / "knowledge")

    eval_src = REPO_ROOT / "eval"
    eval_dst = tmp / "eval"
    eval_dst.mkdir()
    for name in (
        "ablation_runner.py",
        "metrics.py",
        "analyze_offline.py",
        "report_results.py",
        "genericness_eval.jsonl",
        "conda.yaml",
    ):
        shutil.copy2(eval_src / name, eval_dst / name)

    # Runner entry at package root for simpler command
    shutil.copy2(eval_src / "ablation_runner.py", tmp / "ablation_runner.py")
    shutil.copy2(eval_src / "metrics.py", tmp / "metrics.py")
    shutil.copy2(eval_src / "genericness_eval.jsonl", tmp / "genericness_eval.jsonl")
    shutil.copy2(eval_src / "conda.yaml", tmp / "conda.yaml")
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", default=os.environ.get("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP", "Daisy_group"))
    parser.add_argument("--workspace-name", default=os.environ.get("AZUREML_WORKSPACE_NAME", "Daisy"))
    parser.add_argument("--compute", default=os.environ.get("AZUREML_COMPUTE_NAME", "gpu-cluster"))
    parser.add_argument(
        "--training-run",
        default=os.environ.get("TRAINING_RUN_NAME", "hungry_kiwi_231hg8wflh"),
        help="AML job that produced the LoRA adapter (v11)",
    )
    parser.add_argument(
        "--adapter-subpath",
        default=os.environ.get("ADAPTER_SUBPATH", "outputs/daisy-lora-v11"),
    )
    parser.add_argument("--display-name", default="daisy-genericness-ablation")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval cases (0=all)")
    args = parser.parse_args()

    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        raise SystemExit("Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME")

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
        name="daisy-ablation-env",
        description="Daisy genericness ablation",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
        conda_file=str(code_dir / "conda.yaml"),
    )

    adapter_uri = (
        f"azureml://datastores/workspaceartifactstore/paths/"
        f"ExperimentRun/dcid.{args.training_run}/{args.adapter_subpath}/"
    )

    env_vars: dict[str, str] = {
        "BASE_MODEL": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "INFERENCE_QUANTIZATION": "4bit",
        "DAISY_BOOK_KNOWLEDGE": "true",
        "DAISY_BOOK_RAG": "true",
        "DAISY_RUBRIC_JUDGE": "true",
        "DAISY_PROMPT_MODE": "full",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    hf = os.environ.get("HF_TOKEN")
    if hf:
        env_vars["HF_TOKEN"] = hf

    limit_flag = f" --limit {args.limit}" if args.limit else ""
    job = command(
        code=str(code_dir),
        command=(
            "mkdir -p outputs eval/results && "
            "python ablation_runner.py "
            "--adapter ${{inputs.adapter}} "
            "--configs cumulative "
            f"--output outputs/ablation_results.json{limit_flag} "
            "&& PYTHONPATH=. python eval/analyze_offline.py "
            "&& python eval/report_results.py "
            "--input outputs/ablation_results.json "
            "--offline eval/results/offline_audit.json "
            "--output outputs/ablation_report.md "
            "|| (python eval/report_results.py "
            "--input outputs/ablation_results.json "
            "--offline eval/results/offline_audit.json "
            "--output outputs/ablation_report.md)"
        ),
        environment=env,
        compute=args.compute,
        experiment_name="daisy-ablation",
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
