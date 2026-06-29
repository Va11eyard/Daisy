"""Merge Azure deployment env vars with a local YAML patch (patch wins on duplicate keys)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: merge_deployment_env.py <patch.yaml> [out.yaml]",
            file=sys.stderr,
        )
        print(
            "  Default out: azureml/.merged-deploy.yaml (gitignored; delete after deploy).",
            file=sys.stderr,
        )
        return 2
    patch_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("azureml/.merged-deploy.yaml")
    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        print("Azure CLI (az) not found in PATH.", file=sys.stderr)
        return 1
    with patch_path.open(encoding="utf-8") as f:
        patch = yaml.safe_load(f)
    deploy_name = patch.get("name") or "gpu-deployment-finetuned"
    r = subprocess.run(
        [
            az,
            "ml",
            "online-deployment",
            "show",
            "--name",
            deploy_name,
            "--endpoint-name",
            patch.get("endpoint_name") or "daisy-therapy",
            "-g",
            "Daisy_group",
            "-w",
            "Daisy",
            "--subscription",
            "9239bc75-105c-486e-8957-da8e49309c55",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    azure = json.loads(r.stdout)
    env_azure = azure.get("environment_variables") or {}
    env_patch = patch.get("environment_variables") or {}
    merged_env = {**env_azure, **env_patch}
    patch["environment_variables"] = merged_env
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Paths in YAML are resolved relative to this file; keep under azureml/ so ../inference works.
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            patch,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
