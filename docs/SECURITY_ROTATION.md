# Secret rotation runbook

If `deploy-show.json`, `azureml/.merged-deploy.yaml`, or chat logs ever contained live credentials, rotate immediately:

1. **Hugging Face** — Regenerate `HF_TOKEN` at https://huggingface.co/settings/tokens
2. **Azure OpenAI** — Rotate key in Azure Portal → Cognitive Services → Keys
3. **Azure Translator** — Rotate key in the Translator resource
4. **Azure ML endpoint** — Regenerate primary/secondary keys on the online endpoint
5. **Audit** — Review Azure Activity Log and HF token usage for unauthorized access

After rotation, update secrets in Azure ML workspace / deployment env (Studio or Key Vault references). Never save `az ml online-deployment show` output to the repo.
