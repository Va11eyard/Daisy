# Azure: обучение и «пуш» модели (Daisy-Model)

Краткий цикл для **этого** репозитория: датасет → **training job** в Azure ML → скачать или зарегистрировать **LoRA** → привязать к **online endpoint**.

Базовые веса **Qwen 7B** с Hugging Face на compute подтягиваются автоматически — отдельно «загружать 7B в Azure» не нужно. В реестр Azure ML вы **регистрируете артефакт fine-tune** (адаптер + токенизатор), не полный базовый чекпоинт, если не делали full fine-tune.

---

## 1. Подготовка

1. [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) + расширение ML: `az extension add -n ml`
2. Вход: `az login`
3. Подписка: `az account set --subscription "<subscription-id>"`
4. Python-пакеты для сабмита job (локально): `pip install azure-ai-ml azure-identity`

Переменные (PowerShell):

```powershell
$env:AZURE_SUBSCRIPTION_ID = "<id>"
$env:AZURE_RESOURCE_GROUP = "Daisy_group"
$env:AZUREML_WORKSPACE_NAME = "Daisy"
$env:AZUREML_COMPUTE_NAME = "gpu-cluster"
$env:HF_TOKEN = "<huggingface_token_if_required>"
```

---

## 2. Данные для обучения

Соберите `train.jsonl` / `val.jsonl` (см. [DATASET.md](DATASET.md), `scripts/prepare_dataset.py`).

Файлы должны лежать в **`data/train.jsonl`** и **`data/val.jsonl`**. Скрипт `scripts/submit_training_job.py` **копирует** их в `training/` перед упаковкой job.

---

## 3. Запуск обучения в Azure ML

### Вариант A — Python SDK (рекомендуется)

Из корня репозитория:

```powershell
cd E:\WebstormProjects\Daisy-Model
pip install azure-ai-ml azure-identity
$env:HF_TOKEN = "..."
$env:AZURE_SUBSCRIPTION_ID = "..."
$env:AZURE_RESOURCE_GROUP = "Daisy_group"
$env:AZUREML_WORKSPACE_NAME = "Daisy"
python scripts/submit_training_job.py
```

Опционально перед запуском задайте гиперпараметры через env, например:

```powershell
$env:NUM_EPOCHS = "2"
$env:MAX_SEQ_LENGTH = "1024"
$env:PER_DEVICE_TRAIN_BATCH_SIZE = "2"
$env:GRADIENT_ACCUMULATION_STEPS = "8"
```

Скрипт передаёт их в job (см. `training/train.py`).

### Вариант B — YAML + CLI

Скопируйте `data/*.jsonl` в `training/` вручную, затем:

```powershell
az ml job create --file azureml/command_job.yaml `
  --resource-group Daisy_group --workspace-name Daisy --subscription <subscription-id>
```

В `azureml/command_job.yaml` задайте `compute: azureml:gpu-cluster` под ваш кластер. Секрет **`HF_TOKEN`** лучше не писать в YAML: передайте при создании job или используйте [workspace connections / key vault](https://learn.microsoft.com/azure/machine-learning/how-to-use-secret-in-runs).

---

## 4. Мониторинг и артефакты

```powershell
az ml job list --resource-group Daisy_group --workspace-name Daisy --max-results 10 -o table
```

После статуса `Completed` скачайте выход (имя output обычно `default`; проверьте в Studio):

```powershell
az ml job download --name <job-name> --output-name default `
  --download-path .\checkpoints\from-azure --resource-group Daisy_group --workspace-name Daisy
```

Внутри будет путь вида `.../outputs/daisy-lora/` с `adapter_config.json`, весами LoRA и токенизатором.

---

## 5. «Пуш» модели = регистрация в Azure ML Model Registry

Локально (из скачанной папки с адаптером):

```powershell
$env:AZURE_SUBSCRIPTION_ID = "..."
$env:AZURE_RESOURCE_GROUP = "Daisy_group"
$env:AZUREML_WORKSPACE_NAME = "Daisy"
python scripts/register_model.py --path .\checkpoints\from-azure\...\daisy-lora --name daisy-finetuned-lora --version 7
```

Дальше в [azureml/deployment.yaml](../azureml/deployment.yaml) укажите:

`model: azureml:daisy-finetuned-lora:7`

и задеплойте managed online endpoint (см. [AZURE_TRAINING_AND_DEPLOY.md](AZURE_TRAINING_AND_DEPLOY.md), раздел про деплой).

---

## 6. Инференс

- Код скоринга: **`inference/score.py`**
- Образ / conda: **`inference/conda.yaml`**
- Шаблон деплоя: **`azureml/deployment.yaml`**

На endpoint выставьте `BASE_MODEL` тот же, что и при обучении, и при необходимости `INFERENCE_QUANTIZATION=4bit` для T4.

---

## 7. Чеклист

| Шаг | Действие |
|-----|----------|
| 1 | `data/train.jsonl`, `data/val.jsonl` готовы |
| 2 | `HF_TOKEN`, переменные Azure ML заданы |
| 3 | `python scripts/submit_training_job.py` или `az ml job create -f azureml/command_job.yaml` |
| 4 | Job `Completed` → скачать `outputs` |
| 5 | `python scripts/register_model.py --path ...` |
| 6 | Обновить `deployment.yaml` → создать/обновить deployment endpoint |

Подробности по GPU и квантизации: [GPU_TRAINING_AND_INFERENCE.md](GPU_TRAINING_AND_INFERENCE.md).
