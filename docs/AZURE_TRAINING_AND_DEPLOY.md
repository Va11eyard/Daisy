# Обучение и деплой DAISY в Azure ML

Практическое руководство по тому, как в этом репозитории запускается fine-tuning Llama 3.2 3B (LoRA) и как обновляется production endpoint.

---

## 1. Что где лежит

| Компонент | Путь / имя (репозиторий **Daisy-Model**) |
|-----------|------------|
| Workspace Azure ML | `daisy` (пример; ваш workspace) |
| Resource group | `Daisy_group` (пример) |
| Subscription | задаётся в `az` / env `AZURE_SUBSCRIPTION_ID` |
| Online endpoint | задаётся в `azureml/deployment.yaml` (`endpoint_name`) |
| Скрипт инференса | `inference/score.py` |
| Конфиг деплоя | `azureml/deployment.yaml` |
| Обучение | `training/train.py`, данные `data/train.jsonl`, `data/val.jsonl` |
| Сабмит job | `scripts/submit_training_job.py` или `azureml/command_job.yaml` |
| Регистрация LoRA в реестре | `scripts/register_model.py` |
| Быстрый гайд «от данных до деплоя» | [AZURE_PUSH_AND_TRAIN.md](AZURE_PUSH_AND_TRAIN.md) |

*Старые имена вроде `azure_training_package/`, `score_multilingual.py`, `register_model_v6.py` относятся к другим репозиториям/итерациям; в этом проекте используйте пути из таблицы выше.*

---

## 2. Предварительные условия

1. Установлен [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) и расширение ML: `az extension add -n ml`
2. Вход в Azure: `az login`
3. Подписка: `az account set --subscription "<subscription-id>"`
4. Токен Hugging Face для загрузки Llama: переменная `HF_TOKEN` или значение в `job.yaml` (для production лучше Key Vault)

---

## 3. Запуск обучения (training job)

### 3.1 Как устроен job

- Каталог `training/` содержит `train.py`, `conda.yaml` и (после копирования) `train.jsonl` / `val.jsonl`.
- Команда в job: `python train.py`.
- По умолчанию загружается `Qwen/Qwen2.5-7B-Instruct`, LoRA 8-bit (см. `training/train.py`), выход в `./outputs/daisy-lora`.
- Compute: кластер `gpu-cluster` (например `Standard_NC4as_T4_v3` — T4 16 GB).

### 3.2 Отправка job из корня репозитория

```powershell
cd <корень репозитория>
python scripts/submit_training_job.py
```

или

```powershell
az ml job create --file azureml/command_job.yaml --resource-group Daisy_group --workspace-name daisy --subscription <subscription-id>
```

Полный пошаговый сценарий: [AZURE_PUSH_AND_TRAIN.md](AZURE_PUSH_AND_TRAIN.md).

В выводе будет имя job (например `affable_arch_nvfjghkrn4`) и ссылка на Azure ML Studio.

### 3.3 Мониторинг

```powershell
az ml job stream --name <job-name> --resource-group Daisy_group --workspace-name daisy
```

Или в портале: [ml.azure.com](https://ml.azure.com) → Jobs → нужный эксперимент (`daisy-finetuning`).

### 3.4 Список недавних job

```powershell
az ml job list --resource-group Daisy_group --workspace-name daisy --max-results 20 -o table
```

Статус `Completed` означает успешное завершение; `Canceled` / `Failed` — нужно смотреть логи.

---

## 4. Скачивание артефактов после обучения

Артефакты job пишутся в output с именем `default` (см. `az ml job show`).

```powershell
New-Item -ItemType Directory -Force -Path .\checkpoints\daisy-finetuned-v6
az ml job download --name <job-name> --output-name default --download-path .\checkpoints\daisy-finetuned-v6 --resource-group Daisy_group --workspace-name daisy
```

Типичная структура после скачивания:

```
checkpoints\daisy-finetuned-v6\artifacts\outputs\daisy-finetuned\
  adapter_config.json
  adapter_model.safetensors
  chat_template.jinja
  README.md
  checkpoint-1000\   # промежуточные чекпоинты (не обязательны для деплоя)
  ...
```

Для **регистрации в Azure** достаточно финальной папки с LoRA (без тяжёлых `checkpoint-*`), чтобы не загружать сотни мегабайт лишнего. Можно скопировать только нужные файлы в отдельную папку, например `checkpoints\daisy-v6-clean` (как в `register_model_v6.py`).

---

## 5. Регистрация модели в Model Registry

Имя в реестре обычно `daisy-finetuned-lora`, версии инкрементируются (`1`, `2`, …).

- Путь к артефактам: локальная папка с `adapter_config.json`, файлами весов LoRA и токенизатором.
- Вызов: `python scripts/register_model.py --path <папка> --name daisy-finetuned-lora --version <n>` (см. [AZURE_PUSH_AND_TRAIN.md](AZURE_PUSH_AND_TRAIN.md)).

После регистрации в Studio: **Models** → `daisy-finetuned-lora` → версия.

---

## 6. Деплой на managed online endpoint

### 6.1 Связка endpoint ↔ модель ↔ код

Файл `azureml/deployment.yaml` задаёт:

- `model: azureml:daisy-finetuned-lora:<версия>` — какая версия LoRA подставляется в контейнер;
- `code_configuration` → `inference`, `score.py`;
- `environment` (conda + образ CUDA);
- `environment_variables` (переводчик, HF, квантизация и т.д.);
- `instance_type` (например `Standard_NC4as_T4_v3`).

`inference/model_loader.py` ищет полный чекпоинт или LoRA-адаптер под `AZUREML_MODEL_DIR` (см. README).

### 6.2 Обновление после новой тренировки и нового inference

1. Зарегистрировать модель новой версией (п. 5 или `scripts/register_model.py`).
2. В `azureml/deployment.yaml` выставить `model: azureml:daisy-finetuned-lora:<новая-версия>` и при необходимости `name` деплоя / `endpoint_name`.
3. Запустить обновление деплоя через Azure ML CLI или Studio.

Пример:

```powershell
az ml online-deployment update --name gpu-deployment-daisy --endpoint <your-endpoint> --file azureml/deployment.yaml --resource-group Daisy_group --workspace-name daisy
```

### 6.3 Только код, без смены модели

Если менялись только `inference/score.py` или `conda.yaml`, можно обновить код деплоя без смены строки `model`.

### 6.4 Проверка статуса

```powershell
az ml online-deployment show --name gpu-deployment-daisy --endpoint <your-endpoint> --resource-group Daisy_group --workspace-name daisy --query provisioning_state -o tsv
```

Ожидаемое значение: `Succeeded` (обновление часто занимает 5–15 минут).

### 6.5 Логи при проблемах

```powershell
az ml online-deployment get-logs --name gpu-deployment-daisy --endpoint <your-endpoint> --resource-group Daisy_group --workspace-name daisy --lines 200
```

---

## 7. Альтернативные пути обучения

Основной путь в этом репозитории: **`training/train.py`** + **`scripts/submit_training_job.py`** или **`azureml/command_job.yaml`**. Дополнительно см. [AZURE_PUSH_AND_TRAIN.md](AZURE_PUSH_AND_TRAIN.md).

---

## 8. Краткий чеклист «от тренировки до продакшена»

1. `az login`, верная подписка.
2. `python scripts/submit_training_job.py` или `az ml job create --file azureml/command_job.yaml ...`
3. Дождаться `Completed`, при необходимости скачать артефакты `az ml job download ... --output-name default`.
4. Подготовить папку с финальным LoRA (при необходимости без лишних `checkpoint-*`).
5. `python scripts/register_model.py --path ... --name daisy-finetuned-lora --version <n>`
6. В `azureml/deployment.yaml` указать новую версию модели.
7. Создать/обновить online deployment через CLI или Studio.
8. Дождаться `provisioning_state = Succeeded`, прогнать тесты endpoint.

---

## 9. GPU VM size — проверка через `az` (актуальные поля из Azure)

Имя размера из `azureml/deployment.yaml` в этом репозитории: **`Standard_NC4as_T4_v3`** (1× GPU семейства **NCASv3_T4** — NVIDIA **T4**, **16 GB** VRAM по спецификации серии).

Подставьте свой регион вместо `eastus`:

```powershell
az vm list-skus --location eastus --size Standard_NC4as_T4_v3 --resource-type virtualMachines -o table
```

Пример вывода (получено `az` 2026-03-28, регион **eastus**):

```text
ResourceType     Locations    Name                  Zones    Restrictions
---------------  -----------  --------------------  -------  --------------
virtualMachines  eastus       Standard_NC4as_T4_v3  1,2,3    None
```

Детали возможностей (фрагмент JSON из `az vm list-skus ... -o json`): **family** `Standard NCASv3_T4 Family`, **GPUs** `1`, **vCPUs** `4`, **MemoryGB** `28`, **restrictions** `[]` (нет блокировок в этом примере).

**English text for Azure / quota request (fill region and subscription):**

> We need VM size **`Standard_NC4as_T4_v3`** (Azure family **`Standard NCASv3_T4 Family`**, **1× GPU**, **4 vCPUs**, **28 GiB RAM**) for Azure Machine Learning inference. Please confirm **quota / capacity** for this SKU in region **[YOUR_REGION]** (CLI check: `az vm list-skus --location <region> --size Standard_NC4as_T4_v3 --resource-type virtualMachines -o table`). Subscription ID: **[YOUR_SUBSCRIPTION_ID]**.

---

## 10. См. также

- [AZURE_PUSH_AND_TRAIN.md](AZURE_PUSH_AND_TRAIN.md) — пошагово: данные → job → регистрация → деплой.
- [GPU_TRAINING_AND_INFERENCE.md](GPU_TRAINING_AND_INFERENCE.md) — кратко: обучение vs инференс, T4, 4-bit.
- [DATASET.md](DATASET.md) — формат данных и `meta` под прод.
- [EVAL.md](EVAL.md) — проверки после fine-tune.

---

*Имена endpoint, deployment и версий моделей подставляйте своими. Раздел 9 дополнен выводом `az vm list-skus` для проверки SKU в регионе.*
