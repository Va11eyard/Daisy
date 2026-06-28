# Импорт данных с Google Drive в `data/Rus`

Ссылка на папку:  
[https://drive.google.com/drive/folders/1aR-SeZkgpoUfu3t46MlXp4XKuwVsPaX9](https://drive.google.com/drive/folders/1aR-SeZkgpoUfu3t46MlXp4XKuwVsPaX9)

## Почему «ничего не скачалось»

1. **Доступ** — если папка только для вашего аккаунта, скрипты без входа в Google её не видят. Нужен либо доступ «любой по ссылке (читатель)», либо ручное скачивание через браузер.
2. **Место на диске** — при ошибке `No space left on device` сначала освободите место на диске с проектом или укажите другой путь:  
   `python scripts/download_gdrive_rus.py --output D:\datasets\Rus`

## Вариант A: папка публична по ссылке (проще всего)

В Google Drive: **Права доступа → с ограничениями / все, у кого есть ссылка → читатель**.

Затем в корне репозитория:

```bash
pip install gdown
python scripts/download_gdrive_rus.py
```

Файлы появятся в `data/Rus/`.

## Вариант B: папка приватная

- Скачайте архив через браузер: **ПКМ по папке → Скачать**, распакуйте в `data/Rus/`.
- Либо настройте `gdown` с cookies для приватных файлов (см. [документацию gdown](https://github.com/wkentaro/gdown)).

## Вариант C: rclone

Установите [rclone](https://rclone.org/), выполните `rclone config` для Google Drive, затем синхронизируйте папку в `data/Rus`.
