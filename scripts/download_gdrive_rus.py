"""
Скачать папку Google Drive в data/FoodForAIDaisy (корень репозитория).

Текущая папка по умолчанию: «Food for AI Daisy» (подпапки Emergency cases, Eng, TOV, Рус).
Старый датасет можно скачать так:  --id 1aR-SeZkgpoUfu3t46MlXp4XKuwVsPaX9 --output data/Rus

Требования:
  pip install gdown

Публичная папка (доступ «По ссылке: все, у кого есть ссылка» — читатель):
  python scripts/download_gdrive_rus.py

Закрытая папка — см. docs/google_drive_import.md

На Windows при ошибке кодировки консоли скрипт сам переключает stdout на UTF-8.

Скачивание идёт по одному файлу: если один файл недоступен (лимит Google или права),
остальные всё равно докачиваются. Повторный запуск пропускает уже существующие файлы
(resume).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# https://drive.google.com/drive/folders/1IvCLPzZieUb3x9No1gAQ43oYcTkEi87L
FOLDER_ID = "1IvCLPzZieUb3x9No1gAQ43oYcTkEi87L"

REPO_ROOT = Path(__file__).resolve().parents[1]


_WIN_BAD = '<>:"/\\|?*'


def _sanitize_windows_path(p: Path) -> Path:
    """Убирает недопустимые для Windows символы в именах и завершающие пробелы/точки."""
    parts: list[str] = []
    for part in Path(p).parts:
        if part in (".", ".."):
            parts.append(part)
            continue
        if re.match(r"^[A-Za-z]:\\?$", part):
            parts.append(part)
            continue
        s = "".join("_" if c in _WIN_BAD else c for c in part)
        s = s.rstrip(" .")
        parts.append(s if s else "_")
    return Path(*parts)


def download_google_drive_file(file_id: str, dest: Path, timeout: int = 120) -> None:
    """
    Скачивание по id: тот же цикл, что в gdown.download (подтверждение «вирусного» скана,
    Google Docs/Sheets/Slides, редиректы на usercontent).
    """
    from gdown.download import _get_session, get_url_from_gdrive_confirmation
    from gdown.parse_url import parse_url

    # Старый UA из gdown даёт 403 от Google Drive; современный Chrome — нормально.
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    # Без cookies из ~/.cache/gdown/cookies.txt: иначе часто 403 для публичных файлов.
    sess = _get_session(proxy=None, use_cookies=False, user_agent=user_agent)

    url = f"https://drive.google.com/uc?id={file_id}"
    url_origin = url
    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=False)

    res = None
    while True:
        res = sess.get(url, stream=True, verify=True, timeout=timeout)

        if not (gdrive_file_id and is_gdrive_download_link):
            break

        if url == url_origin and res.status_code == 500:
            res.close()
            url = f"https://drive.google.com/open?id={gdrive_file_id}"
            continue

        ct = res.headers.get("Content-Type") or ""
        if ct.startswith("text/html"):
            m = re.search("<title>(.+)</title>", res.text)
            if m and m.groups()[0].endswith(" - Google Docs"):
                res.close()
                url = f"https://docs.google.com/document/d/{gdrive_file_id}/export?format=docx"
                continue
            if m and m.groups()[0].endswith(" - Google Sheets"):
                res.close()
                url = f"https://docs.google.com/spreadsheets/d/{gdrive_file_id}/export?format=xlsx"
                continue
            if m and m.groups()[0].endswith(" - Google Slides"):
                res.close()
                url = f"https://docs.google.com/presentation/d/{gdrive_file_id}/export?format=pptx"
                continue

        if "Content-Disposition" in res.headers:
            break

        try:
            text = res.text
            res.close()
            url = get_url_from_gdrive_confirmation(text)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось получить ссылку скачивания (как в gdown): {e}"
            ) from e
        continue

    assert res is not None

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as f:
            for chunk in res.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    finally:
        res.close()


def _utf8_stdio() -> None:
    """Avoid UnicodeEncodeError on Windows when printing non-CP1251 paths."""
    if sys.platform != "win32":
        return
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "FoodForAIDaisy",
        help="Куда сохранить файлы",
    )
    parser.add_argument(
        "--id",
        default=FOLDER_ID,
        help="ID папки Google Drive",
    )
    args = parser.parse_args()

    try:
        import gdown
        import requests
        from gdown.download_folder import download_folder
    except ImportError:
        print("Установите: pip install gdown requests", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Список файлов (без скачивания)…")

    file_list = download_folder(
        id=args.id,
        output=str(args.output),
        skip_download=True,
        quiet=True,
        remaining_ok=True,  # папки с 50+ файлами (ограничение gdown)
    )
    if not file_list:
        print("Не удалось получить список файлов. Проверьте доступ по ссылке.", file=sys.stderr)
        sys.exit(1)

    ok = 0
    fail = 0
    for item in file_list:
        local_path = _sanitize_windows_path(Path(item.local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_file() and local_path.stat().st_size > 0:
            print(f"[skip] {local_path.name}")
            ok += 1
            continue
        for attempt in range(3):
            try:
                download_google_drive_file(item.id, local_path)
                print(f"[ok] {item.path}")
                ok += 1
                break
            except (requests.exceptions.RequestException, OSError, RuntimeError) as e:
                wait = (attempt + 1) * 15
                print(
                    f"[retry {attempt + 1}/3] {item.path}: {e}\n"
                    f"   Пауза {wait}s (лимит Google / сеть)",
                    file=sys.stderr,
                )
                if attempt == 2:
                    print(
                        f"[FAIL] Скачайте вручную: https://drive.google.com/file/d/{item.id}/view",
                        file=sys.stderr,
                    )
                    fail += 1
                else:
                    time.sleep(wait)
            except Exception as e:
                print(f"[FAIL] {item.path}: {e}", file=sys.stderr)
                fail += 1
                break

    print(f"Готово: успешно {ok}, ошибок {fail}. Каталог: {args.output}")


if __name__ == "__main__":
    main()
