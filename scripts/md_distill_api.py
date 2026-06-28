"""
API distillation: one teacher call per MD chunk (same section-aware chunking as --quality rich).

Writes JSONL (append + flush) so runs can stop/resume. Each record has _distill_id for dedup.

  $env:ANTHROPIC_API_KEY = "..."
  python scripts/md_distill_api.py --output data/raw/md_distilled.jsonl

  # resume after interrupt
  python scripts/md_distill_api.py --output data/raw/md_distilled.jsonl --resume

Then:
  python scripts/prepare_dataset.py --input data/raw/md_distilled.jsonl --output-dir data

OpenAI:
  $env:OPENAI_API_KEY = "..."
  python scripts/md_distill_api.py --provider openai
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from md_to_dialogues import (  # noqa: E402
    _infer_locale,
    _infer_persona,
    _iter_md_files,
    _strip_md_noise,
    _title_from_md,
)
from rich_md_content import (  # noqa: E402
    _chunk_paragraphs as _rich_chunk_paragraphs,
    _paragraphs as _rich_paragraphs,
    _split_sections,
)


def _load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("_distill_id")
            if isinstance(cid, str):
                done.add(cid)
    return done


def _distill_prompt(
    *,
    chunk: str,
    title: str,
    section_title: str | None,
    persona: str,
    locale: str,
    emergency: bool,
) -> str:
    sec = f"Раздел: {section_title}\n" if section_title else ""
    em = (
        "\nКонтекст: материал может касаться кризиса. Не ставь диагнозы; при острой угрозе — направь к очной/экстренной помощи.\n"
        if emergency
        else ""
    )
    loc_note = (
        "Ответы пользователя и ассистента на русском."
        if locale == "ru"
        else f"User and assistant messages in language matching locale={locale}."
    )
    return (
        f"Тема документа: {title}\n{sec}"
        f"Persona ассистента (тон): {persona}. Locale: {locale}. {loc_note}\n"
        f"{em}\n"
        "Создай ОДИН обучающий диалог 4–6 сообщений (чередование user/assistant, первое от user). "
        "Ассистент — поддерживающий, без медицинских диагнозов и без обещаний излечения. "
        "Опирайся ТОЛЬКО на фрагмент ниже; не добавляй факты извне.\n\n"
        f"---\n{chunk}\n---"
    )


def _call_teacher(
    *,
    provider: str,
    model: str,
    user_prompt: str,
    persona: str,
    locale: str,
) -> dict[str, Any] | None:
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY")
        try:
            import anthropic
        except ImportError:
            raise SystemExit("pip install anthropic") from None
        client = anthropic.Anthropic()
        sys_p = (
            "Ты редактор датасета для терапевтичного чат-ассистента. "
            'Верни один JSON: {"messages":[{"role":"user","content":"..."},...],'
            '"meta":{"persona":"'
            + persona
            + '","locale":"'
            + locale
            + '"}}. '
            "Без markdown, без ```. Только валидный JSON."
        )
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            system=sys_p,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "text", None))
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("pip install openai") from None
        client = OpenAI()
        sys_p = (
            "Return a single JSON object only: "
            '{"messages":[...],"meta":{"persona":"...","locale":"..."}}. '
            "4–6 alternating user/assistant messages, Russian if locale ru."
        )
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
        )
        raw = (r.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "messages" not in data:
        return None
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta.setdefault("persona", persona)
    meta.setdefault("locale", locale)
    data["meta"] = meta
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-root", type=Path, default=_ROOT / "data" / "md")
    ap.add_argument("--output", type=Path, default=_ROOT / "data" / "raw" / "md_distilled.jsonl")
    ap.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--max-chunks-per-file", type=int, default=22)
    ap.add_argument("--max-chars", type=int, default=3400)
    ap.add_argument("--min-chunk-chars", type=int, default=260)
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between API calls")
    ap.add_argument("--resume", action="store_true", help="skip _distill_id already in output file")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--limit-total-chunks", type=int, default=None, help="stop after N new chunks")
    args = ap.parse_args()

    default_model = (
        "claude-sonnet-4-20250514"
        if args.provider == "anthropic"
        else os.environ.get("OPENAI_DISTILL_MODEL", "gpt-4o-mini")
    )
    model = os.environ.get("TEACHER_MODEL", default_model)

    md_root = args.md_root.resolve()
    files = _iter_md_files(md_root)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    done = _load_done_ids(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    mode = "a" if args.output.exists() and args.resume else "w"

    with args.output.open(mode, encoding="utf-8") as out:
        for idx, path in enumerate(files):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = _strip_md_noise(raw)
            if len(body) < args.min_chunk_chars:
                continue
            rel = str(path.relative_to(md_root))
            title = _title_from_md(path, body)
            persona = _infer_persona(rel)
            locale = _infer_locale(rel)
            emergency = "Emergency" in rel or "кризис" in rel.lower()

            sections = _split_sections(body)
            bucket: list[tuple[str | None, str]] = []
            for sec_title, sec_body in sections:
                paras = _rich_paragraphs(sec_body)
                if not paras:
                    continue
                for c in _rich_chunk_paragraphs(paras, args.max_chars):
                    if len(c) >= args.min_chunk_chars:
                        bucket.append((sec_title, c))
            bucket = bucket[: args.max_chunks_per_file]

            for chunk_index, (sec_title, chunk) in enumerate(bucket):
                cid = f"{rel}|{chunk_index}"
                if cid in done:
                    continue
                up = _distill_prompt(
                    chunk=chunk,
                    title=title,
                    section_title=sec_title,
                    persona=persona,
                    locale=locale,
                    emergency=emergency,
                )
                rec = _call_teacher(
                    provider=args.provider,
                    model=model,
                    user_prompt=up,
                    persona=persona,
                    locale=locale,
                )
                if rec is None:
                    continue
                rec["_distill_id"] = cid
                meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
                meta.setdefault("user_context", f"source_md:{rel[:200]}")
                rec["meta"] = meta
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done.add(cid)
                new_count += 1
                if args.sleep:
                    time.sleep(args.sleep)
                if args.limit_total_chunks is not None and new_count >= args.limit_total_chunks:
                    print(f"Stopped at --limit-total-chunks={args.limit_total_chunks}", file=sys.stderr)
                    print(f"Appended {new_count} new rows -> {args.output}")
                    return

            if (idx + 1) % 20 == 0:
                print(f"... files {idx + 1}/{len(files)}, new_chunks {new_count}", file=sys.stderr)

    print(f"Appended {new_count} new rows -> {args.output}")


if __name__ == "__main__":
    main()
