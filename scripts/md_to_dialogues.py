"""
Build training JSON (messages + meta) from markdown corpus under data/md.

Modes:
  --mode heuristic   Offline templates (no API).
  --mode api           Distillation: Claude/OpenAI write grounded JSON per chunk.

Quality:
  --quality standard   Original 4-turn dialog, one record per chunk (smaller, simpler).
  --quality rich       Section-aware split, 2 archetypes/chunk, 6-turn on long chunks, richer RU/EN templates.

Then run:
  python scripts/prepare_dataset.py --input data/raw/md_dialogues.json --output-dir data

Environment (api mode):
  ANTHROPIC_API_KEY + pip install anthropic   (--provider anthropic)
  OPENAI_API_KEY    + pip install openai      (--provider openai)

Optional:
  TEACHER_MODEL     Override model id (defaults below).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rich_md_content import (  # noqa: E402
    _chunk_paragraphs as _rich_chunk_paragraphs,
    _paragraphs as _rich_paragraphs,
    _split_sections,
    build_rich_record,
    pick_archetypes_for_chunk,
)

# --- Persona hints from folder path (substring match, first win) ---
_PERSONA_RULES: list[tuple[str, str]] = [
    ("Emergency", "calm_mentor"),
    ("кризис", "calm_mentor"),
    ("Эмпатия", "warm_friend"),
    ("сострадан", "warm_friend"),
    ("выгоран", "practical_helper"),
    ("тревог", "calm_mentor"),
    ("депресс", "gentle_explorer"),
    ("границ", "practical_helper"),
    ("Схема", "wise_teacher"),
    ("КПТ", "wise_teacher"),
    ("DBT", "practical_helper"),
    ("ACT", "gentle_explorer"),
    ("EMDR", "calm_mentor"),
    ("семь", "warm_friend"),
    ("дет", "warm_friend"),
]


def _infer_persona(rel_path: str) -> str:
    low = rel_path.lower()
    for needle, persona in _PERSONA_RULES:
        if needle.lower() in low:
            return persona
    return "flexible"


def _infer_locale(rel_path: str) -> str:
    if "Рус" in rel_path or "рус" in rel_path.lower():
        return "ru"
    if "Eng" in rel_path or "english" in rel_path.lower():
        return "en"
    if "kk" in rel_path.lower() or "қаз" in rel_path:
        return "kk"
    return "ru"


def _strip_md_noise(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_md(path: Path, body: str) -> str:
    m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return path.stem.replace("_", " ").strip()


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]


def _chunk_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        add = len(p) + (2 if buf else 0)
        if buf and buf_len + add > max_chars:
            chunks.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += add
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _split_for_turns(chunk: str) -> tuple[str, str]:
    """Split chunk into two parts at paragraph boundary for multi-turn."""
    paras = [x.strip() for x in chunk.split("\n\n") if x.strip()]
    if len(paras) <= 1:
        mid = max(1, len(chunk) // 2)
        return chunk[:mid].strip(), chunk[mid:].strip()
    k = 1
    while k < len(paras) and sum(len(paras[i]) for i in range(k)) < len(chunk) // 2:
        k += 1
    a = "\n\n".join(paras[:k]).strip()
    b = "\n\n".join(paras[k:]).strip()
    if not b:
        b = a
        a = ""
    return a or chunk[: len(chunk) // 2], b or chunk[len(chunk) // 2 :]


def _stable_pick(seed: str, options: list[str]) -> str:
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


_USER_OPENERS_RU = [
    'Можешь по-простому объяснить по теме «{title}»?',
    'Я запутался(ась). С чего начать разбираться в «{title}»?',
    'Мне важно понять «{title}» без лишней теории. Что главное?',
    'Расскажи спокойно: что здесь важно в теме «{title}»?',
]

_USER_FOLLOWUP_RU = [
    "Спасибо. А что ещё важно учесть на практике?",
    "Понятно. Какие ещё акценты из этого важны?",
    "Ок. Что дальше по смыслу?",
]


def _wrap_assistant(prefix_style: int, body: str, max_chars: int) -> str:
    body = body.strip()
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"
    prefixes = (
        "Я рядом с тобой в этом вопросе. По сути:\n\n",
        "Постараюсь говорить спокойно и по делу.\n\n",
        "Коротко и бережно:\n\n",
    )
    return prefixes[prefix_style % len(prefixes)] + body


def build_heuristic_record(
    *,
    chunk: str,
    title: str,
    rel_path: str,
    persona: str,
    locale: str,
    max_assistant_chars: int,
    seed: str,
) -> dict[str, Any]:
    opener = _stable_pick(seed + "o", _USER_OPENERS_RU).format(title=title)
    follow = _stable_pick(seed + "f", _USER_FOLLOWUP_RU)
    part_a, part_b = _split_for_turns(chunk)
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    a_text = _wrap_assistant(h, part_a, max_assistant_chars)
    b_text = _wrap_assistant(h + 1, part_b, max_assistant_chars)
    meta: dict[str, Any] = {
        "locale": locale,
        "persona": persona,
        "onboarding_summary": "",
        "user_context": f"source_md:{rel_path[:200]}",
    }
    return {
        "messages": [
            {"role": "user", "content": opener},
            {"role": "assistant", "content": a_text},
            {"role": "user", "content": follow},
            {"role": "assistant", "content": b_text},
        ],
        "meta": meta,
    }


def _iter_md_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*.md")):
        rel = str(p.relative_to(root))
        if rel.startswith("_") or "_conversion" in rel:
            continue
        out.append(p)
    return out


def _teacher_anthropic(
    *,
    chunk: str,
    title: str,
    persona: str,
    locale: str,
    model: str,
) -> dict[str, Any] | None:
    try:
        import anthropic
    except ImportError:
        raise SystemExit("api mode needs: pip install anthropic") from None

    client = anthropic.Anthropic()
    sys_prompt = (
        "Ты генератор обучающих диалогов для поддерживающего ассистента (не врач, не диагнозы). "
        "Верни ТОЛЬКО один JSON-объект без markdown, без комментариев. "
        'Схема: {"messages":[{"role":"user"|"assistant","content":"..."},...],'
        '"meta":{"persona":"' + persona + '","locale":"' + locale + '"}}. '
        "3–6 сообщений, чередование user/assistant, начинать с user. "
        "Язык ответов: соответствует locale. "
        "Используй ТОЛЬКО факты из фрагмента; не выдумывай клинические заключения; "
        "если во фрагменте есть риск/кризис — мягко направь к очной помощи, без конкретных диагнозов."
    )
    user_block = f"Тема: {title}\npersona: {persona}\nlocale: {locale}\n\nФрагмент:\n---\n{chunk}\n---"
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_block}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "text", None))
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "messages" not in data:
        return None
    return data


def _teacher_openai(
    *,
    chunk: str,
    title: str,
    persona: str,
    locale: str,
    model: str,
) -> dict[str, Any] | None:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("api mode needs: pip install openai") from None

    client = OpenAI()
    sys_prompt = (
        "Ты генератор обучающих диалогов для поддерживающего ассистента. "
        "Верни ТОЛЬКО JSON: {\"messages\":[...],\"meta\":{...}}. "
        "3–6 сообщений user/assistant; только факты из фрагмента; без диагнозов."
    )
    user_block = f"Тема: {title}\npersona: {persona}\nlocale: {locale}\n\nФрагмент:\n---\n{chunk}\n---"
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_block},
        ],
        temperature=0.4,
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
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="MD corpus → dialogue JSON for prepare_dataset.py")
    ap.add_argument("--md-root", type=Path, default=_ROOT / "data" / "md")
    ap.add_argument("--output", type=Path, default=_ROOT / "data" / "raw" / "md_dialogues.json")
    ap.add_argument("--mode", choices=("heuristic", "api"), default="heuristic")
    ap.add_argument("--quality", choices=("standard", "rich"), default="rich")
    ap.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--max-chunks-per-file", type=int, default=None, help="default: 12 standard, 22 rich")
    ap.add_argument("--archetypes-per-chunk", type=int, default=2, help="rich heuristic only")
    ap.add_argument("--long-chunk-chars", type=int, default=4200, help="rich: use 6-turn dialog at or above this size")
    ap.add_argument("--max-chars", type=int, default=None, help="default: 3600 standard, 3400 rich")
    ap.add_argument("--min-chunk-chars", type=int, default=None, help="default: 280 standard, 260 rich")
    ap.add_argument("--max-assistant-chars", type=int, default=2800)
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.max_chunks_per_file is None:
        args.max_chunks_per_file = 22 if args.quality == "rich" else 12
    if args.max_chars is None:
        args.max_chars = 3400 if args.quality == "rich" else 3600
    if args.min_chunk_chars is None:
        args.min_chunk_chars = 260 if args.quality == "rich" else 280
    random.seed(args.seed)

    md_root = args.md_root.resolve()
    if not md_root.is_dir():
        raise SystemExit(f"Not a directory: {md_root}")

    default_teacher = (
        "claude-sonnet-4-20250514"
        if args.provider == "anthropic"
        else "gpt-4o-mini"
    )
    teacher_model = os.environ.get("TEACHER_MODEL", default_teacher)

    files = _iter_md_files(md_root)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    records: list[dict[str, Any]] = []
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
        if args.mode == "heuristic" and args.quality == "rich":
            sections = _split_sections(body)
            bucket: list[tuple[str | None, str]] = []
            for sec_title, sec_body in sections:
                paras = _rich_paragraphs(sec_body)
                if not paras:
                    continue
                sec_chunks = _rich_chunk_paragraphs(paras, args.max_chars)
                for c in sec_chunks:
                    if len(c) >= args.min_chunk_chars:
                        bucket.append((sec_title, c))
            bucket = bucket[: args.max_chunks_per_file]
            for chunk_index, (sec_title, chunk) in enumerate(bucket):
                arche_list = pick_archetypes_for_chunk(
                    f"{rel}|{chunk_index}",
                    min(args.archetypes_per_chunk, 6),
                )
                n_paras = len([x for x in chunk.split("\n\n") if x.strip()])
                long_dialog = len(chunk) >= args.long_chunk_chars or n_paras > 5
                emergency = "Emergency" in rel or "кризис" in rel.lower()
                for arch in arche_list:
                    seed = f"{rel}|{chunk_index}|{arch}"
                    rec = build_rich_record(
                        chunk=chunk,
                        title=title,
                        section_title=sec_title,
                        rel_path=rel,
                        persona=persona,
                        locale=locale,
                        archetype=arch,
                        max_assistant_chars=args.max_assistant_chars,
                        seed=seed,
                        emergency=emergency,
                        long_dialog=long_dialog,
                    )
                    records.append(rec)
            if (idx + 1) % 50 == 0:
                print(f"... processed {idx + 1}/{len(files)} files, {len(records)} rows", file=sys.stderr)
            continue

        paras = _paragraphs(body)
        if not paras:
            continue
        chunks = _chunk_paragraphs(paras, args.max_chars)
        chunks = [c for c in chunks if len(c) >= args.min_chunk_chars]
        chunks = chunks[: args.max_chunks_per_file]

        for j, chunk in enumerate(chunks):
            seed = f"{rel}|{j}"
            if args.mode == "heuristic":
                rec = build_heuristic_record(
                    chunk=chunk,
                    title=title,
                    rel_path=rel,
                    persona=persona,
                    locale=locale,
                    max_assistant_chars=args.max_assistant_chars,
                    seed=seed,
                )
                records.append(rec)
            else:
                if args.provider == "anthropic":
                    if not os.environ.get("ANTHROPIC_API_KEY"):
                        raise SystemExit("Set ANTHROPIC_API_KEY for api mode")
                    got = _teacher_anthropic(
                        chunk=chunk,
                        title=title,
                        persona=persona,
                        locale=locale,
                        model=teacher_model,
                    )
                else:
                    if not os.environ.get("OPENAI_API_KEY"):
                        raise SystemExit("Set OPENAI_API_KEY for api mode")
                    got = _teacher_openai(
                        chunk=chunk,
                        title=title,
                        persona=persona,
                        locale=locale,
                        model=teacher_model,
                    )
                if got is None:
                    continue
                meta = got.get("meta") if isinstance(got.get("meta"), dict) else {}
                meta.setdefault("locale", locale)
                meta.setdefault("persona", persona)
                meta.setdefault("user_context", f"source_md:{rel[:200]}")
                got["meta"] = meta
                records.append(got)
        if (idx + 1) % 50 == 0:
            print(f"... processed {idx + 1}/{len(files)} files, {len(records)} rows", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
