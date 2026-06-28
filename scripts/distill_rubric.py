"""
Distill therapy books (data/md) into a structured correctness rubric.

Runs on BOOK TEXT ONLY — no user PHI. Output: data/knowledge/rubric.json

  $env:OPENAI_API_KEY = "..."
  python scripts/distill_rubric.py --output data/knowledge/rubric.json --limit 40

  $env:ANTHROPIC_API_KEY = "..."
  python scripts/distill_rubric.py --provider anthropic --resume
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

from md_to_dialogues import _infer_locale, _iter_md_files, _title_from_md  # noqa: E402
from rich_md_content import _chunk_paragraphs, _paragraphs, _split_sections  # noqa: E402

STATES = ("intake", "disclosure", "psychoeducation", "action_planning", "crisis")
MODALITY_HINTS = (
    ("cbt", re.compile(r"\bCBT|cognitive.behavior|кпт|когнитив", re.I)),
    ("dbt", re.compile(r"\bDBT|dialectical|дбт", re.I)),
    ("act", re.compile(r"\bACT|acceptance.commitment", re.I)),
    ("grief", re.compile(r"\bgrief|loss|горе|утрат", re.I)),
    ("anxiety", re.compile(r"\banxiety|тревог", re.I)),
    ("crisis", re.compile(r"\bcrisis|suicid|кризис", re.I)),
)


def _infer_modality(path: str, title: str) -> str:
    blob = f"{path} {title}"
    for name, pat in MODALITY_HINTS:
        if pat.search(blob):
            return name
    return "general"


def _rubric_prompt(*, chunk: str, title: str, modality: str, locale: str) -> str:
    return (
        f"Document: {title}\nModality: {modality}\nLocale hint: {locale}\n\n"
        "Extract a THERAPY RESPONSE RUBRIC from the passage below. "
        "Do NOT copy book prose. Write principles a chat companion should follow.\n\n"
        "Return ONE JSON object:\n"
        '{"state":"intake|disclosure|psychoeducation|action_planning|crisis",'
        '"principles":["..."], "do":["..."], "dont":["..."], "techniques":["..."],'
        '"exemplars":{"en":{"good":"...","bad":"..."},'
        '"ru":{"good":"...","bad":"..."}, "kk":{"good":"...","bad":"..."}}}\n\n'
        f"---\n{chunk[:6000]}\n---"
    )


def _call_teacher(provider: str, model: str, user_prompt: str) -> dict[str, Any] | None:
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system="Return valid JSON only. No markdown fences.",
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "text", None))
    else:
        from openai import OpenAI

        client = OpenAI()
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = (r.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _merge_rubric(base: dict[str, Any], entry: dict[str, Any]) -> None:
    state = entry.get("state") or "intake"
    if state not in STATES:
        state = "intake"
    states = base.setdefault("states", {})
    slot = states.setdefault(
        state,
        {"principles": [], "do": [], "dont": [], "techniques": [], "exemplars": {}},
    )
    for key in ("principles", "do", "dont", "techniques"):
        for item in entry.get(key) or []:
            if isinstance(item, str) and item not in slot[key]:
                slot[key].append(item)
    ex = entry.get("exemplars") or {}
    slot_ex = slot.setdefault("exemplars", {})
    for lang in ("en", "ru", "kk"):
        if lang in ex and isinstance(ex[lang], dict):
            slot_ex[lang] = ex[lang]


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill book MD into rubric.json")
    ap.add_argument("--md-root", type=Path, default=_ROOT / "data" / "md")
    ap.add_argument("--output", type=Path, default=_ROOT / "data" / "knowledge" / "rubric.json")
    ap.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    model = args.model or (
        "claude-sonnet-4-20250514" if args.provider == "anthropic" else "gpt-4o-mini"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rubric: dict[str, Any] = {"version": 1, "source": "book_distill", "states": {}}
    done: set[str] = set()
    if args.resume and args.output.is_file():
        with args.output.open(encoding="utf-8") as f:
            rubric = json.load(f)
        done_path = args.output.with_suffix(".done.json")
        if done_path.is_file():
            done = set(json.loads(done_path.read_text(encoding="utf-8")))

    count = 0
    for md_path in _iter_md_files(args.md_root):
        rel = str(md_path.relative_to(args.md_root)).replace("\\", "/")
        title = _title_from_md(md_path.read_text(encoding="utf-8", errors="replace"))
        locale = _infer_locale(rel)
        modality = _infer_modality(rel, title)
        text = md_path.read_text(encoding="utf-8", errors="replace")
        for section_title, section_body in _split_sections(text):
            for chunk in _chunk_paragraphs(_paragraphs(section_body)):
                cid = hashlib.sha256(f"{rel}|{section_title}|{chunk[:200]}".encode()).hexdigest()[:16]
                if cid in done:
                    continue
                prompt = _rubric_prompt(
                    chunk=chunk, title=title, modality=modality, locale=locale
                )
                entry = _call_teacher(args.provider, model, prompt)
                if entry:
                    _merge_rubric(rubric, entry)
                    done.add(cid)
                    count += 1
                    with args.output.open("w", encoding="utf-8") as f:
                        json.dump(rubric, f, ensure_ascii=False, indent=2)
                    done_path = args.output.with_suffix(".done.json")
                    done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
                if args.limit and count >= args.limit:
                    print(f"Reached limit {args.limit}")
                    return
                time.sleep(args.sleep)

    print(f"Wrote {args.output} ({count} new chunks)")


if __name__ == "__main__":
    main()
