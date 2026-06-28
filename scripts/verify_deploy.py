"""Verify live Azure endpoint runs new inference code (not just env vars)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "_probe_turn.json"


def _count_script_chars(text: str) -> tuple[int, int]:
    latin = 0
    cyr = 0
    for c in text:
        if not c.isalpha():
            continue
        if "\u0400" <= c <= "\u04ff":
            cyr += 1
        elif c.isascii() and c.isalpha():
            latin += 1
    return latin, cyr

# Signatures of OLD inference (template fallback always on, body-scan default)
OLD_BODY_SCAN = "what do you notice in your body"
OLD_TEMPLATE_OPENING = "it sounds hard to put into words right now"
# June canned tail pool (_OPEN_QUESTIONS) — must not appear after pre-June restore
CANNED_RU_TAILS = (
    "что тебе сейчас нужнее — выговориться",
    "с чего бы тебе хотелось начать",
)
PRE_JUNE_BUILD = "2026-04-pre-june-restore"

TESTS = [
    {
        "label": "greeting_empty",
        "message": "HI HOW ARE YOU",
        "history": [],
        "forbid_substrings": [OLD_BODY_SCAN, OLD_TEMPLATE_OPENING],
        "require_question": True,
    },
    {
        "label": "hello_sad_empty",
        "message": "hello still feeling sad",
        "history": [],
        "forbid_substrings": [OLD_BODY_SCAN],
        "forbid_exact": ["Hello.", "Hi.", "Hey."],
        "min_len": 25,
    },
    {
        "label": "breakup_turn2",
        "message": "emptiness in my heart area",
        "history": [
            {
                "role": "user",
                "content": (
                    "im feeling weird you know. i recently broke up w my bf and i cant stop thinking about him. "
                    "at first i was mad but now i feellike i miss him"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Breakups leave a particular kind of hollow — the person is gone but the space they took up isn't. "
                    "What's sitting heaviest right now, the missing or the anger?"
                ),
            },
        ],
        "forbid_substrings": ["that must feel very empty and poignant"],
        "forbid_exact": ["Hello."],
    },
    {
        "label": "meta_feedback_too_short",
        "message": "You are answering too short",
        "history": [
            {"role": "user", "content": "work has been overwhelming lately"},
            {"role": "assistant", "content": "That must be incredibly stressful."},
        ],
        "forbid_substrings": [
            "i'm sorry to hear that",
            "it sounds like you're dealing with",
            "that must be incredibly stressful",
        ],
        "require_question": True,
        "min_len": 40,
    },
    {
        "label": "ru_anxiety_fresh",
        "message": "мне тревожно сегодня",
        "locale": "ru",
        "history": [],
        "forbid_substrings": list(CANNED_RU_TAILS) + ["? : . : ?", "может быть,?"],
        "require_majority_cyrillic": True,
        "require_question": True,
        "min_len": 25,
    },
    {
        "label": "ru_work_stress_no_leak",
        "message": "мой AI модель постоянно ломается, СЕО на меня злится((",
        "locale": "ru",
        "history": [],
        "forbid_substrings": ["translation:", "(translation"],
        "require_majority_cyrillic": True,
        "min_len": 30,
    },
    {
        "label": "anxiety_grounding",
        "message": "I'm feeling anxious today",
        "history": [],
        "conversation_id": "verify-anxiety-grounding-1",
        "forbid_substrings": [
            "what feels most important to talk about right now",
            "bracing can show how much we're trying to hold together",
            "what seems to be on your mind right now",
        ],
        "require_question": True,
        "min_len": 30,
    },
]


def invoke(payload: dict) -> dict:
    PROBE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        raise RuntimeError("Azure CLI not found")
    r = subprocess.run(
        [
            az,
            "ml",
            "online-endpoint",
            "invoke",
            "--name",
            "daisy-therapy",
            "-g",
            "Daisy_group",
            "-w",
            "Daisy",
            "--request-file",
            str(PROBE),
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    outer = json.loads(r.stdout)
    body = json.loads(outer) if isinstance(outer, str) else outer
    if isinstance(body, str):
        body = json.loads(body)
    return body


def main() -> None:
    issues: list[str] = []
    signals_new: list[str] = []

    print("=== Live behavioral probes (daisy-therapy) ===")
    for t in TESTS:
        payload = {
            "message": t["message"],
            "locale": t.get("locale", "en"),
            "history": t.get("history", []),
            "max_tokens": 256,
        }
        if t.get("conversation_id"):
            payload["conversation_id"] = t["conversation_id"]
        body = invoke(payload)
        resp = (body.get("response") or "").strip()
        dbg = body.get("debug_context") or {}
        low = resp.lower()
        print(f"[{t['label']}]")
        safe = resp[:120].encode("ascii", "replace").decode("ascii")
        print(f"  response: {safe}{'...' if len(resp) > 120 else ''}")
        print(f"  state={dbg.get('daisy_state')} build={dbg.get('inference_build')} path={dbg.get('translation_path')}")

        for sub in t.get("forbid_substrings", []):
            if sub in low:
                issues.append(f"{t['label']}: OLD signature {sub!r} still present")
        for exact in t.get("forbid_exact", []):
            if resp == exact:
                issues.append(f"{t['label']}: hollow exact match {exact!r}")
        if t.get("require_question") and "?" not in resp:
            issues.append(f"{t['label']}: no question in reply")
        if t.get("min_len") and len(resp) < t["min_len"]:
            issues.append(f"{t['label']}: reply too short ({len(resp)} chars)")
        if t.get("require_majority_cyrillic"):
            latin, cyr = _count_script_chars(resp)
            if cyr <= latin:
                issues.append(
                    f"{t['label']}: reply not majority Cyrillic (latin={latin}, cyr={cyr})"
                )

        if t.get("require_topic_terms"):
            if not any(term in low for term in t["require_topic_terms"]):
                issues.append(
                    f"{t['label']}: reply lacks user topic terms {t['require_topic_terms']}"
                )

        if t.get("require_voice_retry"):
            vrc = dbg.get("voice_retry_count")
            if vrc is not None and vrc < 1:
                issues.append(f"{t['label']}: voice_retry_count=0 (voice QC did not run)")

        build = dbg.get("inference_build") or ""
        if build and PRE_JUNE_BUILD not in build:
            issues.append(f"{t['label']}: expected INFERENCE_BUILD={PRE_JUNE_BUILD!r}, got {build!r}")

        if dbg.get("brief_retry_count", 0) >= 2:
            signals_new.append(f"{t['label']}: brief_retry_count>=2 (multi-regen path)")
        if OLD_BODY_SCAN not in low and OLD_TEMPLATE_OPENING not in low:
            signals_new.append(f"{t['label']}: no old body-scan template")

    print("\n=== RU finance anxiety thread (screenshot repro) ===")
    thread_turns = [
        "мне тревожно сегодня",
        "мне нужно разобраться почему",
        "я боюсь что останусь без денег…",
        "с финансов",
    ]
    thread_history: list[dict] = []
    conv_id = "verify-ru-finance-thread"
    for i, msg in enumerate(thread_turns):
        body = invoke({
            "message": msg,
            "locale": "ru",
            "history": thread_history,
            "max_tokens": 256,
            "conversation_id": conv_id,
        })
        resp = (body.get("response") or "").strip()
        dbg = body.get("debug_context") or {}
        safe = resp[:100].encode("ascii", "replace").decode("ascii")
        print(f"  turn{i + 1}: {safe}{'…' if len(resp) > 100 else ''}")
        low = resp.lower()
        for tail in CANNED_RU_TAILS:
            if tail in low:
                issues.append(f"ru_finance_thread turn{i + 1}: canned tail {tail!r}")
        if "? : . :" in resp or "может быть,?" in low:
            issues.append(f"ru_finance_thread turn{i + 1}: degenerate punctuation garbage")
        if i == 0:
            pass  # turn1: canned-tail and degenerate checks above are sufficient
        for ch in resp:
            if "\u4e00" <= ch <= "\u9fff":
                issues.append(f"ru_finance_thread turn{i + 1}: CJK leak in RU reply")
                break
        if i == 3:
            latin, cyr = _count_script_chars(resp)
            if latin > cyr and latin >= 12:
                issues.append(f"ru_finance_thread turn{i + 1}: English leak in RU reply")
            if not any(w in low for w in ("финанс", "денег", "деньг", "тревог", "боюсь", "страх", "волн", "беспоко")):
                issues.append(
                    "ru_finance_thread turn4: reply ignores finance/money topic after user said 'с финансов'"
                )
            if "с чего бы тебе хотелось начать" in low:
                issues.append("ru_finance_thread turn4: generic canned opener after finance disclosure")
        thread_history.append({"role": "user", "content": msg})
        thread_history.append({"role": "assistant", "content": resp})

    print("\n=== Cross-chat diversity (anxiety) ===")
    cross_msg = "I'm feeling anxious today"
    cross_a = invoke({
        "message": cross_msg,
        "locale": "en",
        "history": [],
        "max_tokens": 256,
        "conversation_id": "verify-cross-chat-a",
    })
    cross_b = invoke({
        "message": cross_msg,
        "locale": "en",
        "history": [],
        "max_tokens": 256,
        "conversation_id": "verify-cross-chat-b",
    })
    resp_a = (cross_a.get("response") or "").strip()
    resp_b = (cross_b.get("response") or "").strip()
    print(f"  chat_a: {resp_a[:100]}{'…' if len(resp_a) > 100 else ''}")
    print(f"  chat_b: {resp_b[:100]}{'…' if len(resp_b) > 100 else ''}")
    if resp_a and resp_a == resp_b:
        signals_new.append("anxiety_cross_chat: identical responses (informational)")

    print()
    print("=== New-code signals ===")
    for s in signals_new:
        print(f"  + {s}")

    if issues:
        print()
        print("FAIL — possible OLD code or regressions:")
        for x in issues:
            print(f"  - {x}")
        sys.exit(1)

    print()
    print("PASS — live endpoint behavior matches post-fix inference expectations.")


if __name__ == "__main__":
    main()
