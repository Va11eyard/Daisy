"""Ad-hoc validation for the synthesized JSONL batch."""
from __future__ import annotations

import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

files = [
    "data/synthesized/relationships_rus.jsonl",
    "data/synthesized/violence_rus.jsonl",
    "data/synthesized/burnout_rus.jsonl",
    "data/synthesized/empathy_rus.jsonl",
    "data/synthesized/emergency_cases.jsonl",
    "data/synthesized/avoidance_anxious_eng.jsonl",
    "data/synthesized/psychoanalysis_depression_eng.jsonl",
    "data/synthesized/act_eng.jsonl",
    "data/synthesized/relationships_eng.jsonl",
    "data/synthesized/existential_crises_rus.jsonl",
    "data/synthesized/emergency_cases_augmented.jsonl",
]

BANNED_EN = [
    "That makes so much sense",
    "Absolutely!",
    "I hear you",
    "That's so valid",
    "I completely understand",
    "sort of",
    "kind of",
    "Does that make sense?",
    "It sounds like you're going through a lot",
    "That must be really tough",
    "I'm here for you",
    "Take care!",
]
BANNED_RU = [
    "конечно",
    "это звучит тяжело",
    "я тебя слышу",
    "это так понятно",
    "я тебя полностью понимаю",
    "как бы",
    "вроде",
    "типа",
    "понятно?",
    "я рядом",
    "береги себя",
]

LIMITS = {
    "disclosure": (2, 4),
    "psychoeducation": (4, 8),
    "action_planning": (3, 6),
    "crisis": (1, 5),
}
Q_RULES = {
    "disclosure": (1, 1),
    "psychoeducation": (1, 1),
    "action_planning": (1, 1),
    "crisis": (0, 1),
}

total = 0
issues = 0
for f in files:
    with open(f, encoding="utf-8") as fh:
        lines = fh.readlines()
    state_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"PARSE ERROR {f}:{i}: {e}")
            issues += 1
            continue
        total += 1
        assistant = [m["content"] for m in obj["messages"] if m["role"] == "assistant"][0]
        state = obj["meta"]["state"]
        lang = obj["meta"]["language"]
        state_counts[state] = state_counts.get(state, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        lower = assistant.lower()
        hits = [b for b in (BANNED_RU if lang == "ru" else BANNED_EN) if b.lower() in lower]
        if hits:
            print(f"BANNED HIT {f}:{i} state={state} -> {hits}")
            issues += 1
        qcount = assistant.count("?")
        qmin, qmax = Q_RULES[state]
        if not (qmin <= qcount <= qmax):
            print(f"Q COUNT {f}:{i} state={state} qcount={qcount}")
            issues += 1
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", assistant.strip()) if s.strip()]
        sc = len(sentences)
        lim = LIMITS[state]
        if not (lim[0] <= sc <= lim[1]):
            print(f"SENTENCE COUNT {f}:{i} state={state} sc={sc} limit={lim}")
            issues += 1
    print(f"{f}: {len(lines)} lines, states={state_counts}, langs={lang_counts}")

print(f"TOTAL: {total} examples, {issues} issues")
