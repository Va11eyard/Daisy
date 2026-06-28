"""
Synthetic client turns and path→modality/persona heuristics for book→dialogue SFT.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

# --- Modality / persona from folder path (best-effort) ---

_MODALITY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("cbt", "кпт", "cbt "), "CBT"),
    (("dbt", "дпт"), "DBT"),
    (("act ", "/act", "\\act", "act\\", "act/"), "ACT"),
    (("cft", "сострадан", "compassion"), "CFT/Compassion-focused"),
    (("mindfulness", "mbct", "mbsr", "осознан"), "Mindfulness/MBCT"),
    (("act и cft", "act и cft"), "ACT/CFT"),
    (("psychoanalysis", "психоанал", "psychodynamic"), "Psychodynamic"),
    (("emergency", "кризис", "psychosocial"), "Crisis/psychosocial"),
    (("anxiety", "тревог", "panic"), "Anxiety"),
    (("depression", "депресс"), "Depression"),
    (("adhd", "сдвг"), "ADHD"),
    (("bpd", "погранич"), "BPD"),
    (("sexual", "couple", "relationships"), "Relationships/intimacy"),
    (("attachment", "привязан"), "Attachment"),
]


def infer_modality(path: Path) -> str | None:
    s = path.as_posix().lower()
    for needles, label in _MODALITY_RULES:
        if any(n in s for n in needles):
            return label
    return None


def persona_for_modality(modality: str | None) -> str:
    if not modality:
        return "flexible"
    m = modality.lower()
    if "cbt" in m or "dbt" in m:
        return "wise_teacher"
    if "act" in m or "cft" in m or "mindfulness" in m or "compassion" in m:
        return "gentle_explorer"
    if "crisis" in m or "psychosocial" in m or "anxiety" in m:
        return "practical_helper"
    if "psychodynamic" in m or "psycho" in m:
        return "calm_mentor"
    if "relationship" in m or "attachment" in m or "intimacy" in m:
        return "warm_friend"
    return "flexible"


# --- Scenario types ---

SCENARIOS = ("psychoeducation", "technique", "self_apply", "reflect")


# Minimal Kazakh mix (often used alongside Russian in clinical settings)
CLIENT_OPENERS_KK_FALLBACK: list[str] = [
    "Маған қарапайым тілмен түсіндірсеңіз, күрделі терминсіз.",
    "Бұл идеяны практикада қалай қолдануға болады?",
]

# Russian client openers (realistic, not book-meta)
CLIENT_OPENERS_RU: dict[str, list[str]] = {
    "psychoeducation": [
        "Я много читаю про терапию, но в жизни не понимаю, как это со мной стыкуется. Можешь объяснить по-человечески?",
        "Мне нужно, чтобы ты без сложного жаргона разложил(а) это на сессии — я устал(а) от умных слов.",
        "Объясни, как это понимает в клинической практике: что здесь главное для клиента?",
        "Я теряюсь в терминах. Помоги увидеть суть этой идеи так, как будто я на приёме.",
        "Что из этого фрагмента реально важно донести человеку, который не специалист?",
    ],
    "technique": [
        "Как мне превратить это в конкретный шаг или упражнение на неделю, не перегружая себя?",
        "Если я хочу попробовать это на практике — с чего начать в микро-порциях?",
        "Как бы ты сформулировал(а) это как домашнее задание или эксперимент между сессиями?",
        "Какие формулировки и вопросы здесь уместны, чтобы клиент не сопротивлялся?",
    ],
    "self_apply": [
        "У меня срывается на близких и я не могу остановить поток мыслей — как это соотносится с тем, что здесь написано?",
        "Я чувствую вину и стыд, и мне нужно понять, как это обрабатывать бережно.",
        "Мне страшно, что я «делаю всё не так». Помоги разобраться через этот материал.",
        "Я в выгорании и не верю, что что-то поможет — но хочу разобраться в этой идее.",
    ],
    "reflect": [
        "Какие уточняющие вопросы ты бы задал(а) клиенту после этого куска?",
        "Как бы ты сказал(а) это мягко, без давления, но по делу?",
        "Что здесь важно отразить, чтобы человек почувствовал себя услышанным?",
    ],
}

CLIENT_OPENERS_EN: dict[str, list[str]] = {
    "psychoeducation": [
        "I read a lot about therapy but struggle to connect it to my life. Can you explain this in plain language?",
        "Please unpack this without jargon — I need clarity, not buzzwords.",
        "What matters most here for a client in session, in practical terms?",
        "Help me see the core idea as if I were in your office.",
    ],
    "technique": [
        "How would I turn this into a small weekly experiment without overwhelming myself?",
        "What would a concrete first step look like, based on this?",
        "How would you phrase a between-session task so it feels doable?",
    ],
    "self_apply": [
        "I spiral with worry and guilt — how does this passage apply to someone like me?",
        "I feel exhausted and skeptical; help me understand this idea gently.",
    ],
    "reflect": [
        "What follow-up questions would you ask after this?",
        "How would you say this gently in session?",
    ],
}

CLIENT_FOLLOWUP_RU: list[str] = [
    "Можешь один конкретный шаг на эту неделю без морали?",
    "А что если это не сработает — как тогда переговорить с собой без самокритики?",
    "Как это увязать с тем, что я уже пробовал(а)?",
    "Коротко: что мне запомнить из этого ответа на сегодня?",
]

CLIENT_FOLLOWUP_EN: list[str] = [
    "Can you give one concrete step for this week?",
    "If this doesn’t work, how would I adjust without self-attack?",
    "What’s the one takeaway I should remember today?",
]


def pick_scenario(chunk: str, rng: random.Random) -> str:
    _ = chunk
    return rng.choice(SCENARIOS)


def pick_client_opener(locale: str, scenario: str, rng: random.Random) -> str:
    if locale == "kk":
        pool = CLIENT_OPENERS_RU.get(scenario, CLIENT_OPENERS_RU["psychoeducation"]) + CLIENT_OPENERS_KK_FALLBACK
    elif locale == "ru":
        pool = CLIENT_OPENERS_RU.get(scenario, CLIENT_OPENERS_RU["psychoeducation"])
    else:
        pool = CLIENT_OPENERS_EN.get(scenario, CLIENT_OPENERS_EN["psychoeducation"])
    return rng.choice(pool)


def pick_followup(locale: str, rng: random.Random) -> str:
    if locale == "ru" or locale == "kk":
        return rng.choice(CLIENT_FOLLOWUP_RU)
    return rng.choice(CLIENT_FOLLOWUP_EN)


def split_chunk_at_sentences(chunk: str, max_first: float = 0.48) -> tuple[str, str] | None:
    """Split into two substantial parts at sentence boundaries; None if not viable."""
    chunk = chunk.strip()
    if len(chunk) < 900:
        return None
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", chunk)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 4:
        return None
    n = len(parts)
    cut = max(2, min(n - 2, int(n * max_first)))
    first = " ".join(parts[:cut])
    second = " ".join(parts[cut:])
    if len(first) < 280 or len(second) < 280:
        return None
    return first, second
