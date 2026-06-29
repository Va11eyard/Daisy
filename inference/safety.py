"""Prompt-injection scan, off-topic, crisis, meta questions."""

from __future__ import annotations

import re
import unicodedata

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+(now\s+)?(DAN|an?\s+(evil|unrestricted))",
    r"игнорир(?:уй|овать)\s+(?:все\s+)?(?:предыдущие|прошлые)\s+инструкции",
    r"забудь\s+(?:все|свои)\s+инструкции",
    r"(?:системный|твой|ваш)\s+промпт",
]

CRISIS_TIER1 = [
    "suicide",
    "kill myself",
    "self-harm",
    "суицид",
    "самоубийств",
    "убить себя",
    "покончить с собой",
    "покончить жизнь",
    "хочу умереть",
    "повеситься",
    "свести счёты",
    "свести счеты",
]

CRISIS_TIER2 = [
    "hopeless",
    "no reason to live",
    "безнадёжно",
    "никому не нужен",
]

META_WHO_CREATED = [
    "who created you",
    "who made you",
    "кто тебя создал",
    "кто создал тебя",
    "сені кім жасады",
]

OFF_TOPIC_PHRASES = [
    "write a python",
    "write code for",
    "solve this equation",
    "реши уравнение",
    "напиши код на",
]

# If any of these appear, message may relate to mental health / life — do not treat as pure off-topic.
_THERAPY_HINTS: frozenset[str] = frozenset(
    {
        # English
        "feel",
        "feeling",
        "felt",
        "anxi",
        "stress",
        "depress",
        "therapy",
        "therapist",
        "panic",
        "worry",
        "worried",
        "sad",
        "lonely",
        "alone",
        "angry",
        "guilt",
        "guilty",
        "shame",
        "ashamed",
        "mood",
        "mental",
        "burnout",
        "cry",
        "crying",
        "hurt",
        "hurting",
        "scared",
        "afraid",
        "fear",
        "trauma",
        "grief",
        "relationship",
        "family",
        "marriage",
        "partner",
        "breakup",
        "divorce",
        "sleep",
        "insomnia",
        "nightmare",
        "emotion",
        "overwhelm",
        "hopeless",
        "suicid",
        "self-harm",
        "addict",
        "addiction",
        "eating",
        "body image",
        "self-esteem",
        "abuse",
        "assault",
        "ptsd",
        "ocd",
        "adhd",
        "bipolar",
        "meds",
        "medication",
        "counsel",
        "session",
        "trigger",
        "cope",
        "coping",
        "overthink",
        "racing thoughts",
        # Russian
        "чувств",
        "тревог",
        "стресс",
        "депресс",
        "груст",
        "одиноч",
        "страх",
        "бессон",
        "отношения",
        "семь",
        "семье",
        "терап",
        "псих",
        "психолог",
        "боль",
        "устал",
        "выгоран",
        "паник",
        "стыд",
        "вина",
        "злост",
        "обид",
        "настроен",
        "эмоц",
        "суицид",
        "самоубий",
        "насили",
        "травм",
        "зависим",
        "расстрой",
        "сон",
        "кошмар",
        "плач",
        "плакать",
        "ревность",
        "ревную",
        "любов",
        "расставан",
        "развод",
        # Kazakh (common stems)
        "қиын",
        "қорқы",
        "үйқы",
        "маза",
        "қайғы",
        "тревож",
    }
)

# Strong off-topic request patterns (only applied when therapy_relevant() is false).
_OUT_OF_SCOPE_RES: list[re.Pattern[str]] = [
    re.compile(r"\b(?:give|give me|share|send|write)\s+(?:me\s+)?.*?\brecipe\b", re.I),
    re.compile(r"\brecipe\s+for\s+(?:[\w-]+\s+){0,3}(?:pasta|cake|bread|chicken|soup|cookies?)\b", re.I),
    re.compile(r"\b(?:chocolate|vanilla|cheese)?\s*cake\s+recipe\b", re.I),
    re.compile(r"\b(?:how\s+to\s+cook|how\s+to\s+bake|how\s+to\s+make)\s+(?:a\s+)?(?:pasta|cake|soup)\b", re.I),
    re.compile(r"\bрецепт\b.*\b(?:торта|супа|салата|борща|печенья|пирога|блюда)\b", re.I),
    re.compile(r"\b(?:как\s+приготовить|как\s+сварить|как\s+испечь)\b", re.I),
    re.compile(r"\b(?:walkthrough|speedrun|cheat\s*code|achievement\s+guide)\b", re.I),
    re.compile(r"\b(?:чит-код|прохождение\s+игры|как\s+пройти\s+уровень|как\s+пройти\s+миссию)\b", re.I),
    re.compile(r"\b(?:minecraft|fortnite|league of legends|genshin)\b.*\b(?:how\s+to|guide|build)\b", re.I),
    re.compile(r"\bwho\s+won\s+the\s+(?:super\s+bowl|world\s+cup)\b", re.I),
    re.compile(r"\b(?:stock\s+price|crypto|bitcoin)\s+(?:today|prediction)\b", re.I),
    re.compile(r"\b(?:solve|find)\s+(?:the\s+)?(?:integral|derivative|equation)\b", re.I),
    re.compile(r"\b(?:домашн(?:ее|ее)\s+задани|дз\s+по)\s+(?:математик|физик|химии)\b", re.I),
]


def normalize_for_scan(text: str) -> str:
    text = re.sub(r"[\u200B-\u200D\uFEFF\u00AD]", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def therapy_relevant(t: str) -> bool:
    """True if message likely touches wellbeing / life context (do not treat as trivial off-topic)."""
    tl = t.lower()
    return any(h in tl for h in _THERAPY_HINTS)


def sanitize_user_input(text: str) -> str:
    n = normalize_for_scan(text)
    for pat in INJECTION_PATTERNS:
        if re.search(pat, n, re.IGNORECASE):
            return "[Message was blocked due to unsafe content.]"
    return text


def contains_injection(text: str) -> bool:
    n = normalize_for_scan(text or "")
    return any(re.search(pat, n, re.IGNORECASE) for pat in INJECTION_PATTERNS)


def sanitize_prompt_field(text: str) -> str:
    """Strip injection patterns from context fields; empty string if blocked."""
    if not text:
        return ""
    if contains_injection(str(text)):
        return ""
    return str(text).strip()


def normalize_history(history: object, *, max_turns: int = 50) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    out: list[dict[str, str]] = []
    for turn in history[:max_turns]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue
        content = turn.get("content") or turn.get("text") or ""
        content = str(content)[:4000]
        if role == "user":
            content = sanitize_user_input(content)
        out.append({"role": role, "content": content})
    return out


def is_meta_question(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(p in t for p in META_WHO_CREATED)


def _matches_out_of_scope_request(t: str) -> bool:
    return any(r.search(t) for r in _OUT_OF_SCOPE_RES)


def is_off_topic(text: str) -> bool:
    """
    True for coding/math homework requests and clear non-therapy requests (recipes, game guides, etc.)
    when the message does not appear therapy-related. Emotional or clinical keywords disable blocking.
    """
    raw = text or ""
    t = normalize_for_scan(raw).lower()
    if len(t) < 8:
        return False
    if therapy_relevant(t):
        return False
    if is_meta_question(raw):
        return False
    if crisis_tier(raw):
        return False

    if any(p in t for p in OFF_TOPIC_PHRASES):
        legacy_life = (
            "feel",
            "stress",
            "anxi",
            "чувств",
            "тревог",
            "emotion",
            "relationship",
            "family",
            "work",
            "одиноч",
            "груст",
        )
        if any(k in t for k in legacy_life):
            return False
        return True

    if _matches_out_of_scope_request(t):
        return True

    return False


def crisis_tier(text: str) -> int:
    t = (text or "").lower()
    for p in CRISIS_TIER1:
        if p in t:
            return 1
    for p in CRISIS_TIER2:
        if p in t:
            return 2
    return 0
