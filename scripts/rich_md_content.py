"""
Rich dialogue synthesis from MD chunks (templates, archetypes, section-aware split).

Imported by md_to_dialogues.py when --quality rich.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# --- Archetypes: user intent → varied multi-turn scaffolds ---
ARCHETYPES_RU = (
    "explain",
    "apply",
    "reflect",
    "micro_skill",
    "clarify_terms",
    "gentle_challenge",
)

# Openers: {title} and optional {snippet} (first sentence trimmed)
USER_OPENERS_RU: dict[str, list[str]] = {
    "explain": [
        'Объясни, пожалуйста, по-простому: что важно в «{title}»? Я читаю и путаюсь.',
        'Можешь своими словами разложить главную мысль про «{title}»?',
        'Что автор хочет донести в части про «{title}»? Без канцелярита.',
        'Я не понимаю терминологию. В двух словах: о чём речь в «{title}»?',
    ],
    "apply": [
        'Как это из «{title}» можно применить к обычной жизни, без фанатизма?',
        'Если представить себя в сложный день — что из этого текста реально поможет?',
        'Хочу маленький шаг на основе «{title}». Что попробовать в ближайшие дни?',
        'Как перенести идеи из «{title}» на отношения / работу / сон — что уместно?',
    ],
    "reflect": [
        'Мне тревожно, и я ищу опору. Что в этом фрагменте может меня поддержать?',
        'Читаю и ловлю стыд/злость. Как это соотносится с тем, что здесь написано?',
        'Помоги отразить: что я могу почувствовать, если это про меня — без давления.',
        'Хочу понять свою реакцию. На что здесь стоит обратить внимание?',
    ],
    "micro_skill": [
        'Дай один конкретный навык или формулировку из этого материала — чтобы попробовать сегодня.',
        'Какое одно упражнение или приём здесь уместен, если времени мало?',
        'Выдели микро-шаг: что сделать 3–5 минут по этому тексту?',
    ],
    "clarify_terms": [
        'Расшифруй ключевые понятия из этого куска — как для умного друга, не как в учебнике.',
        'Что здесь значит главное слово/идея? Свяжи с примером из жизни.',
        'Где здесь тонкое место — чтобы не перепутать с самообвинением?',
    ],
    "gentle_challenge": [
        'Где здесь граница: что полезно заметить, а что не стоит натягивать на себя?',
        'Помоги не идеализировать текст: что важно не упустить в реальной ситуации?',
        'Что из этого — факты из текста, а что — интерпретация? Раздели аккуратно.',
    ],
}

USER_FOLLOWUPS_RU: dict[str, list[str]] = {
    "explain": [
        'Спасибо. Что ещё из этого фрагмента стоит удержать в голове?',
        'Ок. Какая одна формулировка здесь самая сильная для запоминания?',
        'Понятнее. А что автор подразумевает под последствиями / ограничениями?',
    ],
    "apply": [
        'Хорошо. А если сопротивление или «не хочу» — как смягчить шаг?',
        'Если совсем нет сил — какая минимальная версия этого шага?',
        'Как понять, что шаг сработал, а не стал самобичеванием?',
    ],
    "reflect": [
        'Спасибо. Что мне не приписывать себе лишнего из этого текста?',
        'Как отличить нормальную реакцию от того, что уже про угрозу себе/другим?',
    ],
    "micro_skill": [
        'Если не получится с первого раза — как не бросить и не ругать себя?',
    ],
    "clarify_terms": [
        'Есть ли тут слова, которые люди часто понимают неправильно?',
    ],
    "gentle_challenge": [
        'Где здесь границы чата: что мы не делаем на платформе, даже если текст об этом?',
    ],
}

USER_THIRD_RU = [
    'Последнее: если совсем коротко — одно предложение, что взять с собой?',
    'Сожми в одну мысль, которую можно повторить себе при стрессе.',
    'Что бы ты добавила как оговорку — честно и без паники?',
]

ASSIST_PREFIX_RU: dict[str, tuple[str, ...]] = {
    "explain": (
        'Стараюсь говорить просто и без давления. По тексту:\n\n',
        'Держу рядом с тобой тон. Главное из материала:\n\n',
        'Вот как это можно прочитать бережно:\n\n',
    ),
    "apply": (
        'Перенос на жизнь — всегда маленькими шагами. Из текста можно взять:\n\n',
        'Практично и без «надо всё успеть»:\n\n',
    ),
    "reflect": (
        'Твоё состояние здесь важно. Из написанного опираюсь на такое:\n\n',
        'Без оценки «как надо». В тексте есть такая опора:\n\n',
    ),
    "micro_skill": (
        'Один конкретный микро-шаг из этого фрагмента:\n\n',
    ),
    "clarify_terms": (
        'По смыслу терминов, как в разговоре:\n\n',
    ),
    "gentle_challenge": (
        'Аккуратно разделю факты и то, что люди иногда додумывают:\n\n',
    ),
}

# English (compact pools for Eng paths)
USER_OPENERS_EN: dict[str, list[str]] = {
    "explain": [
        'In simple terms, what matters in «{title}»?',
        'Could you unpack the main idea about «{title}» without jargon?',
    ],
    "apply": [
        'How could I apply this to real life in a small, realistic way?',
    ],
    "reflect": [
        'I feel on edge. What in this excerpt might actually help me ground?',
    ],
    "micro_skill": [
        'One concrete skill or phrase to try today from this passage?',
    ],
    "clarify_terms": [
        'Help me decode the key terms here like a thoughtful friend would.',
    ],
    "gentle_challenge": [
        'Where should I be careful not to misuse this material on myself?',
    ],
}

EMERGENCY_ASSISTANT_TAIL_RU = (
    '\n\n_Если речь об угрозе жизни или острой опасности — важна очная помощь '
    '(экстренные службы, кризисная линия). В чате мы не ставим диагнозы и не заменяем очную терапию._'
)


def _stable_pick(seed: str, options: list[str]) -> str:
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def _snippet(chunk: str, max_len: int = 140) -> str:
    t = chunk.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    """Split markdown body by ## / ### headers; returns (section_title, text)."""
    lines = body.split("\n")
    sections: list[tuple[str | None, str]] = []
    cur_title: str | None = None
    buf: list[str] = []

    for line in lines:
        m = re.match(r"^#{2,3}\s+(.+)$", line.strip())
        if m:
            if buf:
                blob = "\n".join(buf).strip()
                if blob:
                    sections.append((cur_title, blob))
            cur_title = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        blob = "\n".join(buf).strip()
        if blob:
            sections.append((cur_title, blob))
    if not sections:
        return [(None, body.strip())]
    return sections


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


def _split_n_turns(chunk: str, n_parts: int) -> list[str]:
    paras = [x.strip() for x in chunk.split("\n\n") if x.strip()]
    if len(paras) < n_parts:
        # split by length
        L = len(chunk)
        if L < n_parts:
            return [chunk]
        step = max(1, L // n_parts)
        return [chunk[i * step : (i + 1) * step].strip() for i in range(n_parts) if chunk[i * step : (i + 1) * step].strip()]
    # distribute paragraphs across n_parts buckets
    out: list[list[str]] = [[] for _ in range(n_parts)]
    for i, p in enumerate(paras):
        out[i % n_parts].append(p)
    return ["\n\n".join(x).strip() for x in out if any(x)]


def _wrap(prefix: str, body: str, max_chars: int, emergency: bool) -> str:
    body = body.strip()
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"
    s = prefix + body
    if emergency:
        s += EMERGENCY_ASSISTANT_TAIL_RU
    return s


def build_rich_record(
    *,
    chunk: str,
    title: str,
    section_title: str | None,
    rel_path: str,
    persona: str,
    locale: str,
    archetype: str,
    max_assistant_chars: int,
    seed: str,
    emergency: bool,
    long_dialog: bool,
) -> dict[str, Any]:
    """One training record: 4 or 6 messages, archetype-driven."""
    head = _snippet(chunk, 100)
    ctx = f"source_md:{rel_path[:180]} | head:{head}"
    if section_title:
        ctx += f" | section:{section_title[:120]}"

    meta: dict[str, Any] = {
        "locale": locale,
        "persona": persona,
        "onboarding_summary": "",
        "user_context": ctx,
    }

    if locale == "en":
        openers = USER_OPENERS_EN.get(archetype, USER_OPENERS_EN["explain"])
        u1 = _stable_pick(seed + "u1", [o.format(title=title) for o in openers])
        u2 = _stable_pick(
            seed + "u2",
            [
                "What practical angle should I not miss from this excerpt?",
                "Anything I should be careful not to misapply to myself?",
                "One more layer — what else matters here?",
            ],
        )
        prefixes = ("Here's a careful reading:\n\n", "In plain language:\n\n")
        pfx_i = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(prefixes)
        parts = _split_for_two(chunk)
        a1 = _wrap(prefixes[pfx_i], parts[0], max_assistant_chars, emergency)
        a2 = _wrap(prefixes[(pfx_i + 1) % len(prefixes)], parts[1], max_assistant_chars, emergency)
        return {
            "messages": [
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ],
            "meta": meta,
        }

    # Russian (default)
    openers = USER_OPENERS_RU.get(archetype, USER_OPENERS_RU["explain"])
    u1 = _stable_pick(seed + "u1", [o.format(title=title) for o in openers])
    follows = USER_FOLLOWUPS_RU.get(archetype, USER_FOLLOWUPS_RU["explain"])
    u2 = _stable_pick(seed + "u2", follows)

    prefixes = ASSIST_PREFIX_RU.get(archetype, ASSIST_PREFIX_RU["explain"])
    pfx_i = int(hashlib.sha256((seed + "pfx").encode()).hexdigest(), 16) % len(prefixes)

    if not long_dialog:
        parts = _split_for_two(chunk)
        a1 = _wrap(prefixes[pfx_i], parts[0], max_assistant_chars, emergency)
        a2 = _wrap(prefixes[(pfx_i + 1) % len(prefixes)], parts[1], max_assistant_chars, emergency)
        return {
            "messages": [
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ],
            "meta": meta,
        }

    # 6-turn: three user-assistant pairs
    thirds = _split_n_turns(chunk, 3)
    if len(thirds) < 3:
        parts = _split_for_two(chunk)
        a1 = _wrap(prefixes[pfx_i], parts[0], max_assistant_chars, emergency)
        a2 = _wrap(prefixes[(pfx_i + 1) % len(prefixes)], parts[1], max_assistant_chars, emergency)
        return {
            "messages": [
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ],
            "meta": meta,
        }

    u3 = _stable_pick(seed + "u3", USER_THIRD_RU)
    a1 = _wrap(prefixes[pfx_i], thirds[0], max_assistant_chars, emergency)
    a2 = _wrap(prefixes[(pfx_i + 1) % len(prefixes)], thirds[1], max_assistant_chars, emergency)
    a3 = _wrap(prefixes[(pfx_i + 2) % len(prefixes)], thirds[2], max_assistant_chars, emergency)
    return {
        "messages": [
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": a2},
            {"role": "user", "content": u3},
            {"role": "assistant", "content": a3},
        ],
        "meta": meta,
    }


def _split_for_two(chunk: str) -> tuple[str, str]:
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
        return a, a
    return a, b


def pick_archetypes_for_chunk(seed: str, count: int) -> list[str]:
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    order = list(ARCHETYPES_RU)
    rotated = order[h % len(order) :] + order[: h % len(order)]
    return rotated[:count]
