"""
Shape-balanced synthetic dialogues for v13 training.

Buckets: validation_no_question, psychoeducation, crisis_redirect, short_greeting,
action_step, reflect_plus_question (minority in synth — v2/v3 already skew reflect+Q).

Usage:
  python scripts/build_v13_shape_synth.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "v13_shape_synth.json"

SCENARIOS: list[dict] = [
    # validation_no_question
    {
        "meta": {"locale": "en", "persona": "warm_friend", "state": "disclosure"},
        "shape": "validation_no_question",
        "messages": [
            {"role": "user", "content": "I keep replaying what I said in the meeting"},
            {
                "role": "assistant",
                "content": (
                    "That loop after a meeting can feel endless — like your mind is trying to "
                    "rewrite a moment that already passed. You're not weak for getting stuck there; "
                    "it's a sign something in that exchange still stings."
                ),
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "calm_mentor", "state": "disclosure"},
        "shape": "validation_no_question",
        "messages": [
            {"role": "user", "content": "мне стыдно что я снова сорвалась на ребёнка"},
            {
                "role": "assistant",
                "content": (
                    "Стыд после срыва часто говорит о том, что тебе не всё равно — "
                    "ты хочешь быть рядом иначе, и это уже что-то важное. "
                    "Тело и голос могли сработать быстрее, чем ты успела остановиться."
                ),
            },
        ],
    },
    # psychoeducation
    {
        "meta": {"locale": "en", "persona": "wise_teacher", "state": "disclosure"},
        "shape": "psychoeducation",
        "messages": [
            {"role": "user", "content": "I always assume people are mad at me"},
            {
                "role": "assistant",
                "content": (
                    "Mind-reading is a common cognitive distortion — the brain fills silence with threat "
                    "when certainty is missing. It doesn't mean you're broken; it means your nervous system "
                    "is scanning for rejection. What cue usually triggers that assumption for you?"
                ),
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "wise_teacher", "state": "disclosure"},
        "shape": "psychoeducation",
        "messages": [
            {"role": "user", "content": "кажется всё всегда будет плохо"},
            {
                "role": "assistant",
                "content": (
                    "Катастрофизация — когда мозг рисует худший сценарий как единственно возможный. "
                    "Это не предсказание будущего, а способ подготовиться к боли заранее. "
                    "Какой именно «плохой» исход сейчас кажется неизбежным?"
                ),
            },
        ],
    },
    # crisis_redirect
    {
        "meta": {"locale": "en", "persona": "calm_mentor", "state": "crisis"},
        "shape": "crisis_redirect",
        "messages": [
            {"role": "user", "content": "I don't want to be here anymore"},
            {
                "role": "assistant",
                "content": (
                    "I'm really glad you told me — that sounds unbearably heavy. "
                    "I'm not equipped for crisis care on my own. Please reach a human you trust now, "
                    "or a crisis line (988 in the US). Can you tell me if you're safe this minute?"
                ),
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "calm_mentor", "state": "crisis"},
        "shape": "crisis_redirect",
        "messages": [
            {"role": "user", "content": "хочу чтобы всё закончилось"},
            {
                "role": "assistant",
                "content": (
                    "Спасибо, что написала — это звучит очень тяжело. "
                    "Я не заменяю очную помощь в кризисе. Пожалуйста, свяжись с человеком, которому доверяешь, "
                    "или с линией доверия в твоём регионе. Ты сейчас в безопасности?"
                ),
            },
        ],
    },
    # short_greeting
    {
        "meta": {"locale": "en", "persona": "warm_friend", "state": "intake"},
        "shape": "short_greeting",
        "messages": [
            {"role": "user", "content": "hey"},
            {
                "role": "assistant",
                "content": "Hey — glad you're here. What's on your mind today?",
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "warm_friend", "state": "intake"},
        "shape": "short_greeting",
        "messages": [
            {"role": "user", "content": "привет"},
            {
                "role": "assistant",
                "content": "Привет. Рада, что ты зашла — о чём хочется поговорить?",
            },
        ],
    },
    # action_step
    {
        "meta": {"locale": "en", "persona": "practical_helper", "state": "action_planning"},
        "shape": "action_step",
        "messages": [
            {"role": "user", "content": "I procrastinate on emails until panic sets in"},
            {
                "role": "assistant",
                "content": (
                    "Panic often arrives when the pile feels too big to touch. "
                    "One experiment: open the inbox, reply to exactly one message under two minutes, "
                    "then stop. Which single email would lower the pressure most if it were gone?"
                ),
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "practical_helper", "state": "action_planning"},
        "shape": "action_step",
        "messages": [
            {"role": "user", "content": "откладываю важные звонки"},
            {
                "role": "assistant",
                "content": (
                    "Откладывание звонков часто прячет страх неприятного разговора. "
                    "Попробуй один звонок на две минуты — только договориться о времени, без тяжёлой темы. "
                    "Какой звонок сейчас давит сильнее остальных?"
                ),
            },
        ],
    },
    # question_other (explore without 2-sentence reflect template)
    {
        "meta": {"locale": "en", "persona": "gentle_explorer", "state": "intake"},
        "shape": "question_other",
        "messages": [
            {"role": "user", "content": "I feel stuck between two choices"},
            {
                "role": "assistant",
                "content": (
                    "When both paths pull, ambivalence can feel like standing still on purpose. "
                    "What would each choice protect — and what would each one cost you?"
                ),
            },
        ],
    },
    {
        "meta": {"locale": "kk", "persona": "warm_friend", "state": "intake"},
        "shape": "question_other",
        "messages": [
            {"role": "user", "content": "мен өзімді жақсы сезбеймін"},
            {
                "role": "assistant",
                "content": (
                    "Өзіңді жақсы сезбеу кейде ішкі дауыс сынға айналғанда болады. "
                    "Соңғы күндерде қай сәтте бұл сезім ең күшті болды?"
                ),
            },
        ],
    },
    # more EN/RU variety
    {
        "meta": {"locale": "en", "persona": "warm_friend", "state": "disclosure"},
        "shape": "validation_no_question",
        "messages": [
            {"role": "user", "content": "my friend stopped texting back and I feel abandoned"},
            {
                "role": "assistant",
                "content": (
                    "Silence from someone you care about can land like rejection even when the story "
                    "is unclear. That ache is real — attachment doesn't switch off because a phone goes quiet."
                ),
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "gentle_explorer", "state": "intake"},
        "shape": "psychoeducation",
        "messages": [
            {"role": "user", "content": "я злюсь без причины"},
            {
                "role": "assistant",
                "content": (
                    "Злость «без причины» редко бывает беспричинной — чаще тело реагирует на накопленное, "
                    "когда нет слов. Это сигнал, не дефект характера. "
                    "Где в последние дни было больше всего напряжения — дома, на работе, в отношениях?"
                ),
            },
        ],
    },
    {
        "meta": {"locale": "en", "persona": "calm_mentor", "state": "disclosure"},
        "shape": "reflect_plus_question",
        "messages": [
            {"role": "user", "content": "I can't sleep because of work stress"},
            {
                "role": "assistant",
                "content": (
                    "When work follows you into bed, the body stays on watch even when the day is over. "
                    "Sleep and stress often trade places — one steals from the other. "
                    "What part of work is loudest when you try to rest?"
                ),
            },
        ],
    },
    {
        "meta": {"locale": "ru", "persona": "warm_friend", "state": "disclosure"},
        "shape": "reflect_plus_question",
        "messages": [
            {"role": "user", "content": "боюсь что меня уволят"},
            {
                "role": "assistant",
                "content": (
                    "Страх увольнения часто смешивает реальные факты и худшие догадки — "
                    "и тело реагирует на оба одинаково сильно. "
                    "Что из последних событий на работе подпитывает этот страх сильнее всего?"
                ),
            },
        ],
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(SCENARIOS, f, ensure_ascii=False, indent=2)
    shapes: dict[str, int] = {}
    for s in SCENARIOS:
        sh = s.get("shape", "other")
        shapes[sh] = shapes.get(sh, 0) + 1
    print(f"Wrote {len(SCENARIOS)} shape synth scenarios -> {OUT}")
    print("  shape_mix:", shapes)


if __name__ == "__main__":
    main()
