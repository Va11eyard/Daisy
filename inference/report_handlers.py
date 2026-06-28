"""weekly_report and dynamics_insights — JSON from model."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from generation import extract_json_object, generate_reply

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def run_dynamics_insights(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    data: dict,
) -> str:
    checkins = data.get("checkins") or []
    period_days = int(data.get("period_days") or 7)
    if not checkins:
        return json.dumps(
            {
                "emotion": "Пройди чек-ин, чтобы получить инсайт об эмоциях 🤍",
                "stress": "Пройди чек-ин, чтобы получить инсайт о стрессе 🤍",
                "energy": "Пройди чек-ин, чтобы получить инсайт об энергии 🤍",
                "support": "Пройди чек-ин, чтобы получить инсайт о поддержке 🤍",
            },
            ensure_ascii=False,
        )

    lines = []
    for c in checkins:
        d = c.get("date", "")
        parts = []
        for k, label in (
            ("emotion", "эмоции"),
            ("stress", "стресс"),
            ("energy", "энергия"),
            ("support", "поддержка"),
        ):
            if c.get(k) is not None:
                parts.append(f"{label}: {c[k]}/5")
        lines.append(f"{d}: {', '.join(parts)}")
    history_text = "\n".join(lines)

    system = (
        "You are Daisy. Analyze check-in data for the period and output ONLY JSON with keys: "
        "emotion, stress, energy, support — each a short warm insight (1-2 sentences)."
    )
    user = f"PERIOD: {period_days} days\nCHECK-INS:\n{history_text}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = generate_reply(model, tokenizer, prompt, max_new_tokens=400, temperature=0.5)
    parsed = extract_json_object(raw)
    if parsed:
        return json.dumps(
            {
                "emotion": str(parsed.get("emotion", "")),
                "stress": str(parsed.get("stress", "")),
                "energy": str(parsed.get("energy", "")),
                "support": str(parsed.get("support", "")),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "emotion": "Эмоциональный фон меняется, продолжай наблюдать 🤍",
            "stress": "Обрати внимание на стресс-триггеры.",
            "energy": "Давай себе отдых, когда энергия ниже.",
            "support": "Поддержка важна — не стесняйся просить о ней.",
        },
        ensure_ascii=False,
    )


def run_weekly_report(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    data: dict,
) -> str:
    checkins = data.get("checkins") or []
    period_days = int(data.get("period_days") or 7)
    profile = data.get("profile") or {}
    memory_topics = data.get("memory_topics") or []

    lines = []
    for c in checkins:
        d = c.get("date", "")
        parts = []
        for k, label in (
            ("emotion", "эмоции"),
            ("stress", "стресс"),
            ("energy", "энергия"),
            ("support", "поддержка"),
        ):
            if c.get(k) is not None:
                parts.append(f"{label}: {c[k]}/5")
        lines.append(f"{d}: {', '.join(parts)}")
    history_text = "\n".join(lines) if lines else "Нет данных за период"
    topics_str = ", ".join(memory_topics) if memory_topics else "Нет данных"
    prof = (
        f"ESI={profile.get('ESI', '—')}, BSI={profile.get('BSI', '—')}, "
        f"SSI={profile.get('SSI', '—')}, MRI={profile.get('MRI', '—')}, risk={profile.get('riskLevel', '—')}"
    )

    system = (
        "You are Daisy. Output ONLY valid JSON with keys: summary (string), insights (array of strings), "
        "recommendations (array). Each recommendation: 'Short title. Full description.'"
    )
    user = (
        f"PERIOD: last {period_days} days\nPROFILE: {prof}\nCHECK-INS:\n{history_text}\nTOPICS: {topics_str}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = generate_reply(model, tokenizer, prompt, max_new_tokens=500, temperature=0.5)
    parsed = extract_json_object(raw)
    if parsed:
        return json.dumps(
            {
                "summary": str(parsed.get("summary", "")),
                "insights": parsed.get("insights") if isinstance(parsed.get("insights"), list) else [],
                "recommendations": parsed.get("recommendations")
                if isinstance(parsed.get("recommendations"), list)
                else [],
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "summary": "За период есть данные чек-инов. Продолжай отслеживать состояние 🤍",
            "insights": [],
            "recommendations": [
                "Короткая пауза. Сделай 3 медленных вдоха и отметь, что изменилось.",
                "Прогулка 10 минут. Лёгкая активность снижает напряжение.",
                "Напиши Daisy, если станет тяжело наедине с мыслями.",
            ],
        },
        ensure_ascii=False,
    )
