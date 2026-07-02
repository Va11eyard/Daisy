"""Generate eval/cross_topic_regression.jsonl (64 cases)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cases: list[dict] = []


def add(cluster: str, locale: str, msg: str, keywords: tuple[str, ...], **extra: object) -> None:
    n = len([c for c in cases if c.get("cluster") == cluster and c.get("locale") == locale]) + 1
    c: dict = {
        "id": f"{cluster}_{locale}_{n}",
        "cluster": cluster,
        "locale": locale,
        "message": msg,
        "keywords": keywords,
        "history": [],
    }
    c.update(extra)
    cases.append(c)


# breakup
for msg, kw in [
    ("we broke up last month and I still cant stop thinking about him", ("breakup", "miss", "think", "left", "feel")),
    ("my boyfriend left me and my heart feels empty", ("breakup", "empty", "left", "miss", "feel")),
    ("after the breakup I miss him even though I was angry at first", ("breakup", "miss", "angry", "feel")),
    ("I ended things but now I regret it and cant sleep", ("breakup", "regret", "sleep", "ended", "feel")),
]:
    add("breakup", "en", msg, kw)
for msg, kw in [
    ("привет я рассталась с парнем", ("расста", "парн", "чувств", "разрыв", "тяжел")),
    ("привет я рассталась с парнем и чувствую пустоту внутри", ("пуст", "расста", "чувств", "внутр")),
    ("мы расстались и я не могу перестать о нем думать", ("расста", "дум", "перест", "чувств")),
    ("после расставания тоскую хотя сначала была злость", ("расста", "тоск", "зл", "чувств")),
]:
    add("breakup", "ru", msg, kw, user_gender="female")

# work
for msg, kw in [
    ("my ceo is raging at me every day", ("ceo", "boss", "work", "stress", "angry")),
    ("My boss keeps yelling at me and I cant take it anymore", ("boss", "yell", "work", "stress")),
    ("Work has been crushing me my manager is furious every day", ("work", "manager", "crush", "furious")),
    ("I might get fired and Im panicking about money", ("fired", "panic", "money", "work", "job")),
]:
    add("work", "en", msg, kw)
for msg, kw in [
    ("на работе всё давит начальник орёт я выгораю", ("работ", "начальник", "дав", "выгор", "стресс")),
    ("руководитель каждый день на меня срывается сил нет", ("руковод", "срыв", "сил", "работ")),
    ("меня могут уволить и я в панике", ("увол", "паник", "работ", "деньг")),
    ("коллеги токсичные а начальник требует невозможного", ("коллег", "начальник", "токс", "работ")),
]:
    add("work", "ru", msg, kw)

# anxiety
for msg, kw in [
    ("Im feeling anxious today", ("anxious", "anxiety", "today", "feel", "worry")),
    ("my chest is tight and I cant calm down", ("chest", "calm", "anx", "tight", "body")),
    ("I keep worrying something bad will happen", ("worry", "bad", "happen", "anx")),
    ("panic attacks are back and Im scared", ("panic", "scared", "attack", "anx")),
]:
    add("anxiety", "en", msg, kw)
for msg, kw in [
    ("привет сегодня мне не хорошо", ("сегодн", "плох", "тяжел", "чувств", "тревог")),
    ("сегодня я проснулась тревожно", ("утр", "тревог", "просн", "чувств")),
    ("грудь сжимается от тревоги не могу успокоиться", ("груд", "тревог", "успок", "сжим")),
    ("боюсь что случится что-то плохое", ("бою", "плох", "тревог", "случ")),
]:
    extra = {"user_gender": "female"} if "проснулась" in msg else {}
    add("anxiety", "ru", msg, kw, **extra)

# stress
for msg, kw in [
    ("Lets talk about stress", ("stress", "stressed", "pressure", "feel")),
    ("everything feels overwhelming lately", ("overwhelm", "stress", "lately", "feel")),
    ("I cant switch off after work", ("work", "switch", "stress", "cant")),
    ("my shoulders are always tense", ("tense", "shoulder", "body", "stress")),
]:
    add("stress", "en", msg, kw)
for msg, kw in [
    ("давай поговорим о стрессе", ("стресс", "напряж", "устал", "давлен")),
    ("всё навалилось и я на грани", ("навал", "гран", "тяжел", "устал")),
    ("после работы не могу расслабиться", ("работ", "расслаб", "после", "напряж")),
    ("плечи постоянно напряжены", ("плеч", "напряж", "тел", "стресс")),
]:
    add("stress", "ru", msg, kw)

# grief
for msg, kw in [
    ("my mother died last year and holidays are hard", ("mother", "died", "grief", "loss", "hard")),
    ("I still cry when I think about my dad", ("dad", "cry", "think", "loss", "miss")),
    ("the anniversary of their death is coming up", ("anniversary", "death", "coming", "loss")),
    ("I feel guilty for laughing sometimes", ("guilty", "laugh", "grief", "feel")),
]:
    add("grief", "en", msg, kw)
for msg, kw in [
    ("мама умерла год назад праздники тяжёлые", ("мам", "умер", "праздн", "тяжел", "гор")),
    ("плачу когда вспоминаю отца", ("отц", "плач", "вспом", "гор")),
    ("скоро годовщина смерти близкого", ("годовщ", "смерт", "близк", "гор")),
    ("чувствую вину когда мне бывает весело", ("вин", "весел", "гор", "чувств")),
]:
    add("grief", "ru", msg, kw)

# clarity
for msg, kw in [
    ("help me sort out my thoughts", ("thought", "sort", "clarify", "mind")),
    ("my head is spinning with too many decisions", ("decision", "head", "spin", "thought")),
    ("I cant tell what I actually want", ("want", "tell", "confus", "feel")),
    ("everything feels tangled and I need clarity", ("tangle", "clarity", "feel", "sort")),
]:
    add("clarity", "en", msg, kw)
for msg, kw in [
    ("помоги разобраться в мыслях", ("мысл", "разобр", "помог", "голов")),
    ("в голове каша из решений", ("голов", "реш", "каш", "мысл")),
    ("не понимаю чего я на самом деле хочу", ("хоч", "поним", "сам", "дел")),
    ("всё перепуталось нужна ясность", ("ясн", "перепут", "нуж", "мысл")),
]:
    add("clarity", "ru", msg, kw)

# somatic
for msg, kw in [
    ("I feel emptiness in my chest after the breakup", ("chest", "empt", "body", "feel")),
    ("my stomach knots up when I think about work", ("stomach", "knot", "work", "body")),
    ("I cant sleep my body wont relax", ("sleep", "body", "relax", "cant")),
    ("there is a lump in my throat when I try to speak", ("throat", "lump", "speak", "body")),
]:
    add("somatic", "en", msg, kw)
for msg, kw in [
    ("чувствую пустоту в груди", ("груд", "пуст", "чувств", "тел")),
    ("живот сжимается когда думаю о работе", ("живот", "работ", "сжим", "дума")),
    ("тело не расслабляется ночью", ("тел", "ноч", "расслаб", "сон")),
    ("ком в горле когда пытаюсь говорить", ("горл", "ком", "говор", "тел")),
]:
    add("somatic", "ru", msg, kw)

path = ROOT / "eval" / "cross_topic_regression.jsonl"
with path.open("w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"Wrote {len(cases)} cases to {path}")
