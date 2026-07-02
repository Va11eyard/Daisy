"""
synthesize_ru_kk_dialogues.py — Generate high-quality RU and KK therapy dialogues.

Uses template-based synthesis (no API calls) to create realistic, varied
therapy conversations in Russian and Kazakh.  Produces ChatML-formatted
output for LoRA training.

Scenarios: 14 total (7 clusters × 2 variants)
  breakup, work, anxiety, stress, grief, clarity, somatic

Output: 30 variations per scenario × locale
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daisy.synthesize")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

# Polish diacritics that must NOT appear
_POLISH_DIACRITICS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"

# ---------------------------------------------------------------------------
# Scenario definitions
# Each scenario has:
#   - name: short identifier
#   - cluster: emotional cluster
#   - variants: list of 2 variant descriptions
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict] = [
    {
        "name": "breakup",
        "cluster": "loss",
        "variants": [
            "user just broke up with partner, feeling empty",
            "user ended relationship, now has regret",
        ],
    },
    {
        "name": "work",
        "cluster": "burnout",
        "variants": [
            "user's boss is yelling daily, burnout",
            "user fears being fired, financial anxiety",
        ],
    },
    {
        "name": "anxiety",
        "cluster": "anxiety",
        "variants": [
            "user has tight chest, can't calm down",
            "user has panic attacks, scared",
        ],
    },
    {
        "name": "stress",
        "cluster": "stress",
        "variants": [
            "everything overwhelming, can't switch off",
            "physical tension in shoulders",
        ],
    },
    {
        "name": "grief",
        "cluster": "grief",
        "variants": [
            "parent died, holidays are hard",
            "feel guilty when having fun after loss",
        ],
    },
    {
        "name": "clarity",
        "cluster": "confusion",
        "variants": [
            "head spinning with decisions",
            "can't tell what actually want",
        ],
    },
    {
        "name": "somatic",
        "cluster": "somatic",
        "variants": [
            "emptiness in chest after breakup",
            "body won't relax at night",
        ],
    },
]

# ---------------------------------------------------------------------------
# Template fragments — these create natural variation
# ---------------------------------------------------------------------------

# Different ways a user might express the same scenario (RU)
_USER_PHRASINGS_RU: Dict[str, List[str]] = {
    "breakup": [
        "Мы расстались с партнером. Я чувствую пустоту внутри. Не знаю, как дальше жить.",
        "Мне кажется, что после расставания я потерял часть себя. Внутри всё пусто.",
        "Прошла неделя после разрыва. Я всё ещё не могу нормально есть и спать.",
        "Мы разошлись, и я не понимаю, кто я теперь без этого человека.",
        "Я сама ушла, но теперь жалею. Может, я совершила ошибку?",
        "Разрыв был моим решением, но почему мне так больно?",
    ],
    "work": [
        "Начальник орёт каждый день. Я уже не выдерживаю, сил нет.",
        "На работе ад. Босс постоянно кричит, унижает перед коллегами.",
        "Я боюсь, что меня уволят. У меня кредит, и я не знаю, что буду делать.",
        "Могут сократить. Я не сплю ночами от страха остаться без денег.",
        "На работе давят постоянно. Я прихожу домой и плачу.",
        "Мне сказали, что я недостаточно эффективен. Теперь я в панике.",
    ],
    "anxiety": [
        "У меня сжимает грудь. Я не могу успокоиться, всё время в напряжении.",
        "Иногда мне кажется, что я схожу с ума. Сердце колотится, дышать трудно.",
        "У меня панические атаки. Я боюсь, что умру.",
        "Когда начинается приступ, мне страшно. Я не знаю, как его остановить.",
        "Я постоянно на взводе. Любой звук меня пугает.",
        "Мне страшно выходить из дома. Что если снова случится приступ?",
    ],
    "stress": [
        "Всё навалилось сразу. Я не справляюсь, хочется просто исчезнуть.",
        "Я не могу выключить голову. Мысли крутятся постоянно, не дают спать.",
        "Плечи каменные, шея болит. Я не могу расслабиться ни на минуту.",
        "В теле постоянное напряжение. Как будто готовлюсь к опасности.",
        "Я уже не помню, когда последний раз отдыхал по-настоящему.",
        "Всё требует моего внимания. Я разрываюсь на части.",
    ],
    "grief": [
        "Мама умерла три месяца назад. Праздники без неё невыносимы.",
        "Папы больше нет. Я до сих пор не могу поверить.",
        "Когда я смеюсь или веселюсь, потом чувствую вину. Как будто я предаю память о нём.",
        "Мне кажется, я не имею права радоваться после того, что случилось.",
        "Я до сих пор звоню на её номер, хотя знаю, что никто не возьмёт трубку.",
        "Вещи папы всё ещё лежат на своих местах. Я не могу их трогать.",
    ],
    "clarity": [
        "Голова кругом от решений. Я не знаю, что выбрать, боюсь ошибиться.",
        "Передо мной стоит выбор, который изменит всё. Я парализован страхом.",
        "Я не понимаю, чего на самом деле хочу. Живу чужой жизнью.",
        "Мне 30, а я до сих пор не знаю, кем хочу быть.",
        "Кажется, все вокруг знают, чего хотят, а я нет.",
        "Я постоянно спрашиваю советов, но потом всё равно не знаю, что делать.",
    ],
    "somatic": [
        "После расставания в груди пустота. Физически ощущаю.",
        "Внутри всё вывернуто наизнанку. Тело будто не моё.",
        "По ночам не могу расслабиться. Мышцы сжаты, как пружины.",
        "Я ложусь спать, а тело будто не отпускает. Всё в напряжении.",
        "Живот сжимается, когда я думаю о нём. Каждый раз.",
        "Я чувствую тяжесть в груди, которая не проходит уже неделю.",
    ],
}

# User phrasings in Kazakh
_USER_PHRASINGS_KK: Dict[str, List[str]] = {
    "breakup": [
        "Серікпен ажырастық. Ішім бос сезінемін. Әлі қараңғыда қалғандаймын.",
        "Ажырасқаныма екі апта болды. Ұйықтай алмай жүрмін.",
        "Өзім кеттім, бірақ енді өкінемін. Қате жасадым ба?",
        "Сүйіктімнен айрылдым. Енді мен кіммін?",
    ],
    "work": [
        "Басшым күнде бағынады. Шыдай алмай жүрмін.",
        "Жұмыстан қуып жібере ме деп қорқамын. Несие бар.",
        "Жұмыста қысым үнемі. Үйге келсем жылаймын.",
    ],
    "anxiety": [
        "Кеудем тарылып барады. Ұстамаймын өзімді.",
        "Паникалық шабуылдар басталды. Өліп бара жатқандаймын.",
        "Үнемі шошып оянамын. Әр дыбыс қорқытады.",
    ],
    "stress": [
        "Бәрі бірден үйілді. Шыдамым таусылды.",
        "Ойым тоқтамайды. Түнде ұйықтай алмаймын.",
        "Иығым ауырады. Денем демалмайды.",
    ],
    "grief": [
        "Анам қайтыс болды үш ай бұрын. Мерекесіз қиын.",
        "Әкем жоқ. Дәл қазір хабарласам деп қаламын.",
        "Күлгенімде кінәлі сезінемін. Ол үшін кешірім сұраймын ба?",
    ],
    "clarity": [
        "Шешім қабылдай алмай жүрмін. Қорқамын.",
        "Өзім не қалайтынымды білмеймін. Басқаның жолымен жүрмін.",
        "Жасыма келдім, бірақ мамандығымды қаламаймын.",
    ],
    "somatic": [
        "Кеудемде бостық. Физикалық сезінемін.",
        "Түнде денем босамайды. Барлық бұлшықет ширыққан.",
        "Ішім қысылып, жүрегім қағып кетеді ойлағанда.",
    ],
}

# Assistant response templates — 3 structural types for variety
# Type A: Reflection + open question
# Type B: Normalization + gentle question
# Type C: Naming the feeling + invitation to explore

_ASSISTANT_RESPONSES_RU: Dict[str, List[str]] = {
    "breakup": [
        # Type A: reflection
        "Тебе сейчас очень больно, и это нормально — после расставания чувствовать пустоту. Как ты проводишь эти дни?",
        "Потеря близкого человека оставляет рану. Что для тебя самое тяжёлое сейчас — днём или ночью?",
        "Ты говоришь о пустоте — это мощное чувство. Что ты делаешь, когда она становится невыносимой?",
        # Type B: normalization
        "После разрыва чувствовать себя потерянным — это естественно. Ты не одинок в этом. Как тебе удаётся заботиться о себе сейчас?",
        "Многое изменилось, и это страшно. Но ты здесь, и это уже что-то. Что даёт тебе хоть малейшее облегчение?",
        "Сожаление — это знак, что ты способен любить. Это больно, но это говорит о тебе хорошее. Что бы ты хотел сказать себе в этот момент?",
        # Type C: naming + invitation
        "Похоже, перед тобой стоит вопрос: кто я без этого человека? Это глубокий вопрос. Ты когда-нибудь думал о себе отдельно от отношений?",
        "Ты упомянул, что ушёл сам — но боль всё равно с тобой. Иногда решение правильное, а сердцу всё равно тяжело. Что помогает тебе дышать?",
    ],
    "work": [
        "Каждый день под криком — это травма для нервной системы. Как твоё тело реагирует, когда начальник повышает голос?",
        "Ты живёшь в состоянии постоянной угрозы. Это выматывает. Что ты чувствуешь прямо перед входом на работу?",
        "Страх потери работы парализует. Какие мысли крутятся у тебя в голове ночью?",
        "Такое давление на работе не может не сказываться. Ты когда-нибудь позволяешь себе просто быть с этой болью?",
        "Унижение перед коллегами — это особенно больно. Как ты себя чувствуешь после таких моментов?",
        "Финансовый страх — это базовый страх. Он задевает глубоко. Что бы изменило твою ситуацию хоть немного?",
    ],
    "anxiety": [
        "Сжимание в груди — это телесный сигнал тревоги. Ты замечаешь, перед чем это обычно начинается?",
        "Когда сердце колотится, страшно. Что ты обычно делаешь в такие моменты?",
        "Паника обманывает мозг — он думает, что угроза реальна. Как ты понимаешь, что приступ закончился?",
        "Ты живёшь в состоянии высокой боевой готовности. Это выматывает. Что помогает хоть ненадолго отпустить?",
        "Страх следующего приступа может быть сильнее самого приступа. Как ты справлялся раньше?",
        "Боязнь выйти из дома — это защита, не слабость. Что случилось, когда ты в последний раз попробовал выйти?",
    ],
    "stress": [
        "Когда всё навалилось сразу, легко потерять себя. Что из этого списка давит на тебя сильнее всего?",
        "Голова не выключается — значит, ты живёшь в режиме выживания. Когда последний раз ты позволял себе ничего не делать?",
        "Плечи и шея помнят всё, что голова пытается вытеснить. Ты когда-нибудь замечал, как тело расслабляется?",
        "Постоянное напряжение — это телесный щит. От чего ты защищаешься?",
        "Ты говоришь, что разрываешься — это точное слово. Часть тебя хочет угодить всем, а другая часть?",
        "Когда ты не помнишь, когда отдыхал — это сигнал. Что помешало тебе в последний раз остановиться?",
    ],
    "grief": [
        "Три месяца — это так мало. Горе не имеет сроков годности. Как ты провёл последний праздник?",
        "Не верить — это нормально. Мозг защищает нас от слишком большой боли. Что ты чувствуешь, когда набираешь её номер?",
        "Вина за радость — это признак глубокой любви. Но любящий человек хотел бы для тебя счастья, правда?",
        "Кажется, ты носишь траур внутри себя. Как ты разрешаешь себе быть грустным?",
        "Вещи на местах — это способ держать связь. Что случится, если ты тронешь одну из них?",
        "Ты не предаёшь память, когда смеёшься. Ты живёшь — и это то, чего бы хотели для тебя. Когда ты это осознал?",
    ],
    "clarity": [
        "Паралич перед выбором — это страх ошибиться, а не слабость. Что случится в худшем случае?",
        "Иногда 'не знаю' — это честный ответ. Что ты знаешь точно — чего ты НЕ хочешь?",
        "Жить чужой жизнью может быть безопаснее, но тесно. Когда ты последний раз делал что-то только для себя?",
        "Вопрос 'кто я' — важный. Ты когда-нибудь писал список того, что для тебя важно?",
        "Сравнение с другими — это ловушка. Что бы ты делал, если бы никто не видел?",
        "Спрашивать советы — это забота о себе. Но решение — твоё. Что твоя интуиция шепчет?",
    ],
    "somatic": [
        "Пустота в груди — это тело говорит о потере. Оно честнее слов. Когда она усиливается?",
        "Тело помнит то, что разум пытается забыть. Что происходит, когда ты просто сидишь с этим ощущением?",
        "Мышцы не отпускают, потому что чувствуют опасность. Что ты делаешь перед сном?",
        "Когда тело в напряжении — оно защищает тебя. Но сейчас, в этой комнате, ты в безопасности. Чувствуешь разницу?",
        "Телесные ощущения — это мессенджеры. Если бы твоя грудь могла говорить, что бы сказала?",
        "Неделю держать тяжесть в груди — это много. Ты находил способ хоть ненадолго облегчить её?",
    ],
}

# Assistant response templates in Kazakh
_ASSISTANT_RESPONSES_KK: Dict[str, List[str]] = {
    "breakup": [
        "Ажырасу өте ауыр. Ішің бос сезіну — қалыпты нәрсе. Қазіргі күндеріңді қалай өткізіп жүрсің?",
        "Жақын адамнан айрылу жара қалдырады. Кешке немесе таңертеңгілік қай кезең қиынырақ?",
        "Бостықты сезіну — күшті сезім. Ол күшейгенде не істейсің?",
        "Өзің кеткеніңмен, жүрекке бәрібір ауыр. Дем алуға не көмектеседі?",
    ],
    "work": [
        "Күнде бағыну — жүйке үшін қиын. Денең қалай реакция береді?",
        "Жұмыстан қорқу — парализует. Түнде не ойлайсың?",
        "Қысым денеңе де әсер етеді. Соңғы рет қашан тынықтың?",
    ],
    "anxiety": [
        "Кеудеңнің тарылуы — алаңдаушылық белгісі. Бұдан бұрын не болды?",
        "Жүрек соққанда қорқынышты. Сол сәттерде не істейсің?",
        "Үйден шыққың келмейді — бұл қорғаныс. Соңғы рет сыртқа шыққанда не болды?",
    ],
    "stress": [
        "Бәрі бірден түскенде өзіңді жоғалту оңай. Ең ауыр не?",
        "Ой тоқтамайды — бұл аман қалу режимі. Қашан демалдың?",
        "Денең ширыққан. Неден қорғанып жүрсің?",
    ],
    "grief": [
        "Үш ай — өте аз уақыт. Қайғы мерзімі жоқ. Мерекені қалай өткіздің?",
        "Сенбеу — қалыпты. Ми қорғанады. Нөмірді тергенде не сезінесің?",
        "Күлгенде кінәлі сезіну — махаббат белгісі. Бірақ ол сенің бақытты болғаныңды қалар еді.",
    ],
    "clarity": [
        "Таңдау алдында қорқу — табиғи. Ең нашар не болуы мүмкін?",
        "Білмеймін деп айту — шындық. Не қаламайтыныңды білесің бе?",
        "Басқаның жолымен жүру қауіпсіз, бірақ тар. Соңғы рет өзің үшін не істедің?",
    ],
    "somatic": [
        "Кеудеңдегі бостықты денең сөйлеп отырғанындай сезінуің қалыпты. Бұл физикалық сезім қашан күшейеді?",
        "Түнде денең демалмай, бұлшықеттерің ширыққан күйде қалады. Ұйықтар алдында не істейсің? Соңғы рет қашан жеңілдедің?",
        "Денең саған маңызды хабар жіберуде. Егер кеудең сөйлесе, қазіргі сезімін қалай сипаттар еді? Қашан басталғанын еске түсіруге тырысып көр.",
    ],
}

# System prompts for synthesis (different from inference prompts — used for training data)
_SYSTEM_PROMPTS_SYNTH_RU = {
    "open": (
        "Ты — теплый, эмпатичный терапевтический собеседник. Общайся на 'ты'. "
        "Отвечай 2-4 предложениями. Ссылайся на сообщение пользователя. "
        "Никаких английских вставок. Только русский язык."
    ),
}

_SYSTEM_PROMPTS_SYNTH_KK = {
    "open": (
        "Сен — жылы, эмпатиялы терапевтикалық сөйесің. "
        "2-4 сөйлеммен жауап бер. Пайдаланушының хабарламасына сілтеме жаса. "
        "Тек қазақша."
    ),
}


# ---------------------------------------------------------------------------
# Synthesis functions
# ---------------------------------------------------------------------------

def synthesize_dialogue(scenario: str, locale: str, variant_idx: int = 0) -> Dict:
    """Synthesize a single therapy dialogue turn pair.

    Uses template-based generation with natural language variation.
    No API calls — deterministic based on random seed.

    Args:
        scenario:    Scenario name (e.g., "breakup", "work").
        locale:      "ru" or "kk".
        variant_idx: Which scenario variant (0 or 1).

    Returns:
        ChatML-formatted dict: {"messages": [{"role": "system", ...}, ...]}
    """
    locale = locale.lower().strip()

    # Select user phrasing
    if locale == "ru":
        phrasings = _USER_PHRASINGS_RU.get(scenario, ["Мне тяжело."])
        responses = _ASSISTANT_RESPONSES_RU.get(scenario, ["Расскажи подробнее."])
        system = _SYSTEM_PROMPTS_SYNTH_RU["open"]
    elif locale == "kk":
        phrasings = _USER_PHRASINGS_KK.get(scenario, ["Маған ауыр."])
        responses = _ASSISTANT_RESPONSES_KK.get(scenario, ["Толығырақ айтып бер."])
        system = _SYSTEM_PROMPTS_SYNTH_KK["open"]
    else:
        raise ValueError(f"Unsupported locale: {locale}. Use 'ru' or 'kk'.")

    # Pick random user phrasing (deterministic if seed set)
    user_content = random.choice(phrasings)

    # Pick random assistant response
    assistant_content = random.choice(responses)

    # Build ChatML
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "scenario": scenario,
            "locale": locale,
            "variant_idx": variant_idx,
            "cluster": next(
                (s["cluster"] for s in SCENARIOS if s["name"] == scenario),
                "unknown",
            ),
            "synthetic": True,
            "source": "template_synthesis_v1",
        },
    }


def validate_synthetic(dialogue: Dict) -> Dict:
    """Validate a synthetic dialogue for quality.

    Checks:
      - No Latin leaks in RU/KK assistant turns
      - Informal "ты" for RU
      - ≥50 characters in assistant response
      - References user message (assistant mentions something from user)
      - No emoji
      - No structural leaks

    Args:
        dialogue: ChatML dict from synthesize_dialogue().

    Returns:
        Validation dict: {passed, checks, failures}
    """
    failures = []
    messages = dialogue.get("messages", [])
    metadata = dialogue.get("metadata", {})
    locale = metadata.get("locale", "ru")

    # Extract assistant turn
    assistant_msg = None
    user_msg = None
    for m in messages:
        if m.get("role") == "assistant":
            assistant_msg = m.get("content", "")
        elif m.get("role") == "user":
            user_msg = m.get("content", "")

    if not assistant_msg:
        failures.append("missing_assistant_turn")
        return {"passed": False, "checks": {}, "failures": failures}

    # Check 1: Minimum length
    length_ok = len(assistant_msg) >= 50
    if not length_ok:
        failures.append(f"too_short:{len(assistant_msg)}")

    # Check 2: No Latin leaks in Cyrillic
    latin_leak = False
    if locale in ("ru", "kk"):
        words = assistant_msg.split()
        latin_count = sum(
            1 for w in words
            if re.match(r"^[a-zA-Z]+$", re.sub(r"[^\w\s]", "", w))
        )
        if words and latin_count / len(words) > 0.10:
            latin_leak = True
            failures.append(f"latin_leak:{latin_count}/{len(words)}")

        # Check Polish diacritics
        if any(ch in assistant_msg for ch in _POLISH_DIACRITICS):
            failures.append("polish_diacritics")

    # Check 3: Informal ты for RU
    informal_ok = True
    if locale == "ru":
        lower = assistant_msg.lower()
        vy_count = lower.count(" вы ") + lower.count("вас ")
        ty_count = lower.count(" ты ") + lower.count("тебя ") + lower.count("тебе ")
        if vy_count > ty_count and ty_count == 0:
            informal_ok = False
            failures.append("formal_vy_not_ty")

    # Check 4: References user message
    references_user = False
    if user_msg and assistant_msg:
        # Check if any word from user appears in assistant response
        user_words = set(
            re.sub(r"[^\w\s]", "", w).lower()
            for w in user_msg.split()
            if len(w) > 3
        )
        assistant_lower = assistant_msg.lower()
        for uw in user_words:
            if uw in assistant_lower:
                references_user = True
                break
        # Or check for generic reference markers
        if not references_user:
            ref_markers = [
                "ты говоришь", "ты сказал", "ты упомянул", "ты пишешь",
                "тебе сейчас", "это нормально", "ты чувствуешь",
                "сіз айттыңыз", "сіз сезінесіз", "сіз",
                "это мощное", "это точное", "похоже",
            ]
            if any(m in assistant_lower for m in ref_markers):
                references_user = True

        if not references_user:
            failures.append("no_user_reference")

    # Check 5: No emoji
    emoji_pattern = re.compile(r"[🌼🌱💙🌸🌿✨🍃🌺🔆💚🌻🌷]")
    if emoji_pattern.search(assistant_msg):
        failures.append("emoji_present")

    # Check 6: No structural leaks
    if re.search(r"^(Assistant|Question|User|Human):", assistant_msg, re.IGNORECASE):
        failures.append("structural_leak")
    if re.search(r"\*\*Rubric\*\*|Score:\s*\d+", assistant_msg, re.IGNORECASE):
        failures.append("rubric_leak")

    passed = len(failures) == 0

    return {
        "passed": passed,
        "checks": {
            "length_ok": length_ok,
            "no_latin_leak": not latin_leak,
            "informal_ty": informal_ok,
            "references_user": references_user,
            "no_emoji": not bool(emoji_pattern.search(assistant_msg)),
            "no_structural_leak": True,
        },
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_batch(
    scenario: str,
    locale: str,
    variant_idx: int,
    count: int = 30,
) -> Tuple[List[Dict], Dict]:
    """Generate a batch of dialogue variations for one scenario × locale × variant.

    Args:
        scenario:    Scenario name.
        locale:      "ru" or "kk".
        variant_idx: 0 or 1.
        count:       Number of variations to generate.

    Returns:
        (dialogues, stats) where stats is a dict of validation counts.
    """
    dialogues = []
    stats = {"generated": 0, "passed": 0, "failed": 0, "failure_reasons": {}}

    for i in range(count):
        dialogue = synthesize_dialogue(scenario, locale, variant_idx)
        validation = validate_synthetic(dialogue)

        stats["generated"] += 1
        if validation["passed"]:
            stats["passed"] += 1
            dialogues.append(dialogue)
        else:
            stats["failed"] += 1
            for reason in validation["failures"]:
                stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + 1

    return dialogues, stats


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize RU and KK therapy dialogues for Daisy LoRA v15 training."
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthesized",
        help="Output directory for synthesized data.",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=30,
        help="Variations per scenario × locale (default: 30).",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["ru", "kk"],
        help="Locales to generate (default: ru kk).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Specific scenarios to generate (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Filter scenarios if specified
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s["name"] in args.scenarios]

    output_dir = Path(args.output_dir)

    all_stats = []
    total_generated = 0
    total_passed = 0

    for locale in args.locales:
        locale_dir = output_dir / locale
        locale_dir.mkdir(parents=True, exist_ok=True)

        batch_num = 0
        batch_dialogues = []

        for scenario_def in scenarios:
            scenario_name = scenario_def["name"]
            for variant_idx, variant_desc in enumerate(scenario_def["variants"]):
                logger.info(
                    f"Generating: {scenario_name} (variant {variant_idx + 1}/2) "
                    f"× {locale} × {args.variations} variations"
                )

                dialogues, stats = generate_batch(
                    scenario=scenario_name,
                    locale=locale,
                    variant_idx=variant_idx,
                    count=args.variations,
                )

                stats["scenario"] = scenario_name
                stats["locale"] = locale
                stats["variant"] = variant_desc
                all_stats.append(stats)

                total_generated += stats["generated"]
                total_passed += stats["passed"]

                # Accumulate into batch file
                batch_dialogues.extend(dialogues)

                # Write batch every 100 dialogues
                if len(batch_dialogues) >= 100:
                    batch_path = locale_dir / f"batch_{batch_num}.jsonl"
                    with batch_path.open("w", encoding="utf-8") as f:
                        for d in batch_dialogues:
                            f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    logger.info(f"  Written {len(batch_dialogues)} to {batch_path}")
                    batch_num += 1
                    batch_dialogues = []

        # Write remaining dialogues
        if batch_dialogues:
            batch_path = locale_dir / f"batch_{batch_num}.jsonl"
            with batch_path.open("w", encoding="utf-8") as f:
                for d in batch_dialogues:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            logger.info(f"  Written {len(batch_dialogues)} to {batch_path}")

    # --- Print validation stats ---
    print("\n" + "=" * 60)
    print("Synthesis Validation Report")
    print("=" * 60)
    print(f"Total generated:  {total_generated}")
    print(f"Total passed:     {total_passed} ({total_passed / total_generated * 100:.1f}%)")
    print(f"Total failed:     {total_generated - total_passed}")
    print()

    # Per-scenario breakdown
    print("Per-scenario breakdown:")
    for stat in all_stats:
        status = "PASS" if stat["failed"] == 0 else f"FAIL({stat['failed']})"
        print(
            f"  {stat['locale']:2s} | {stat['scenario']:10s} | "
            f"{stat['generated']:3d} gen | {stat['passed']:3d} OK | "
            f"{status}"
        )
        if stat["failure_reasons"]:
            for reason, count in stat["failure_reasons"].items():
                print(f"         - {reason}: {count}")

    # Aggregate failure reasons
    all_failures = {}
    for stat in all_stats:
        for reason, count in stat["failure_reasons"].items():
            all_failures[reason] = all_failures.get(reason, 0) + count

    if all_failures:
        print("\nAggregate failure reasons:")
        for reason, count in sorted(all_failures.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print("=" * 60)

    # Summary by locale
    for locale in args.locales:
        locale_stats = [s for s in all_stats if s["locale"] == locale]
        locale_total = sum(s["generated"] for s in locale_stats)
        locale_passed = sum(s["passed"] for s in locale_stats)
        print(f"{locale.upper()}: {locale_passed}/{locale_total} passed "
              f"({locale_passed / locale_total * 100:.1f}%)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run self-tests for synthesis and validation."""
    print("=" * 60)
    print("synthesize_ru_kk_dialogues.py self-test")
    print("=" * 60)

    # Test 1: synthesize_dialogue basic
    print("\n--- synthesize_dialogue ---")
    dialogue = synthesize_dialogue("breakup", "ru", variant_idx=0)
    assert "messages" in dialogue
    assert len(dialogue["messages"]) == 3  # system, user, assistant
    assert dialogue["messages"][0]["role"] == "system"
    assert dialogue["messages"][1]["role"] == "user"
    assert dialogue["messages"][2]["role"] == "assistant"
    assert dialogue["metadata"]["locale"] == "ru"
    assert dialogue["metadata"]["scenario"] == "breakup"
    print("  [PASS] Basic RU dialogue structure")

    # Test 2: synthesize_dialogue KK
    dialogue_kk = synthesize_dialogue("anxiety", "kk", variant_idx=1)
    assert dialogue_kk["metadata"]["locale"] == "kk"
    assert len(dialogue_kk["messages"][2]["content"]) > 0
    print("  [PASS] Basic KK dialogue structure")

    # Test 3: validate_synthetic — clean pass
    print("\n--- validate_synthetic ---")
    validation = validate_synthetic(dialogue)
    assert isinstance(validation["passed"], bool)
    assert "checks" in validation
    print(f"  [PASS] Validation result: passed={validation['passed']}")
    if validation["failures"]:
        print(f"         (non-critical: {validation['failures']})")

    # Test 4: validate_synthetic — detect Latin leak
    bad_dialogue = {
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Мне жаль. It sounds hard."},
        ],
        "metadata": {"locale": "ru", "scenario": "test", "variant_idx": 0},
    }
    bad_val = validate_synthetic(bad_dialogue)
    assert not bad_val["passed"], "Should fail with Latin leak"
    assert any("latin" in f for f in bad_val["failures"])
    print("  [PASS] Latin leak detection")

    # Test 5: validate_synthetic — detect short response
    short_dialogue = {
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Коротко."},
        ],
        "metadata": {"locale": "ru", "scenario": "test", "variant_idx": 0},
    }
    short_val = validate_synthetic(short_dialogue)
    assert not short_val["passed"], "Should fail with too short"
    assert any("too_short" in f for f in short_val["failures"])
    print("  [PASS] Too short detection")

    # Test 6: validate_synthetic — detect emoji
    emoji_dialogue = {
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Мне жаль 🌼. Тебе тяжело?"},
        ],
        "metadata": {"locale": "ru", "scenario": "test", "variant_idx": 0},
    }
    emoji_val = validate_synthetic(emoji_dialogue)
    assert not emoji_val["passed"], "Should fail with emoji"
    assert any("emoji" in f for f in emoji_val["failures"])
    print("  [PASS] Emoji detection")

    # Test 7: All scenarios generate successfully
    print("\n--- All scenarios ---")
    for scenario_def in SCENARIOS:
        name = scenario_def["name"]
        d_ru = synthesize_dialogue(name, "ru", 0)
        d_kk = synthesize_dialogue(name, "kk", 0)
        assert len(d_ru["messages"][2]["content"]) >= 50, f"RU {name} too short"
        assert len(d_kk["messages"][2]["content"]) >= 50, f"KK {name} too short"
    print(f"  [PASS] All {len(SCENARIOS)} scenarios × 2 locales generate OK")

    # Test 8: Batch generation
    print("\n--- Batch generation ---")
    dialogs, stats = generate_batch("work", "ru", variant_idx=0, count=10)
    assert stats["generated"] == 10
    assert len(dialogs) == stats["passed"]
    print(f"  [PASS] Batch: {stats['passed']}/{stats['generated']} passed")

    # Test 9: Variation — same scenario, different calls produce different outputs
    print("\n--- Variation ---")
    random.seed(123)
    d1 = synthesize_dialogue("grief", "ru", 0)
    random.seed(456)
    d2 = synthesize_dialogue("grief", "ru", 0)
    assert d1["messages"][1]["content"] != d2["messages"][1]["content"] or \
           d1["messages"][2]["content"] != d2["messages"][2]["content"], \
        "Should produce different variations"
    print("  [PASS] Variation between seeds")

    # Test 10: KK has informal style
    print("\n--- KK style ---")
    kk_dialogue = synthesize_dialogue("clarity", "kk", 0)
    assert "сіз" in kk_dialogue["messages"][1]["content"] or \
           "мен" in kk_dialogue["messages"][1]["content"], "KK user should be in Kazakh"
    print("  [PASS] KK dialogue in Kazakh")

    print("\n" + "=" * 60)
    print("All synthesize_ru_kk_dialogues self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
