# Conversation Memory Fix — Multi-Turn Context Tracking

## The Problem (from your real chat)

```
Turn 1: Мне тревожно сегодня
        → OK response about anxiety

Turn 2: Не страх а тревога
        → Corrects fear→anxiety, but DRIFT: "уточнил" (formal), generic question

Turn 3: Я сломал свою модельку и мой шеф на меня ругается
        → FAIL: "это непростая ситуация" — GENERIC, doesn't mention:
          - шеф / boss
          - моделька / model
          - сломал / broke
          - ругается / yelling
          - anxiety connection (why I'm anxious)
        → LEAK: Assistant: ,.
```

**Diagnosis:** The model is NOT reading conversation history. Each turn is treated as a first message. The system prompt doesn't instruct the model to:
1. Reference previous turns
2. Connect the current topic to what was said before
3. Use consistent register (`ты` throughout)
4. Echo the user's specific words (topic anchoring)

---

## Fix 1: System Prompt with History Awareness

### The problem in current system prompt

If the prompt only sends the current user message without context:
```
System: You are a therapy AI...
User: Я сломал свою модельку...
Assistant: ??? (no idea about previous anxiety)
```

### The fix: Include conversation history in the prompt

```python
def build_messages_with_history(current_message: str, history: list, locale: str) -> list:
    """Build full message list with system prompt + history + current message.
    
    history: list of {role: "user"|"assistant", content: str}
    """
    
    # Build context summary from history
    history_summary = _summarize_history(history)
    
    # Build system prompt with history context
    if locale == "ru":
        system_prompt = (
            "Ты — Дэйзи, терапевтический собеседник. Говори неформально, на 'ты'. "
            "ВАЖНО: читай историю разговора ниже. Каждый ответ должен: "
            "1) Ссылаться на то, что человек уже сказал ранее "
            "2) Связывать текущую тему с предыдущими сообщениями "
            "3) Использовать конкретные слова человека (шеф, моделька, тревога) "
            "4) НЕ начинать с общих фраз типа 'это непростая ситуация' — "
            "   начинай с конкретики: 'Когда шеф ругается из-за сломанной модельки...' "
            "\n\n"
            f"Контекст разговора:\n{history_summary}\n"
        )
    elif locale == "en":
        system_prompt = (
            "You are Daisy, a therapy chatbot. "
            "CRITICAL: Read the conversation history below. Each response must: "
            "1) Reference what the person said earlier "
            "2) Connect the current topic to previous messages "
            "3) Use their specific words (boss, model, anxiety) "
            "4) NEVER start with generic phrases like 'that sounds difficult' — "
            "   start with specifics: 'When your boss yells about the broken model...' "
            "\n\n"
            f"Conversation context:\n{history_summary}\n"
        )
    else:
        system_prompt = _build_kk_prompt(history_summary)
    
    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add full history (last 6 turns max to fit in context window)
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    return messages


def _summarize_history(history: list) -> str:
    """Create a brief summary of the conversation so far."""
    if not history:
        return "(начало разговора)"
    
    # Extract key topics from each user turn
    topics = []
    for turn in history:
        if turn["role"] == "user":
            # Extract key nouns/verbs (simplified — can use NLP)
            content = turn["content"].lower()
            # Add key phrases as context items
            topics.append(f"- Человек сказал: '{turn['content'][:80]}'")
    
    return "\n".join(topics[-4:])  # Last 4 user messages
```

### What the model now sees

```
System: Ты — Дэйзи... 
        Контекст разговора:
        - Человек сказал: 'Мне тревожно сегодня'
        - Человек сказал: 'Не страх а тревога'
        - Человек сказал: 'Я сломал свою модельку и мой шеф на меня ругается'
        
User: Я сломал свою модельку и мой шеф на меня ругается и я не знаю что делать

Assistant: [Now the model KNOWS about the anxiety + work connection]
```

---

## Fix 2: Topic-Anchored Response Format

### The instruction that must be in EVERY system prompt

```python
_TOPIC_ANCHOR_RULE_RU = (
    "ПРАВИЛО ОТВЕТА:\n"
    "1. Начни с НАЗВАНИЯ конкретной ситуации человека, используя ЕГО слова\n"
    "2. Покажи связь с предыдущими сообщениями (если есть история)\n"
    "3. Задай вопрос\n"
    "\n"
    "ПРИМЕРЫ:\n"
    "Плохо: 'Это непростая ситуация...' — СЛИШКОМ ОБЩО\n"
    "Плохо: 'Я понимаю, тебе тяжело...' — НЕТ КОНКРЕТИКИ\n"
    "\n"
    "Хорошо: 'Когда шеф ругается из-за сломанной модельки — это 
"
    "         добавляет тревоги к той тревоге, что уже была. 
"
    "         Что именно он сказал, когда узнал про поломку?'\n"
    "\n"
    "Хорошо: 'Тревога из-за работы и сломанной модели — она в теле 
"
    "         где сидит? В груди, в животе?'\n"
)

_TOPIC_ANCHOR_RULE_EN = (
    "RESPONSE RULE:\n"
    "1. Start by NAMING the person's specific situation using THEIR words\n"
    "2. Show connection to previous messages (if there's history)\n"
    "3. Ask a follow-up question\n"
    "\n"
    "BAD: 'That sounds difficult...' — TOO VAGUE\n"
    "BAD: 'I understand you're going through a lot...' — NO SPECIFICS\n"
    "\n"
    "GOOD: 'When your boss yells about the broken model — that piles onto \n"
    "       the anxiety you already felt this morning. What did he say \n"
    "       when he found out?'\n"
    "\n"
    "GOOD: 'The work stress from the broken model — where do you feel \n"
    "       that in your body? Chest, stomach?'\n"
)
```

---

## Fix 3: Register Consistency (`ты` enforcement)

### The problem

Your conversation shows register drift:
- Turn 1: "тебе" ✓ (informal)
- Turn 2: "уточнил" ✗ (formal-ish, should be "уточнил**а**" if female or just avoid)
- Turn 3: "тебя" ✓ (informal)

### The fix: Explicit `ты` instruction

```python
_REGISTER_RULE_RU = (
    "ОБРАЩЕНИЕ: ВСЕГДА 'ты', НИКОГДА 'вы'.\n"
    "Плохо: 'Известно, что прикосновения к эмоциям...' — как учебник\n"
    "Плохо: 'вызывает этот страх' — отстранённо\n"
    "\n"
    "Хорошо: 'Когда шеф орёт...' — живо, конкретно\n"
    "Хорошо: 'Тебе сейчас страшно...' — близко, на 'ты'\n"
)
```

---

## Fix 4: Updated clean_model_text (catching the Assistant leak)

```python
def clean_model_text(text: str) -> str:
    """Clean model output — strip leaks, trailing debris, normalize."""
    if not text:
        return text
    
    original = text
    
    # 1. Strip from START: role headers
    text = re.sub(r"^(Assistant|Question|User|Human)[:\s]*", "", text, 
                  flags=re.IGNORECASE | re.MULTILINE)
    
    # 2. Strip from END: all known leak patterns
    # Order matters — most specific first
    end_patterns = [
        r"Assistant\s*[:;,.\s]*.*",      # "Assistant: ,.", "Assistant;", "Assistant ,"
        r"Question\s*[:;,.\s]*.*",
        r"User\s*[:;,.\s]*.*",
        r"Human\s*[:;,.\s]*.*",
        r"❮REFINE❯.*",
        r"ASURE[A-Z_]*.*",                # ASUREONEPRECEDES...
        r"OffsetTable.*",
        r"Daisy\s*x\d*.*",               # Daisy x02, Daisy x2023-04-18
        r"👋[a-zA-Z0-9]*.*",              # 👋ly, 👋lyric123
        r"\s*[,.:;\s]+$",                # trailing punctuation
    ]
    
    # Apply each pattern — strip everything from match to end
    for pattern in end_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 3. Remove meta-instruction phrases (training data contamination)
    meta_phrases = [
        r"Подстраивайся под текущую потребность человека[^.]*\.",
        r"Веди себя как реальный человек[^.]*\.",
        r"Ответь на русском языке[^.]*\.",
        r"response as a person[^.]*\.",
        r"\[" + r"[" + r"'" + r"" + r"].*?\]",  # ['что-то плохое', 'всё'] — JSON artifacts
    ]
    for phrase in meta_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)
    
    # 4. Collapse duplicate words
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    
    # 5. Clean punctuation loops
    text = re.sub(r"[.,\s]{4,}", "... ", text)
    text = re.sub(r"\.{4,}", "...", text)
    
    # 6. Strip trailing emoji + punctuation
    text = re.sub(r"[🌼🌱💙🌸🌿✨👋🏼🤔️😊🌟\s,.:;]+$", "", text)
    
    # 7. Final trim
    text = text.strip(" \n\t,.:;")
    
    # 8. Ensure the response ends with complete sentence
    # If it ends mid-word, find last complete sentence
    if text and not text[-1] in ".!?":
        # Find last sentence boundary
        last_boundary = max(
            text.rfind(". "),
            text.rfind("? "),
            text.rfind("! "),
        )
        if last_boundary > len(text) * 0.5:  # Only cut if we'd keep >50%
            text = text[:last_boundary + 1]
    
    return text.strip()
```

---

## What Your Conversation Should Look Like (Fixed)

```
You: Мне тревожно сегодня
Daisy: Тревога — она в теле где сейчас? В груди, в животе? 
        Расскажи, с чего началось сегодня.

You: Не страх а тревога
Daisy: Понял, тревога — это другое. Она шуршит фоном 
        и не отпускает. Что запустило её сегодня утром?

You: Я сломал свою модельку и теперь мой шеф на меня ругается
Daisy: Вот оно что — сломанная моделька + шеф, который орёт. 
        Неудивительно, что тревога зашкаливает. 
        Когда шеф начал ругаться — что конкретно он сказал? 
        И ты что почувствовал в этот момент — стыд, страх, злость?
```

**Key differences:**
- Every response uses **your specific words** (моделька, шеф, ругается, тревога)
- The model **connects across turns** (anxiety from morning → work stress now)
- **Consistent `ты`** throughout
- **No generic openings** — every response starts with the specific situation
- **No leaks** — clean stop at end of thought

---

## Quick Implementation Checklist

| Fix | File to Change | Time |
|-----|---------------|------|
| 1. History in system prompt | `score_qwen3.py` or `score_qwen3_aml.py` — `build_messages_with_history()` | 30 min |
| 2. Topic anchor instruction | `system_prompt_qwen3.py` — add `_TOPIC_ANCHOR_RULE_RU` / `_EN` | 15 min |
| 3. Register consistency | `system_prompt_qwen3.py` — add `_REGISTER_RULE_RU` | 10 min |
| 4. End-of-response cleanup | `score_qwen3.py` — update `clean_model_text()` | 20 min |
| 5. max_tokens=256 | `deployment-qwen3-lora-v15.yaml` or env var | 5 min |
| 6. Expanded stop_strings | `score_qwen3.py` — generation call | 10 min |

**Total: ~90 minutes to deploy and test.**

Then run regression — you should see:
- 0 structural/script leaks (generation fix)
- Fewer keyword_mismatch (topic anchoring)
- Better RU scores (register consistency + history awareness)
- Target: 65-70% overall (up from 57%)
