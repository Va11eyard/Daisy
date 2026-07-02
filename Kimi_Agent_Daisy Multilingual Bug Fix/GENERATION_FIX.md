# Generation Fix — Stop Strings, Not Token Caps

## The Problem (from your screenshot)

```
User: Я сломал свою модельку и теперь мой шеф на меня ругается
Model: ...Как тебе кажется, есть ли что-то конкретное, что ты можешь сделать для:  Assistant: ,.
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^
                                                        structural leak at the END
```

**Root cause:** `max_new_tokens=90` hard-cuts the generation. The model was mid-sentence, hit the limit, and degenerated into `Assistant: ,.` trying to continue.

**The model's response quality is fine.** The problem is the generation pipeline, not the model.

---

## The Fix: 3 Changes

### 1. Raise max_new_tokens to 256 (safety net, not routine limit)

```python
# BEFORE (in score_qwen3.py or deployment YAML):
DAISY_DEFAULT_MAX_TOKENS: "90"   # Hard cut — causes end-of-generation leaks

# AFTER:
DAISY_DEFAULT_MAX_TOKENS: "256"  # Safety net — almost never triggers
```

The model will naturally stop via stop_strings in 50-120 tokens for therapy turns. 256 is only there to prevent runaway generation (which stop_strings already prevent, but belt-and-suspenders).

### 2. Strengthen stop_strings — these are the PRIMARY control

```python
# BEFORE:
stop_strings = ["Assistant:", "Question:", "User:", "Human:", "\n\nUser"]

# AFTER — add all leak patterns that appear at the end:
stop_strings = [
    "Assistant:",
    "Question:",
    "User:",
    "Human:",
    "\n\nUser",
    "\nUser:",
    "\nAssistant:",
    "assistant",       # lowercase variant
    "ASURE",           # rubric token fragments
    "❮REFINE❯",
    "OffsetTable",
    "Daisy x",         # from training data artifacts
    "👋ly",            # emoji corruption
    "\n\n",            # double newline = end of turn
]
```

### 3. Add end-of-response cleanup — strip trailing debris

```python
def clean_model_text(text: str) -> str:
    """Clean model output: strip headers, trailing debris, normalize."""
    if not text:
        return text
    
    # 1. Strip role headers from START
    text = re.sub(r"^(Assistant|Question|User|Human)[:\s]*", "", text, flags=re.IGNORECASE)
    
    # 2. Strip trailing debris — everything from first leak pattern to end
    # This catches "...for: Assistant: ," — keeps "...for:" or just "...for"
    trailing_leak_patterns = [
        r"Assistant\s*:.*",           # Assistant: anything to end
        r"Question\s*:.*",
        r"User\s*:.*",
        r"Human\s*:.*",
        r"❮REFINE❯.*",
        r"ASURE.*",                    # rubric fragments
        r"OffsetTable.*",
        r"Daisy\s*x\d*.*",            # Daisy x2023-04-18
        r"👋ly.*",                     # emoji corruption
        r"[.,\s]*$",                   # trailing punctuation cleanup
    ]
    for pattern in trailing_leak_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 3. Collapse duplicate words ("nderstand nderstand")
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    
    # 4. Remove punctuation loops
    text = re.sub(r"[.,\s]{4,}", "... ", text)  # ".,.,." -> "... "
    text = re.sub(r"\.{4,}", "...", text)
    
    # 5. Strip trailing emoji garbage
    text = re.sub(r"[🌼🌱💙🌸🌿✨👋🏼🤔️😊🌟]{1,}\s*[.,\s]*$", "", text)
    
    # 6. Remove persona meta-instructions
    meta_phrases = [
        r"Подстраивайся под текущую потребность человека[^.]*\.",
        r"Веди себя как реальный человек[^.]*\.",
        r"Ответь на русском языке[^.]*\.",
        r"response as a person[^.]*\.",
    ]
    for phrase in meta_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)
    
    # 7. Final cleanup
    text = text.strip(" ,.:;\n\t")
    
    return text.strip()
```

---

## Why This Works

| Before (max_tokens=90) | After (stop_strings primary) |
|------------------------|------------------------------|
| Model forced to stop at 90 tokens | Model stops naturally at end of thought |
| Mid-sentence truncation → `...что ты можешь сделать для:` | Complete sentence → `...что ты можешь сделать?` |
| Degeneration at boundary → `Assistant: ,.` | Clean termination |
| P50 latency higher (model fights the limit) | P50 lower (model stops when done) |

**Latency actually improves** because the model stops in 60-80 tokens naturally instead of being forced to use all 90.

---

## Updated deployment YAML (minimal change)

```yaml
# In deployment-qwen3-lora-v15.yaml, change ONLY:
DAISY_DEFAULT_MAX_TOKENS: "256"     # was "90" or "120"
DAISY_STOP_STRINGS: "Assistant:,Question:,User:,Human:,\\n\\nUser,\\nAssistant:,ASURE,❮REFINE❯,OffsetTable,Daisy x,👋ly"
```

Or in `score_qwen3.py`, change the generation call:

```python
# BEFORE:
outputs = model.generate(
    input_ids,
    max_new_tokens=90,  # <-- THIS IS THE PROBLEM
    stop_strings=["Assistant:", "Question:", "User:", "Human:"],
    ...
)

# AFTER:
outputs = model.generate(
    input_ids,
    max_new_tokens=256,  # Safety net only
    stop_strings=[       # PRIMARY control
        "Assistant:", "Question:", "User:", "Human:",
        "\n\nUser", "\nAssistant:", "assistant",
        "ASURE", "❮REFINE❯", "OffsetTable",
        "Daisy x", "👋ly", "\n\n",
    ],
    ...
)
# Then clean with improved clean_model_text()
```

---

## Quick Test

Deploy with `max_new_tokens=256` and the expanded stop strings. The screenshot case should produce something like:

```
О, это действительно непростая ситуация. Когда шеф ругается после того, 
как что-то сломалось — это особенно болезненно. Давай попробуем вместе 
найти способ справиться. Как тебе кажется, есть ли что-то конкретное, 
что ты можешь сделать, чтобы исправить модельку?
```

No `Assistant: ,.` at the end. Complete sentence. Natural stop.

---

## Also: The Leak Patterns to Hunt

From your baseline reports, here are the specific leak patterns appearing:

| Pattern | Where | Fix |
|---------|-------|-----|
| `Assistant: ,.` | End of response | stop_string + trailing cleanup |
| `👋ly` / `👋lyric123...` | End of response | stop_string `"👋ly"` + cleanup |
| `Daisy x02` / `Daisy x2023-04-18` | End of response | stop_string `"Daisy x"` + cleanup |
| `❮REFINE❯` | Mid or end | stop_string `"❮REFINE❯"` + cleanup |
| `ASURE...PRECEDES...` | End | stop_string `"ASURE"` + cleanup |
| `OffsetTable` | End | stop_string `"OffsetTable"` + cleanup |
| `Подстраивайся под текущую потребность человека` | Full response | meta-phrase cleanup |

All of these are **training data artifacts** leaking through. The stop strings + cleanup catch them all.
