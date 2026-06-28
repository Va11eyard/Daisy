# Daisy AI — Plain-Language Overview

This page explains what Daisy is and how it works **without technical jargon**. For file names, APIs, and architecture detail, see [PITCH_TECHNICAL.md](PITCH_TECHNICAL.md).

---

## What Daisy is (in one breath)

Daisy is a mental health–focused AI companion that runs on **your own cloud setup** (Azure), not as a generic chatbot pasted on top of someone else’s API. Messages for the main conversation are processed inside that environment. The AI is **trained on clinical and therapy-related material**, turned into natural dialogues—not on “the whole internet” like a general assistant.

---

## How a message gets answered (the journey)

When you send text in the app, a few things happen in order:

1. **Sign-in and safety checks.** The server knows who you are. Obvious crisis wording can get a **fixed, careful reply** right away (and resources), without waiting for the AI model.

2. **Private details softened.** If the message looks like it contains things you wouldn’t want stored raw (certain IDs, phone patterns, etc.), those get **redacted** before storage, and the app logs that something was caught—not the secret itself.

3. **Context is gathered.** Daisy pulls in your **recent chat history**, **onboarding** (what you shared when you started), **memory** from past sessions, and a **short psychological profile snapshot** when available—not just the latest sentence.

4. **The AI model runs on your Azure endpoint.** All of that is packaged and sent to **your** fine-tuned model. That model is **not** ChatGPT, Claude, or Google’s API for this main step.

5. **The reply is cleaned up.** The system trims the answer so it ends cleanly (for example at a full sentence), then the app saves Daisy’s reply and any updates to memory.

So: **your words → safety and privacy steps → rich context → your model → saved answer.** Nothing in that core path depends on a public commercial chat API.

---

## The “five moods” of a conversation

Daisy doesn’t talk the same way in every turn. It tries to notice **what kind of moment** the chat is in:

- **Intake** — opening space, gentle questions, not rushing to label or fix.
- **Disclosure** — you’re sharing something heavy; Daisy focuses on **witnessing**, not jumping to advice or silver linings.
- **Psychoeducation** — you’re asking *why* or *how does this work*; Daisy can explain in plain language, step by step.
- **Action planning** — you want **concrete next steps**; replies stay small and realistic, not a laundry list of life changes.
- **Crisis** — safety first: clear language and help lines, not metaphor or therapy theory.

Rules built into the product discourage hollow phrases (“I hear you!” as empty filler), push for **more precise words** for feelings, and usually **one question at the end** so the conversation stays focused. That layer is separate from “which AI model” you use: even a different model would still need this **behavior shell** to feel like Daisy.

---

## How Daisy was trained (simple version)

- **Sources:** Notes and books from real clinical traditions (things like CBT, DBT, ACT, attachment, crisis protocols), organized into **about two dozen** clinical source folders and **over a hundred** distilled insight threads, then turned into **practice dialogues** in Daisy’s voice—not copy-pasted textbook pages.

- **Size:** The current production training set is **387** supervised examples, about **half Russian and half English**, by design.

- **Method:** The team fine-tunes a **7 billion parameter** open model with an efficient method (LoRA), so behavior shifts toward therapy-style dialogue without rebuilding a giant model from zero.

- **Versions:** When bad data slipped into an older training run (version 9), the team **threw the dataset away and rebuilt** (version 10). That’s normal product discipline: fix the data, not just the marketing.

---

## Privacy and where data lives

- **Cloud:** Daisy is built to run on **Azure**, with encrypted connections between the app and the brain of the system. Where exactly servers sit (including regions you care about) is a **deployment setting** your team controls.

- **No vendor lock-in for the main brain:** The core assistant does not **require** OpenAI, Anthropic, or Google for its main answers. Optional tools (for example translation) may use other services **only if you turn them on**—they are not the default therapist.

---

## Memory and personalization

Daisy is meant to remember **themes over time**—work stress, sleep, relationships—not only the last message. The app stores structured **memory items**, **conversation history**, and a **psych profile snapshot** so replies can align with how stabilized or overwhelmed the system thinks the user feels (and suggest calmer steps vs. problem-solving when appropriate).

A **future** idea is importing old chats from other assistants so Daisy can see **long-running patterns** (“this theme shows up for a long time”) without treating an old export as a current crisis. That import path is still **in development**.

---

## Languages

**Russian and English** are built into training and checks. **Kazakh** is partially prepared in the product (locale and some text paths), but full therapy-quality Kazakh is on the **roadmap**, not finished yet.

---

## Why competitors can’t copy this overnight

1. **The “wrapper” matters.** Tone rules, safety order, conversation phases, and memory matter as much as the raw model.

2. **The dataset is yours.** The distilled insights and bilingual dialogues are not a public download.

3. **Time.** A new user of a generic model has **no history**; Daisy’s value grows as **your** saved context grows.

---

## Honest limits

- A **7B** model is strong for supportive chat but is **not** the same as the largest frontier models for heavy reasoning.

- Daisy is **supportive**, not a replacement for therapists, doctors, or emergency services—especially in crisis.

- Some topic areas (e.g. deep somatic work, structured grief programs, ADHD-specific tracks) are **not fully covered** yet; they’re planned for later versions.

- **Kazakh** support is not complete end-to-end yet.

---

*For the same story with architecture names and code pointers, read [PITCH_TECHNICAL.md](PITCH_TECHNICAL.md).*
