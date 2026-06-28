# Daisy model API (Azure ML `/score`)

This repository implements the same **shape** of request/response as the Daisy website expects. For the full field list and product notes, see the reference docs in the Therapy-Multi-Agent project (`WEBSITE_API_CONTRACT.md`, `MODEL_PERSONA_INTEGRATION.md`).

For how this differs from a multi-agent stack and which Daisy routes call Azure ML vs the legacy CBT API, see [MULTI_AGENT_AND_ROUTING.md](./MULTI_AGENT_AND_ROUTING.md).

## Chat request (JSON)

- `message` — user text (required for normal chat).
- `history` — previous turns only (`role` + `content`), no current message.
- `max_tokens`, `temperature`, `locale` / `language`.
- `persona` — style key(s), e.g. `warm_friend` or two styles comma-separated: `warm_friend,wise_teacher` (same as the website).
- `onboarding_summary` — JSON object or string from onboarding. When present, `ai_profile.communication_style` may be an array of **1–2** style IDs; the scoring script **prefers this array** over the top-level `persona` field (see `MODEL_PERSONA_INTEGRATION.md` in Therapy-Multi-Agent).
- `user_context` — long-term memory string from the site/DB.
- `request_ai_profile` — `true` on first turn when the site wants `ai_profile` back.
- `request_type` — `weekly_report` or `dynamics_insights` for special flows.

## Chat response (JSON)

- `response` (string, required).
- `persona_used`, `protocol_used`, `language`, `model`.
- `ai_profile` (optional) — `summary`, `goals`, `concerns`, `communication_style`, `updatedAt`.
- `memory_update` (optional) — short strings for the site to persist.
- `disclaimer_ru` / `disclaimer_en`, `translation_enabled`, `debug_context`.

Secrets (`HF_TOKEN`, translator keys, Azure OpenAI) must be supplied via Azure ML environment variables or Key Vault references, not committed to git.
