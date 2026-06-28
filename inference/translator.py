"""Optional Azure Translator + Azure OpenAI for multilingual chat (same priority idea as legacy stack)."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self) -> None:
        self.use_azure_translator = os.getenv("USE_AZURE_TRANSLATOR", "false").lower() == "true"
        self.use_openai = (
            os.getenv("USE_OPENAI_TRANSLATION", "false").lower() == "true"
            or os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
        )
        self.azure_key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.azure_endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
        self.azure_region = os.getenv("AZURE_TRANSLATOR_REGION", "global")
        self.openai_client = None
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

        if self.use_openai:
            try:
                from openai import AzureOpenAI

                ep = os.getenv("AZURE_OPENAI_ENDPOINT")
                key = os.getenv("AZURE_OPENAI_KEY")
                if ep and key:
                    self.openai_client = AzureOpenAI(
                        azure_endpoint=ep,
                        api_key=key,
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                    )
            except Exception as e:
                logger.warning("Azure OpenAI init failed: %s", e)

        if self.use_azure_translator and not self.azure_key:
            logger.warning("USE_AZURE_TRANSLATOR true but AZURE_TRANSLATOR_KEY missing")
            self.use_azure_translator = False

        self._cache: dict[str, str] = {}

    def detect_language(self, text: str) -> str:
        from reply_language import detect_intent_language

        return detect_intent_language(text) or "en"

    def translate(self, text: str, src: str, tgt: str, user_gender: str | None = None) -> str:
        if src == tgt or not text.strip():
            return text
        key = hashlib.md5(f"{text[:800]}|{src}|{tgt}|{user_gender}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        out = None
        if self.use_azure_translator and self.azure_key:
            try:
                out = self._azure_translate(text, src, tgt)
            except Exception as e:
                logger.warning("Azure Translator failed: %s", e)
        if out is None and self.openai_client:
            out = self._openai_translate(text, src, tgt, user_gender)
        if out is None:
            return text
        self._cache[key] = out
        return out

    def _azure_translate(self, text: str, src: str, tgt: str) -> str:
        lang_map = {"kk": "kk", "ru": "ru", "en": "en"}
        params = {"api-version": "3.0", "from": lang_map.get(src, src), "to": lang_map.get(tgt, tgt)}
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_key,
            "Ocp-Apim-Subscription-Region": self.azure_region,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }
        r = requests.post(
            self.azure_endpoint.rstrip("/") + "/translate",
            params=params,
            headers=headers,
            json=[{"text": text}],
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data[0]["translations"][0]["text"].strip()

    def _openai_translate(self, text: str, src: str, tgt: str, user_gender: str | None) -> str | None:
        if not self.openai_client:
            return None
        names = {"en": "English", "ru": "Russian", "kk": "Kazakh"}
        system = (
            f"Translate from {names.get(src, src)} to {names.get(tgt, tgt)}. Output only the translation."
        )
        if user_gender and tgt in ("ru", "kk") and src == "en":
            system += " Use correct grammatical gender for the user (" + user_gender + ")."
        resp = self.openai_client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=min(4096, len(text) + 500),
        )
        ch = resp.choices[0].message.content
        return ch.strip() if ch else None

    def generate_user_profile(
        self,
        onboarding_text: str,
        user_context: str,
        last_message: str | None,
    ) -> dict[str, Any] | None:
        if not self.openai_client:
            return _fallback_profile(onboarding_text, user_context)
        try:
            prompt = (
                "From the onboarding and context, produce JSON with keys: "
                "summary (string), goals (string array), concerns (string array), "
                "communication_style (string). No markdown."
                f"\n\nONBOARDING:\n{onboarding_text[:2000]}\n\nCONTEXT:\n{user_context[:1500]}\n\nLAST:\n{(last_message or '')[:500]}"
            )
            resp = self.openai_client.chat.completions.create(
                model=self._deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content or ""
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                return __import__("json").loads(raw[start : end + 1])
        except Exception as e:
            logger.warning("generate_user_profile failed: %s", e)
        return _fallback_profile(onboarding_text, user_context)

    def generate_memory_update(self, user_msg: str, assistant_msg: str, locale: str) -> list[str] | None:
        if not self.openai_client:
            return None
        try:
            resp = self.openai_client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "List 1-3 short memory bullets (facts) about the user from this exchange. JSON array of strings only.",
                    },
                    {
                        "role": "user",
                        "content": f"User ({locale}): {user_msg[:600]}\nAssistant: {assistant_msg[:600]}",
                    },
                ],
                temperature=0.2,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or "[]"
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                arr = __import__("json").loads(raw[start : end + 1])
                if isinstance(arr, list):
                    return [str(x) for x in arr[:3]]
        except Exception as e:
            logger.warning("generate_memory_update failed: %s", e)
        return None

    def summarize_memory(self, user_context: str) -> str:
        if not self.openai_client or len(user_context) <= 1500:
            return user_context
        try:
            resp = self.openai_client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize these user memory facts in <=800 chars, keep names and key facts:\n{user_context[:8000]}",
                    }
                ],
                temperature=0.2,
                max_tokens=400,
            )
            t = resp.choices[0].message.content
            return t.strip() if t else user_context
        except Exception:
            return user_context[:1500]


def _fallback_profile(onboarding_text: str, user_context: str) -> dict[str, Any]:
    summary = (onboarding_text or user_context or "").strip()[:500]
    if not summary:
        return {}
    return {
        "summary": summary,
        "goals": [],
        "concerns": [],
        "communication_style": "flexible_companion",
    }
