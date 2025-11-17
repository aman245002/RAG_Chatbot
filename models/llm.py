# models/llm.py
"""
Groq-based LLM wrapper. Uses the groq SDK if installed; otherwise falls back
to a simple HTTP POST to Groq's OpenAI-compatible endpoint.

Set GROQ_API_KEY in your .env or Streamlit secrets. You can select provider
by setting DEFAULT_LLM in config (values: 'groq' or 'openai') — default is 'groq'.
"""

import os
import json
from typing import Optional
from config.config import config

# Try SDK import first
_HAS_GROQ_SDK = False
try:
    from groq import Groq
    _HAS_GROQ_SDK = True
except Exception:
    _HAS_GROQ_SDK = False

# requests fallback
import requests

GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_API_BASE}/chat/completions"


class GroqClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY missing. Set it in .env or Streamlit secrets.")
        self.model = model or getattr(config, "GROQ_MODEL", None) or "llama-3.3-70b-versatile"

        if _HAS_GROQ_SDK:
            try:
                # The Groq SDK uses Groq() or GroqClient depending on version; use Groq() here
                self.client = Groq(api_key=self.api_key)
            except Exception:
                # last resort: don't crash, use http fallback
                self.client = None
        else:
            self.client = None

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
        """
        Use SDK if available, otherwise call the HTTP endpoint directly.
        Returns plain text answer (first choice).
        """
        messages = [{"role": "user", "content": prompt}]

        # Try SDK
        if self.client is not None:
            try:
                # many SDK versions expose .chat.completions.create
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # SDK returns an object with choices similar to OpenAI
                try:
                    return resp.choices[0].message.content.strip()
                except Exception:
                    # fallback string conversion
                    return str(resp)
            except Exception as e:
                # fallback to HTTP below
                pass

        # HTTP fallback
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = requests.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            # include response text to help debugging
            raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")
        data = r.json()
        # OpenAI-compatible format: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # last-resort: return whole body
            return json.dumps(data)


class OpenAIClientFallback:
    """
    Minimal OpenAI wrapper (legacy) — kept for compatibility if you want to
    use OpenAI later. It expects openai>=1.0.0 installed and configured.
    """
    def __init__(self, model: Optional[str] = None):
        try:
            from openai import OpenAI as _OpenAI
        except Exception:
            raise RuntimeError("OpenAI SDK not installed. Install `openai` if you want to use OpenAI provider.")
        api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing.")
        self.client = _OpenAI(api_key=api_key)
        self.model = model or getattr(config, "OPENAI_MODEL", None) or "gpt-4o-mini"

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            return str(resp)


# Public factory that chooses provider
def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None):
    """
    provider: 'groq' (default) or 'openai'
    """
    if provider is None:
        provider = getattr(config, "DEFAULT_LLM", "groq").lower()

    if provider == "groq":
        return GroqClient(model=model)
    elif provider == "openai":
        return OpenAIClientFallback(model=model)
    else:
        raise ValueError("Unsupported provider. Choose 'groq' or 'openai'.")


# Convenience class name used elsewhere in code (keeps previous imports working)
class OpenAIClientAlias:
    """
    Adapter so existing code that imports OpenAIClient will still work.
    By default this returns a Groq-backed client (so you don't need to
    change the rest of your code).
    """
    def __init__(self, model: Optional[str] = None):
        # use provider from config or default to groq
        provider = getattr(config, "DEFAULT_LLM", "groq").lower()
        self._client = get_llm_client(provider=provider, model=model)

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
        return self._client.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

# --- compatibility alias (preserve existing imports) -----------------------
# Many parts of the code expect `from models.llm import OpenAIClient`.
# Provide a backwards-compatible name that wraps the selected provider.
class OpenAIClient(OpenAIClientAlias):
    """
    Backwards-compatible alias: existing code can import OpenAIClient and get
    the configured provider (Groq by default). No changes required elsewhere.
    """
    pass
