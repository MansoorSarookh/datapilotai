"""
DataPilot AI — Multi-Provider LLM Router
Routes LLM requests to available providers with automatic fallback.
Supports: Groq, Google Gemini, Ollama (local), Rule-based heuristics.
"""

import streamlit as st
from typing import Optional, List, Dict

# Provider configurations
PROVIDERS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "display": "Groq (llama-3.3-70b)",
        "priority": 1,
    },
    "gemini": {
        "model": "gemini-2.0-flash",
        "display": "Google Gemini Flash",
        "priority": 2,
    },
    "ollama": {
        "model": "llama3.2",
        "display": "Ollama (Local)",
        "priority": 3,
    },
}


class LLMRouter:
    """Routes LLM requests to available providers with automatic fallback."""

    def __init__(self):
        self._clients = {}
        self._preferred = None

    def set_preferred_provider(self, provider: str):
        self._preferred = provider if provider in PROVIDERS else None

    def get_preferred_provider(self) -> Optional[str]:
        return self._preferred

    def get_available_providers(self) -> List[str]:
        available = []
        for name in PROVIDERS:
            if self._get_client(name) is not None:
                available.append(name)
        return available

    def _get_client(self, provider: str):
        if provider in self._clients:
            return self._clients[provider]

        client = None
        if provider == "groq":
            client = self._init_groq()
        elif provider == "gemini":
            client = self._init_gemini()
        elif provider == "ollama":
            client = self._init_ollama()

        self._clients[provider] = client
        return client

    def _init_groq(self):
        try:
            from groq import Groq
            api_key = st.secrets.get("GROQ_API_KEY", "")
            if api_key:
                return Groq(api_key=api_key)
        except Exception:
            pass
        return None

    def _init_gemini(self):
        try:
            import google.generativeai as genai
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                return genai
        except Exception:
            pass
        return None

    def _init_ollama(self):
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                return "ollama_active"
        except Exception:
            pass
        return None

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 600,
        temperature: float = 0.4,
        provider: Optional[str] = None,
    ) -> str:
        """Send chat request to LLM with automatic fallback."""
        # Determine provider order
        if provider and provider in PROVIDERS:
            order = [provider]
        elif self._preferred:
            order = [self._preferred] + [p for p in PROVIDERS if p != self._preferred]
        else:
            order = sorted(PROVIDERS.keys(), key=lambda p: PROVIDERS[p]["priority"])

        for prov in order:
            client = self._get_client(prov)
            if client is None:
                continue
            try:
                if prov == "groq":
                    return self._chat_groq(client, messages, max_tokens, temperature)
                elif prov == "gemini":
                    return self._chat_gemini(client, messages, max_tokens, temperature)
                elif prov == "ollama":
                    return self._chat_ollama(messages, max_tokens, temperature)
            except Exception:
                continue

        return None  # All providers failed

    def _chat_groq(self, client, messages, max_tokens, temperature) -> str:
        model = PROVIDERS["groq"]["model"]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _chat_gemini(self, genai_module, messages, max_tokens, temperature) -> str:
        model_name = PROVIDERS["gemini"]["model"]
        model = genai_module.GenerativeModel(model_name)
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")
        combined = "\n\n".join(prompt_parts)
        response = model.generate_content(
            combined,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response.text.strip()

    def _chat_ollama(self, messages, max_tokens, temperature) -> str:
        import requests
        model = PROVIDERS["ollama"]["model"]
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=30,
        )
        return response.json()["message"]["content"].strip()

    def get_active_provider_display(self) -> str:
        """Return the display name of the first available provider."""
        for prov in sorted(PROVIDERS.keys(), key=lambda p: PROVIDERS[p]["priority"]):
            if self._get_client(prov) is not None:
                return PROVIDERS[prov]["display"]
        return "Rule-based Fallback"


# Singleton instance
@st.cache_resource
def get_llm_router() -> LLMRouter:
    return LLMRouter()
