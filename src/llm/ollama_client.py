from __future__ import annotations
import json
import requests
from typing import Dict, Any

class OllamaClient:
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        context_size: int = 8192,
        max_tokens: int = 2048,
        timeout: tuple[int, int] = (15, 600),
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = {
            "temperature": temperature,
            "num_ctx": context_size,
            "num_predict": max_tokens,
        }

    def extract_json(
        self,
        system_prompt: str,
        user_content: str,
    ) -> Dict[str, Any]:
        response_text = self._chat(system_prompt, user_content)
        return self._safe_json_parse(response_text)

    def _chat(self, system_prompt: str, user_content: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "format": "json",
            "stream": False,
            "options": self.options,
        }
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("Ollama response does not contain JSON")
        cleaned = text[start:end]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON returned by model:\n{cleaned}"
            ) from exc
