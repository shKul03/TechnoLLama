from __future__ import annotations
import json
import requests
from typing import Dict, Any, Literal


DocumentType = Literal["invoice", "expense", "unknown"]


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
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

    # -----------------------------
    # Generic JSON extractor
    # -----------------------------
    def extract_json(
        self,
        system_prompt: str,
        user_content: str,
    ) -> Dict[str, Any]:
        response_text = self._chat(system_prompt, user_content)
        return self._safe_json_parse(response_text)

    # -----------------------------
    # STEP 1: Document classification
    # -----------------------------
    def classify_bill(
        self,
        classifier_prompt: str,
        ocr_text: str,
    ) -> Dict[str, Any]:
        """
        Expected output:
        {
          "document_type": "invoice" | "expense" | "unknown",
          "category": "...",
          "confidence": "low" | "medium" | "high"
        }
        """
        return self.extract_json(
            system_prompt=classifier_prompt,
            user_content=ocr_text,
        )

    # -----------------------------
    # STEP 2: Domain extraction
    # -----------------------------
    def extract_financial_document(
        self,
        document_type: DocumentType,
        ocr_text: str,
        invoice_prompt: str,
        expense_prompt: str,
    ) -> Dict[str, Any]:
        if document_type == "invoice":
            return self.extract_json(invoice_prompt, ocr_text)

        if document_type == "expense":
            return self.extract_json(expense_prompt, ocr_text)

        raise ValueError("Unsupported or unknown document type")

    # -----------------------------
    # Optional STEP 3: API mapping
    # -----------------------------
    def transform_to_api_payload(
        self,
        api_prompt: str,
        structured_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.extract_json(
            system_prompt=api_prompt,
            user_content=json.dumps(structured_data),
        )

    # -----------------------------
    # Low-level chat call
    # -----------------------------
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
