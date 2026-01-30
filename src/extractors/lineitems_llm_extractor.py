from pathlib import Path
from typing import Dict, List
import json
from llm.ollama_client import OllamaClient

class LineItemsLLMExtractor:
    def __init__(
        self,
        ollama_client: OllamaClient,
        prompt_text: str,
    ):
        self.ollama = ollama_client
        self.prompt = prompt_text

    def extract(self, pagewise_json_path: Path) -> Dict:
        pages = self._load_pages(pagewise_json_path)
        final_output: Dict = {}
        for page in pages:
            page_payload = json.dumps(page, ensure_ascii=False, indent=2)
            page_result = self.ollama.extract_json(
                system_prompt=self.prompt,
                user_content=page_payload,
            )
            self._deep_merge(final_output, page_result)
        return final_output

    @staticmethod
    def _load_pages(path: Path) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected page-wise JSON array")
        return data

    @staticmethod
    def _deep_merge(base: Dict, incoming: Dict):
        for key, value in incoming.items():
            if isinstance(value, dict):
                base.setdefault(key, {})
                LineItemsLLMExtractor._deep_merge(base[key], value)
            elif isinstance(value, list):
                base.setdefault(key, [])
                base[key].extend(value)
            else:
                if value not in ("", None):
                    base[key] = value
