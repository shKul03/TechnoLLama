from pathlib import Path
from typing import Dict, List
import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR
from llm.ollama_client import OllamaClient

class ClaimMetadataExtractor:
    def __init__(
        self,
        ollama_client: OllamaClient,
        prompt_text: str,
        intermediate_dir: Path,
        max_pages: int = 2,
        confidence_threshold: float = 0.5,
    ):
        self.ollama = ollama_client
        self.prompt = prompt_text
        self.max_pages = max_pages
        self.confidence_threshold = confidence_threshold
        self.intermediate_dir = intermediate_dir
        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True
        )

    def extract(self, pdf_path: Path) -> Dict:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        ocr_text = self._run_ocr(pdf_path)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        ocr_file = self.intermediate_dir / f"{pdf_path.stem}_ocr.txt"
        ocr_file.write_text(ocr_text, encoding="utf-8")
        print(f"OCR text saved → {ocr_file}")
        if not ocr_text.strip():
            print("OCR produced empty text")
        # Send OCR text to Ollama
        return self.ollama.extract_json(
            system_prompt=self.prompt,
            user_content=ocr_text,
        )
    # --------------------------------------------------
    # OCR 
    # --------------------------------------------------
    def _run_ocr(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        all_lines: List[str] = []
        pages_to_process = min(self.max_pages, len(doc))
        for page_index in range(pages_to_process):
            print(f"OCR processing page {page_index + 1}")
            page = doc[page_index]
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            #  Ensure RGB
            if pix.n == 4:
                img = img[:, :, :3]
            result = self.ocr.predict(img)[0]
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            for text, score in zip(texts, scores):
                if score >= self.confidence_threshold:
                    all_lines.append(text)
        print(f"OCR completed ({len(all_lines)} lines extracted)")
        return "\n".join(all_lines)
