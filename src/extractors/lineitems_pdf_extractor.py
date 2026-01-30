from pathlib import Path
import json
import os
import tempfile
import pdfplumber
import pytesseract
import fitz
import numpy as np
from PIL import Image

class LineItemsPdfExtractor:
    def __init__(self, tesseract_path: str | None = None):
        if os.name == "nt" and tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract(self, pdf_path: Path, output_json: Path) -> None:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                text = page.extract_text(layout=True)
                # -------- Page-specific OCR fallback --------
                if not text or len(text.strip()) < 10:
                    if page_number == 1:
                        text = self._ocr_page_with_pdfplumber(page)
                    elif page_number == 2:
                        text = self._ocr_page_with_pymupdf(pdf_path, page_number)
                pages.append({
                    "page_number": page_number,
                    "plain_text": text or "",
                    "tables": page.extract_tables()
                })
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(pages, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Line-items page JSON created → {output_json}")

    # ---------------- OCR helpers ----------------
    def _ocr_page_with_pdfplumber(self, page) -> str:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_path = tmp.name
            page.to_image(resolution=400).save(img_path)
            img = Image.open(img_path).convert("L")
            img = img.point(lambda x: 0 if x < 180 else 255, "1")
            text = pytesseract.image_to_string(img, config="--psm 3")
            os.remove(img_path)
            return text.strip()
        except Exception:
            return ""

    def _ocr_page_with_pymupdf(self, pdf_path: Path, page_number: int) -> str:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
        is_scanned = np.array(img).std() < 40
        config = "--oem 3 --psm 3" if is_scanned else "--oem 3 --psm 6"
        return pytesseract.image_to_string(img, config=config).strip()
