import json
from pathlib import Path

from llm.ollama_client import OllamaClient

# ===============================
# PATH CONFIGURATION
# ===============================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Inputs --------
BILL_OCR_TEXT = INPUT_DIR / "bill_ocr.txt"

# -------- Prompts --------
PROMPT_BILL_CLASSIFIER = Path("prompts/classifier/bill_classifier.txt")
PROMPT_INVOICE_EXTRACT = Path("prompts/extractors/invoice_extraction_prompt.txt")
PROMPT_EXPENSE_EXTRACT = Path("prompts/extractors/expense_extraction_prompt.txt")
PROMPT_INVOICE_API = Path("prompts/netsuite/invoice_netsuite_prompt.txt")
PROMPT_EXPENSE_API = Path("prompts/netsuite/expense_netsuite_prompt.txt")

# -------- Outputs --------
FINAL_JSON = OUTPUT_DIR / "bill_api_payload.json"


# ===============================
# UTILS
# ===============================

def assert_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_prompt(path: Path) -> str:
    assert_file(path)
    return path.read_text(encoding="utf-8")


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    print("\n🚀 Starting Bill Processing Pipeline\n")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    assert_file(BILL_OCR_TEXT)

    # -------------------------------
    # INIT LLM CLIENT
    # -------------------------------
    ollama = OllamaClient(model="llama3.2:3b")

    # -------------------------------
    # LOAD PROMPTS (PATHS DEFINED ABOVE)
    # -------------------------------
    classifier_prompt = load_prompt(PROMPT_BILL_CLASSIFIER)
    invoice_extract_prompt = load_prompt(PROMPT_INVOICE_EXTRACT)
    expense_extract_prompt = load_prompt(PROMPT_EXPENSE_EXTRACT)
    invoice_api_prompt = load_prompt(PROMPT_INVOICE_API)
    expense_api_prompt = load_prompt(PROMPT_EXPENSE_API)

    # -------------------------------
    # READ OCR TEXT
    # -------------------------------
    ocr_text = BILL_OCR_TEXT.read_text(encoding="utf-8")

    # -------------------------------
    # STEP 1: CLASSIFY BILL
    # -------------------------------
    classification = ollama.classify_bill(
        classifier_prompt=classifier_prompt,
        ocr_text=ocr_text,
    )

    document_type = classification.get("document_type")
    confidence = classification.get("confidence")

    print(f"→ Classified as: {document_type} (confidence: {confidence})")

    if document_type not in {"invoice", "expense"}:
        raise ValueError("Unsupported or unknown bill type")

    # -------------------------------
    # STEP 2: EXTRACT STRUCTURED DATA
    # -------------------------------
    structured_data = ollama.extract_financial_document(
        document_type=document_type,
        ocr_text=ocr_text,
        invoice_prompt=invoice_extract_prompt,
        expense_prompt=expense_extract_prompt,
    )

    # -------------------------------
    # STEP 3: CONVERT TO API FORMAT
    # -------------------------------
    if document_type == "invoice":
        api_payload = ollama.transform_to_api_payload(
            invoice_api_prompt,
            structured_data,
        )
    else:
        api_payload = ollama.transform_to_api_payload(
            expense_api_prompt,
            structured_data,
        )

    # -------------------------------
    # SAVE FINAL OUTPUT
    # -------------------------------
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(api_payload, f, indent=2)

    print(f"\n✅ Bill processed successfully")
    print(f"📦 Output saved to → {FINAL_JSON}\n")


if __name__ == "__main__":
    main()
