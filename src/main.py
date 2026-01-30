import json
from llm.ollama_client import OllamaClient
from extractors.claim_metadata_extractor import ClaimMetadataExtractor
from extractors.lineitems_pdf_extractor import LineItemsPdfExtractor
from extractors.lineitems_llm_extractor import LineItemsLLMExtractor
from reporting.word_report_builder import WordReportBuilder
from pathlib import Path

# ===============================
# PATH CONFIGURATION
# ===============================
DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
INTERMEDIATE_DIR = DATA_DIR / "pipeline_outputs"
SAMPLE_PDF = INPUT_DIR / "sample.pdf"
LINEITEMS_PDF = INPUT_DIR / "LineItems_PDF.pdf"
LINEITEMS_JSON = INTERMEDIATE_DIR / "extracted_LineItems_data.json"
LLM_OUTPUT_JSON = INTERMEDIATE_DIR / "llm_lineitems.json"
PROMPT_CLAIM = Path("prompts/claim_metadata_prompt.txt")
PROMPT_LINEITEMS = Path("prompts/line_items_prompt.txt")
LINEITEM_CONFIG = Path("config/line_results_config.json")
FINAL_DOC = OUTPUT_DIR / "claim_summary.docx"

def assert_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
# ==============================
# MAIN PIPELINE
# ===============================
def main():
    print("\n Starting AI Claim Processing Pipeline\n")
    # -------------------------------
    # PRE VALIDATIONS
    # -------------------------------
    assert_file(SAMPLE_PDF)
    assert_file(LINEITEMS_PDF)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ollama = OllamaClient(model="llama3.2:3b")
    # -------------------------------
    # STEP 1: CLAIM METADATA
    # -------------------------------
    print("Extracting claim metadata...")
    claim_prompt = PROMPT_CLAIM.read_text(encoding="utf-8")
    claim_extractor = ClaimMetadataExtractor(
        ollama_client=ollama,
        prompt_text=claim_prompt,
        intermediate_dir=INTERMEDIATE_DIR
    )
    claim_metadata = claim_extractor.extract(SAMPLE_PDF)
    if not any(v for v in claim_metadata.values()):
        print("Warning: Claim metadata extraction returned empty values")
    # -------------------------------
    # STEP 2: LINE ITEMS PDF → PAGE-WISE JSON
    # -------------------------------
    print("\n Extracting line items PDF (OCR + tables)")
    pdf_extractor = LineItemsPdfExtractor()
    pdf_extractor.extract(
        pdf_path=LINEITEMS_PDF,
        output_json=LINEITEMS_JSON,
    )
    print(f"Line items page-wise JSON saved → {LINEITEMS_JSON}")
    # -------------------------------
    # STEP PAGE-WISE JSON → STRUCTURED LINE ITEMS
    # -------------------------------
    print("\n Extracting structured line items via LLM")
    line_prompt = PROMPT_LINEITEMS.read_text(encoding="utf-8")
    line_items_extractor = LineItemsLLMExtractor(
        ollama_client=ollama,
        prompt_text=line_prompt,
    )
    line_items_json = line_items_extractor.extract(LINEITEMS_JSON)
    print("\n LLM Line Items JSON:")
    with open(LLM_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(line_items_json, f, indent=2)
    print(f"\n LLM Line Items JSON saved → {LLM_OUTPUT_JSON}")
    # -------------------------------
    # STEP 3: BUILD WORD REPORT
    # -------------------------------
    print("Building Word report...")
    report = WordReportBuilder(FINAL_DOC)
    report.add_claim_metadata(claim_metadata)
    # Line items table
    config = json.loads(LINEITEM_CONFIG.read_text(encoding="utf-8"))
    columns = config["columns"]
    services = (
    line_items_json.get("verification_of_treatment", {}).get("services_rendered", [])
    + line_items_json.get("billing_explanation", {}).get("services", [])
    )
    cpt_fix_map = {"97O1d": "97010"}
    for service in services:        
        service["modifier"] = service.get("modifier", "")
        service["schedule"] = service.get("description", "")
        service["rvu"] = service.get("rvu", 6)
        service["conversion_factor"] = service.get("conversion_factor", 9.55)
        service["charges"] = f"{float(service['rvu']) * float(service['conversion_factor']):.2f}"
        service["units"] = service.get("units", "")
        code = service.get("procedure_code")
        if code in cpt_fix_map:
            service["procedure_code"] = cpt_fix_map[code]
    report.add_line_items(services, columns)
    report.save()
    print(f"\n Report generated successfully: {FINAL_DOC}\n")

if __name__ == "__main__":
    main()
