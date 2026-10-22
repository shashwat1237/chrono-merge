import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from nlp_extractor import build_medspacy_pipeline, extract_clinical_mentions, filter_active_incidents
from reconciliation_engine import reconcile_incidents
from evidence_validator import validate_cluster_evidence, build_llm_client

def run_full_pipeline(data_filepath: str):
    nlp = build_medspacy_pipeline()
    llm_client = build_llm_client()
    
    with open(data_filepath, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} benchmark cases.\n" + "=" * 60)

    for case in cases:
        print(f"\nProcessing Case: {case['case_id']} ({case['case_family']})")
        case_active_mentions = []

        for note in case["notes"]:
            case_active_mentions.extend(filter_active_incidents(extract_clinical_mentions(note, nlp)))
            
        if not case_active_mentions:
            print("  --> No active incident mentions found.")
            continue

        clusters = reconcile_incidents(case_active_mentions, distance_threshold=0.55)
        print(f"  --> Identified {len(clusters)} distinct incident(s).")

        for cluster_id, mentions in clusters.items():
            print(f"\n  Evaluating Cluster {cluster_id} ({len(mentions)} notes):")
            cluster_notes = [{"note_id": m.note_id, "timestamp": m.timestamp, "raw_text": m.raw_text} for m in mentions]
            
            try:
                validation = validate_cluster_evidence(cluster_notes, llm_client)
                print(f"    Evidence Found: {validation.evidence_found}")
                print(f"    Uncertainty: {validation.uncertainty_flag}")
                print(f"    Reasoning: {validation.reasoning}")
                if validation.exact_quote:
                    print(f"    Quote: '{validation.exact_quote}'")
            except Exception as e:
                print(f"    LLM Validation Failed: {e}")

if __name__ == "__main__":
    benchmark_path = Path("data/master_exhaustive_benchmark.json")
    if not benchmark_path.exists():
        benchmark_path = Path("data/benchmark_cases.json")
    if benchmark_path.exists():
        run_full_pipeline(str(benchmark_path))
    else:
        print(f"File not found: {benchmark_path}")
