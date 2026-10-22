import sys
import json
import logging
from pathlib import Path

logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from nlp_extractor import build_medspacy_pipeline, extract_clinical_mentions, filter_active_incidents
from reconciliation_engine import reconcile_incidents
from evidence_validator import validate_cluster_evidence, build_llm_client

def inspect_demo(case_id: str = "TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS"):
    benchmark_path = Path("data/master_exhaustive_benchmark.json")
    if not benchmark_path.exists():
        print(f"Error: Dataset not found at {benchmark_path}")
        return

    with open(benchmark_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    target_case = next((c for c in cases if c["case_id"] == case_id), None)
    if not target_case:
        target_case = cases[0]

    print("=" * 80)
    print("CHRONOMERGE: 60-SECOND EXECUTIVE INSPECTION DEMO")
    print("=" * 80)
    print(f"Case ID:        {target_case['case_id']}")
    print(f"Challenge Type: {target_case.get('case_family', 'General')}")
    print(f"Description:    {target_case.get('description', '')}")
    print("-" * 80)
    print("RAW INPUT CLINICAL NURSING NOTES:")
    for note in target_case["notes"]:
        print(f"  • [{note['timestamp']}] (Note {note['note_id']}): \"{note['text']}\"")
    
    print("-" * 80)
    # Expected Ground Truth
    gold_clusters = {}
    for note in target_case["notes"]:
        if note.get("gold_is_incident") and note.get("gold_cluster_id"):
            gold_clusters.setdefault(note["gold_cluster_id"], []).append(note["note_id"])

    print("EXPECTED GROUND TRUTH ANSWER:")
    print(f"  • Distinct Incidents: {len(gold_clusters)}")
    for gid, notes in gold_clusters.items():
        print(f"    - Incident {gid}: Notes {sorted(notes)}")
    if "gold_evidence_validation" in target_case:
        print("  • Expected Evidence Audit Requirements:")
        for gid, exp in target_case["gold_evidence_validation"].items():
            print(f"    - Incident {gid} -> Assessment Found: {exp['expected_evidence_found']} (Uncertainty: {exp['expected_uncertainty_flag']})")

    print("-" * 80)
    print("RUNNING CHRONOMERGE SYSTEM PIPELINE...")
    nlp = build_medspacy_pipeline()
    llm_client = build_llm_client()

    active_mentions = []
    for note in target_case["notes"]:
        extracted = extract_clinical_mentions(note, nlp)
        filtered = filter_active_incidents(extracted)
        active_mentions.extend(filtered)

    pred_clusters_raw = reconcile_incidents(active_mentions, distance_threshold=0.55)

    print("\nSYSTEM ANSWER (PREDICTED):")
    print(f"  • Identified Incidents: {len(pred_clusters_raw)}")
    cluster_audits = {}
    for cid, m_list in pred_clusters_raw.items():
        notes_in_cluster = sorted(list(set(m.note_id for m in m_list)))
        print(f"    - Proposed Incident Cluster {cid}: Notes {notes_in_cluster}")
        
        # Deduplicate notes in cluster
        seen_notes = {}
        for m in m_list:
            if m.note_id not in seen_notes:
                seen_notes[m.note_id] = {"timestamp": m.timestamp, "raw_text": m.raw_text, "note_id": m.note_id}
        cluster_notes = list(seen_notes.values())

        if llm_client:
            try:
                val = validate_cluster_evidence(cluster_notes, llm_client)
                cluster_audits[cid] = val
                print(f"      [LLM Compliance Audit]: Evidence Found: {val.evidence_found} | Uncertainty: {val.uncertainty_flag}")
                print(f"      [Clinical Reasoning]:   {val.reasoning}")
                if val.exact_quote:
                    match_type = "LITERAL EXACT" if val.is_literal_match else "NORMALIZED ALIGNED"
                    print(f"      [Verbatim Quote]:       \"{val.exact_quote}\"")
                    print(f"      [Quote Verification]:   VERIFIED in Note {val.source_note_id} (Offsets: {val.char_start}..{val.char_end}, {match_type})")
                elif val.unverified_quote:
                    print(f"      [Unverified Candidate]: \"{val.unverified_quote}\" (Failed source authentication; rejected from usable evidence)")
                    print(f"      [Quote Verification]:   FAILED (Unverified quote rejected)")
                else:
                    print(f"      [Verbatim Quote]:       None documented")
            except Exception as e:
                print(f"      [LLM Compliance Audit]: Error calling auditor: {e}")
        else:
            print("      [LLM Compliance Audit]: Skipped (GROQ_API_KEY not set; Part A local clustering completed).")

    print("-" * 80)
    # Verification Comparison
    g_sorted = sorted([sorted(v) for v in gold_clusters.values()])
    p_sorted = sorted([sorted(list(set(m.note_id for m in m_list))) for m_list in pred_clusters_raw.values()])
    is_grouping_match = (g_sorted == p_sorted)

    print("EVALUATION RESULT:")
    if is_grouping_match:
        print("  • Incident Grouping: PASS — Predicted partitions match Ground Truth with 100% precision.")
    else:
        print("  • Incident Grouping: DISCREPANCY — Predicted partitions deviated from Ground Truth.")

    # Check evidence audit against gold standard if present (Greedy 1-to-1 Jaccard Overlap Matching)
    gold_ev = target_case.get("gold_evidence_validation", {})
    if gold_ev and llm_client:
        candidates = []
        for gid in gold_ev.keys():
            gold_notes = set(gold_clusters.get(gid, []))
            for cid, m_list in pred_clusters_raw.items():
                c_notes = set(m.note_id for m in m_list)
                overlap = len(c_notes & gold_notes)
                if overlap > 0:
                    union = len(c_notes | gold_notes)
                    jaccard = (overlap / union) if union > 0 else 0.0
                    candidates.append((jaccard, overlap, gid, cid))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        assigned_gids = set()
        assigned_cids = set()
        one_to_one = {}
        for jaccard, overlap, gid, cid in candidates:
            if gid not in assigned_gids and cid not in assigned_cids:
                assigned_gids.add(gid)
                assigned_cids.add(cid)
                one_to_one[gid] = cid

        all_ev_pass = True
        for gid, exp in gold_ev.items():
            if gid in one_to_one and one_to_one[gid] in cluster_audits:
                best_cid = one_to_one[gid]
                val = cluster_audits[best_cid]
                ev_match = (val.evidence_found == exp["expected_evidence_found"] and val.uncertainty_flag == exp["expected_uncertainty_flag"])
                if not ev_match:
                    all_ev_pass = False
            else:
                all_ev_pass = False

        if all_ev_pass:
            print("  • Evidence Audit:    PASS — All cluster evidence evaluations matched ground truth expectations.")
        else:
            print("  • Evidence Audit:    DISCREPANCY — One or more clusters deviated from expected audit standard.")
    elif not llm_client:
        print("  • Evidence Audit:    SKIPPED — Set GROQ_API_KEY in .env to run Part B LLM evidence validation.")
    print("=" * 80)

if __name__ == "__main__":
    case_arg = sys.argv[1] if len(sys.argv) > 1 else "TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS"
    inspect_demo(case_arg)
