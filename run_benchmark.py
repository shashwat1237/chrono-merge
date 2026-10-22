import json
import sys
import os
import re
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
import dateparser

# Silence third-party framework logs
logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from nlp_extractor import build_medspacy_pipeline, extract_clinical_mentions, filter_active_incidents, get_embed_model
from reconciliation_engine import reconcile_incidents
from evidence_validator import validate_cluster_evidence, build_llm_client

def compute_pairwise_metrics(gold_clusters, pred_clusters, all_incident_notes):
    """
    Computes pairwise clustering metrics:
    - True Positives (pairs correctly grouped together)
    - False Negatives / Missed Merges (pairs that belong together but were split)
    - False Positives / False Merges (pairs that belong to different incidents but were merged)
    """
    notes_list = sorted(list(all_incident_notes))
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    # Map note to gold cluster ID
    note_to_gold = {}
    for gid, nset in gold_clusters.items():
        for n in nset:
            note_to_gold[n] = gid
            
    # Map note to pred cluster ID
    note_to_pred = {}
    for pid, nset in pred_clusters.items():
        for n in nset:
            note_to_pred[n] = pid

    n = len(notes_list)
    for i in range(n):
        for j in range(i + 1, n):
            n1, n2 = notes_list[i], notes_list[j]
            same_gold = (n1 in note_to_gold and n2 in note_to_gold and note_to_gold[n1] == note_to_gold[n2])
            same_pred = (n1 in note_to_pred and n2 in note_to_pred and note_to_pred[n1] == note_to_pred[n2])
            
            if same_gold and same_pred:
                tp += 1
            elif same_gold and not same_pred:
                fn += 1  # Missed merge
            elif not same_gold and same_pred:
                fp += 1  # False merge
            else:
                tn += 1
                
    return tp, fn, fp, tn

def evaluate_naive_note_counter(cases):
    """
    Naive baseline: Counts every note with any fall-related keyword as a new incident.
    """
    keyword_pat = re.compile(r'\b(fall|falls|fell|falling|slipped|tripped|floor)\b', re.IGNORECASE)
    total_detected = 0
    for case in cases:
        for note in case["notes"]:
            if keyword_pat.search(note["text"]):
                total_detected += 1
    return total_detected

def evaluate_keyword_window_matcher(cases, window_hours: float = 4.0):
    """
    Simple keyword window baseline: Matches notes with fall keywords, then groups them
    if their log timestamps are within a flat 4-hour window.
    """
    keyword_pat = re.compile(r'\b(fall|falls|fell|falling|slipped|tripped|floor)\b', re.IGNORECASE)
    total_incidents = 0
    for case in cases:
        matching_notes = []
        for note in case["notes"]:
            if keyword_pat.search(note["text"]):
                dt = dateparser.parse(note["timestamp"]) or datetime.now()
                matching_notes.append(dt)
        matching_notes.sort()

        # Simple forward-window clustering
        clusters = []
        for dt in matching_notes:
            assigned = False
            for c in clusters:
                if abs((dt - c[0]).total_seconds()) <= window_hours * 3600:
                    c.append(dt)
                    assigned = True
                    break
            if not assigned:
                clusters.append([dt])
        total_incidents += len(clusters)
    return total_incidents

def run_reproducible_benchmark(dataset_path: str = "data/master_exhaustive_benchmark.json", mode: str = "full"):
    t_suite_start = time.perf_counter()
    print("=" * 80)
    print("CHRONOMERGE: REPRODUCIBLE SYNTHETIC BENCHMARK EVALUATION")
    print(f"Dataset: {dataset_path} | Mode: {mode.upper()}")
    print("=" * 80)

    # 1. Measure Cold-Start Pipeline Initialization (Loading spaCy & SentenceTransformer weights into RAM)
    t_model_init_start = time.perf_counter()
    nlp = build_medspacy_pipeline()
    try:
        embed_model = get_embed_model(require_model=True)
        model_loaded = True
    except Exception as e:
        embed_model = None
        model_loaded = False
        print("\n" + "!" * 80)
        print(f"[MODEL STATUS]: Embedding model initialization failed: {e}")
        print("[MODEL STATUS]: Operating in RULES-ONLY mode. Zero-shot semantic fallback will be disabled.")
        print("[NOTE FOR OFFLINE ENVIRONMENTS]: Pretrained weights for 'all-MiniLM-L6-v2' must be cached")
        print("                                 in the local HuggingFace cache before running without network access.")
        print("!" * 80 + "\n")
    model_init_elapsed = time.perf_counter() - t_model_init_start
    
    run_evidence_eval = (mode == "full")
    llm_client = None
    llm_init_elapsed = 0.0
    if run_evidence_eval:
        t_llm_start = time.perf_counter()
        try:
            llm_client = build_llm_client()
            if not llm_client:
                print("[NOTE] GROQ_API_KEY not found. Running Part A and baselines offline.")
                run_evidence_eval = False
        except Exception as e:
            print(f"[NOTE] Could not initialize Groq client ({e}). Running Part A and baselines offline.")
            run_evidence_eval = False
        llm_init_elapsed = time.perf_counter() - t_llm_start
    
    pipeline_init_elapsed = model_init_elapsed + llm_init_elapsed

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Incident Detection Metrics (Part A)
    total_cases = len(cases)
    exact_grouping_count = 0
    incident_count_matches = 0
    total_gold_incidents = 0
    total_pred_incidents = 0
    false_incident_detections = 0  # Incidents predicted when none occurred (e.g. ruled out)
    
    total_tp = 0
    total_fn_missed_merges = 0
    total_fp_false_merges = 0
    total_part_a_elapsed = 0.0
    
    # Evidence Validation Metrics (Part B) - Explicitly separated denominators
    total_gold_audit_targets = 0     # All gold incidents requiring evidence audit across benchmark
    matched_incidents_audited = 0    # Gold incidents successfully matched to a predicted cluster (unique targets)
    unmatched_gold_incidents = 0     # Gold incidents missed by detector or unassigned
    audit_attempts = 0               # Total HTTP requests dispatched to LLM (including rate-limit retries)
    audits_succeeded = 0             # Completed API responses parsed without exception
    api_failures = 0                 # Network/API exceptions after all retries exhausted
    evidence_finding_correct = 0     # Correct evidence finding (True/False)
    uncertainty_calib_correct = 0    # Correct uncertainty flag
    positive_evidence_clusters = 0   # Clusters reporting evidence_found=True
    positive_quotes_verified = 0     # Verified in source note
    positive_quotes_literal = 0      # Literally identical to source text
    positive_quotes_rejected = 0     # Rejected unverified quotes
    audit_exceptions = []            # API error logs
    coverage_exceptions = []         # Unmatched incident logs

    discrepancies = []
    cases_for_audit = []

    for case in cases:
        cid = case["case_id"]
        cfam = case.get("case_family", "General")
        
        # 1. Gold ground truth
        gold_clusters = {}
        all_incident_notes = set()
        for note in case["notes"]:
            if note.get("gold_is_incident") and note.get("gold_cluster_id"):
                gold_clusters.setdefault(note["gold_cluster_id"], set()).add(note["note_id"])
                all_incident_notes.add(note["note_id"])

        # 2. Extract & Reconcile (Part A)
        t_case_start = time.perf_counter()
        active_mentions = []
        for note in case["notes"]:
            extracted = extract_clinical_mentions(note, nlp)
            filtered = filter_active_incidents(extracted)
            active_mentions.extend(filtered)

        pred_clusters_raw = reconcile_incidents(active_mentions, distance_threshold=0.55)
        pred_clusters = {p_id: set(m.note_id for m in m_list) for p_id, m_list in pred_clusters_raw.items()}

        gold_k = len(gold_clusters)
        pred_k = len(pred_clusters)
        total_gold_incidents += gold_k
        total_pred_incidents += pred_k

        if gold_k == 0 and pred_k > 0:
            false_incident_detections += pred_k
        elif pred_k > gold_k:
            false_incident_detections += (pred_k - gold_k)

        if gold_k == pred_k:
            incident_count_matches += 1

        # Check exact grouping partition match
        g_partitions = sorted([sorted(list(s)) for s in gold_clusters.values()])
        p_partitions = sorted([sorted(list(s)) for s in pred_clusters.values()])
        is_exact_match = (g_partitions == p_partitions)

        if is_exact_match:
            exact_grouping_count += 1
        else:
            discrepancies.append({
                "case_id": cid,
                "case_family": cfam,
                "gold": g_partitions,
                "pred": p_partitions,
                "description": case.get("description", "")
            })

        # Pairwise metrics for notes present in incident pool
        tp, fn, fp, tn = compute_pairwise_metrics(gold_clusters, pred_clusters, all_incident_notes | set().union(*pred_clusters.values()))
        total_tp += tp
        total_fn_missed_merges += fn
        total_fp_false_merges += fp
        total_part_a_elapsed += (time.perf_counter() - t_case_start)

        # Store case artifacts for downstream audit
        cases_for_audit.append((case, pred_clusters_raw, gold_clusters))

    part_a_elapsed = total_part_a_elapsed

    # Compute Baselines
    naive_count = evaluate_naive_note_counter(cases)
    kw_count = evaluate_keyword_window_matcher(cases, window_hours=4.0)

    # 3. Downstream Evidence Validation (Part B - Greedy 1-to-1 Jaccard Overlap Matching)
    # Evaluates each ground-truth incident with audit requirements against its single best-matching
    # predicted cluster, eliminating duplicate audits of split fragments and preventing order-dependent bias.
    t_part_b_start = time.perf_counter()
    part_b_elapsed = 0.0

    if run_evidence_eval and llm_client:
        for case, pred_clusters_raw, gold_clusters in cases_for_audit:
            cid = case["case_id"]
            gold_ev_map = case.get("gold_evidence_validation", {})
            if not gold_ev_map:
                continue

            total_gold_audit_targets += len(gold_ev_map)

            if not pred_clusters_raw:
                # Detector missed all incidents for this case; record as coverage gaps (no API call made)
                for gid in gold_ev_map.keys():
                    unmatched_gold_incidents += 1
                    coverage_exceptions.append((cid, gid, "Case produced 0 predicted clusters; gold incident missed by detector."))
                continue

            # Greedy 1-to-1 Jaccard Overlap Matching
            # Pre-compute all valid overlap candidates between gold incidents and predicted clusters
            candidates = []
            for gid in gold_ev_map.keys():
                gold_notes = gold_clusters.get(gid, set())
                for p_id, m_list in pred_clusters_raw.items():
                    cluster_note_ids = set(m.note_id for m in m_list)
                    overlap = len(cluster_note_ids & gold_notes)
                    if overlap > 0:
                        union = len(cluster_note_ids | gold_notes)
                        jaccard = (overlap / union) if union > 0 else 0.0
                        candidates.append((jaccard, overlap, gid, p_id))

            # Sort descending by Jaccard score, then overlap count
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

            assigned_gids = set()
            assigned_pids = set()
            one_to_one_matched = {}

            for jaccard, overlap, gid, p_id in candidates:
                if gid not in assigned_gids and p_id not in assigned_pids:
                    assigned_gids.add(gid)
                    assigned_pids.add(p_id)
                    one_to_one_matched[gid] = p_id

            for gid, expected_ev in gold_ev_map.items():
                if gid in one_to_one_matched:
                    matched_incidents_audited += 1
                    best_pid = one_to_one_matched[gid]
                    best_m_list = pred_clusters_raw[best_pid]
                    seen_notes = {}
                    for m in best_m_list:
                        if m.note_id not in seen_notes:
                            seen_notes[m.note_id] = {"timestamp": m.timestamp, "raw_text": m.raw_text, "note_id": m.note_id}
                    cluster_notes = list(seen_notes.values())

                    val = None
                    try:
                        for attempt in range(3):
                            audit_attempts += 1
                            try:
                                val = validate_cluster_evidence(cluster_notes, llm_client)
                                break
                            except Exception as inner_e:
                                if "429" in str(inner_e) and attempt < 2:
                                    time.sleep(2.5 * (attempt + 1))
                                    continue
                                raise inner_e

                        audits_succeeded += 1
                        if val.evidence_found == expected_ev["expected_evidence_found"]:
                            evidence_finding_correct += 1
                        if val.uncertainty_flag == expected_ev["expected_uncertainty_flag"]:
                            uncertainty_calib_correct += 1
                        
                        if val.evidence_found:
                            positive_evidence_clusters += 1
                            if val.quote_verified_in_source:
                                positive_quotes_verified += 1
                                if val.is_literal_match:
                                    positive_quotes_literal += 1
                            else:
                                positive_quotes_rejected += 1
                    except Exception as e:
                        api_failures += 1
                        audit_exceptions.append((cid, gid, str(e)))
                else:
                    # Gold incident had no uniquely available cluster (missed or claimed by another incident; no API call dispatched)
                    unmatched_gold_incidents += 1
                    coverage_exceptions.append((cid, gid, "Incident missed by detector or overlapping cluster claimed by another gold incident; no unique cluster available to audit."))

        part_b_elapsed = time.perf_counter() - t_part_b_start

    # =========================================================================
    # INDEPENDENT REPORTING
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: INCIDENT DETECTION & RECONCILIATION METRICS (PART A - LOCAL CPU)")
    print("=" * 80)
    print(f"Total Test Cases:                     {total_cases}")
    print(f"Total Ground Truth Incidents:         {total_gold_incidents}")
    print(f"Total Predicted Incidents:            {total_pred_incidents}")
    print(f"Incident Count Accuracy:              {incident_count_matches}/{total_cases} ({incident_count_matches/total_cases*100:.1f}%)")
    print(f"Perfect Partition Grouping Accuracy:  {exact_grouping_count}/{total_cases} ({exact_grouping_count/total_cases*100:.1f}%)")
    print(f"Part A Active Elapsed Wall-Clock Time: {part_a_elapsed:.2f} seconds (Measured via time.perf_counter() across all cases)")
    print("-" * 80)
    print("PAIRWISE RECONCILIATION ERROR ANALYSIS:")
    print(f"  • True Positive Note Pairs Joined:   {total_tp}")
    print(f"  • False Merges (Overmerging Errors):  {total_fp_false_merges}  (Observed on this synthetic benchmark)")
    print(f"  • Missed Merges (Undermerging Errors):{total_fn_missed_merges}  (Same incident split into separate records)")
    print(f"  • Net False Incident Detections:      {false_incident_detections}  (Spurious incidents generated)")

    print("\n" + "=" * 80)
    print("REPRODUCIBLE ALGORITHMIC BASELINE COMPARISON")
    print("=" * 80)
    print(f"{'Method':<35} | {'Reported Falls':<14} | {'True Falls':<10} | {'Overcount Error'}")
    print("-" * 80)
    print(f"{'Naive Note Counter (Any Keyword)':<35} | {naive_count:<14} | {total_gold_incidents:<10} | +{(naive_count - total_gold_incidents)/total_gold_incidents*100:.1f}%")
    print(f"{'Simple 4-Hour Keyword Window':<35} | {kw_count:<14} | {total_gold_incidents:<10} | +{(kw_count - total_gold_incidents)/total_gold_incidents*100:.1f}%")
    print(f"{'ChronoMerge Engine':<35} | {total_pred_incidents:<14} | {total_gold_incidents:<10} | +{(total_pred_incidents - total_gold_incidents)/total_gold_incidents*100:.1f}%")
    print("=" * 80)

    if run_evidence_eval:
        print("\n" + "=" * 80)
        print("TASK 2: DOWNSTREAM EVIDENCE VALIDATION METRICS (PART B - GPT-OSS-120B VIA GROQ)")
        print("Method: Greedy 1-to-1 Jaccard Overlap Matching against Gold Standard")
        print("=" * 80)
        retry_count = max(0, audit_attempts - matched_incidents_audited)
        print(f"Total Gold Audit Requirements:        {total_gold_audit_targets}")
        print(f"Unique Cluster Audit Targets:         {matched_incidents_audited}")
        print(f"Unmatched / Missed Incidents:         {unmatched_gold_incidents}")
        print(f"Total LLM API Requests Dispatched:    {audit_attempts} ({matched_incidents_audited} initial + {retry_count} retries)")
        print(f"Completed Successful Audits:          {audits_succeeded}")
        print(f"Failed / Crashed Audits:              {api_failures}")
        print("-" * 80)
        if total_gold_audit_targets > 0:
            coverage_rate = (matched_incidents_audited / total_gold_audit_targets) * 100
            print(f"  1. Detector Audit Coverage Rate:    {matched_incidents_audited}/{total_gold_audit_targets} ({coverage_rate:.1f}%) [Denominator = Total Gold Audit Targets]")
        if audit_attempts > 0:
            api_rate = (audits_succeeded / audit_attempts) * 100
            print(f"  2. Request Completion Rate:         {audits_succeeded}/{audit_attempts} ({api_rate:.1f}%) [Denominator = Dispatched HTTP Requests ({audit_attempts})]")
        if audits_succeeded > 0:
            ev_acc = (evidence_finding_correct / audits_succeeded) * 100
            unc_acc = (uncertainty_calib_correct / audits_succeeded) * 100
            print(f"  3. Evidence Finding Accuracy:       {evidence_finding_correct}/{audits_succeeded} ({ev_acc:.1f}%) [Denominator = Completed Audits]")
            print(f"  4. Uncertainty Calibration Accuracy:{uncertainty_calib_correct}/{audits_succeeded} ({unc_acc:.1f}%) [Denominator = Completed Audits]")
        if total_gold_audit_targets > 0:
            e2e_acc = (evidence_finding_correct / total_gold_audit_targets) * 100
            print(f"  5. End-to-End Finding Accuracy:     {evidence_finding_correct}/{total_gold_audit_targets} ({e2e_acc:.1f}%) [Denominator = Total Gold Targets (Penalizes Missed Incidents)]")
        if positive_evidence_clusters > 0:
            q_rate = (positive_quotes_verified / positive_evidence_clusters) * 100
            lit_rate = (positive_quotes_literal / positive_evidence_clusters) * 100
            norm_count = positive_quotes_verified - positive_quotes_literal
            norm_rate = (norm_count / positive_evidence_clusters) * 100
            print(f"  6. Verbatim Quote Verification:     {positive_quotes_verified}/{positive_evidence_clusters} ({q_rate:.1f}%) [Denominator = Positive Evidence Clusters]")
            print(f"     - Literally Identical to Source: {positive_quotes_literal}/{positive_evidence_clusters} ({lit_rate:.1f}% character-for-character match)")
            print(f"     - Normalized Span Aligned:       {norm_count}/{positive_evidence_clusters} ({norm_rate:.1f}% casing/whitespace resolved to raw span)")
        else:
            print(f"  6. Verbatim Quote Verification:     N/A (0 clusters reported positive evidence)")

        if coverage_exceptions:
            print(f"\n[Coverage Gaps / Missed Audits]: {len(coverage_exceptions)} gold incidents had no unique matched cluster:")
            for cexc in coverage_exceptions:
                print(f"  - Case {cexc[0]} (Gold Incident {cexc[1]}): {cexc[2]}")
        if audit_exceptions:
            print(f"\n[API Call Exceptions]: {len(audit_exceptions)} errors during LLM network requests:")
            for exc in audit_exceptions:
                print(f"  - Case {exc[0]} (Cluster {exc[1]}): {exc[2]}")
    else:
        print("\n[NOTE] Task 2 Evidence Auditing was skipped (offline mode selected). Run with --mode full to evaluate LLM evidence audits.")

    total_suite_elapsed = time.perf_counter() - t_suite_start
    print("\n" + "=" * 80)
    print("SYSTEM EXECUTION PROFILING & HARDWARE TIMING")
    print("=" * 80)
    model_status_label = "MedSpaCy + SentenceTransformer into RAM" if model_loaded else "MedSpaCy only (Rules-only mode; SentenceTransformer offline)"
    if run_evidence_eval and llm_init_elapsed > 0:
        print(f"  • Pipeline and Client Initialization: {pipeline_init_elapsed:.2f} seconds (Models: {model_init_elapsed:.2f}s [{model_status_label}]; Client: {llm_init_elapsed:.2f}s)")
    else:
        print(f"  • Pipeline and Model Initialization:  {model_init_elapsed:.2f} seconds ({model_status_label})")
    print(f"  • Part A Active Inference:          {part_a_elapsed:.2f} seconds elapsed wall-clock time (20 Cases / 69 Notes extraction & clustering)")
    if run_evidence_eval:
        print(f"  • Part B Remote LLM Auditing (API): {part_b_elapsed:.2f} seconds ({matched_incidents_audited} unique targets across {audit_attempts} dispatched requests via Groq)")
    print(f"  • Total End-to-End Suite Runtime:   {total_suite_elapsed:.2f} seconds")
    print("=" * 80)

    # =========================================================================
    # EXPLICIT FAILURE REPORTING (THE 4 DISCREPANCIES)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"AUDIT OF NON-PERFECT CASES: {len(discrepancies)} SCENARIOS REQUIRING ATTENTION")
    print("=" * 80)
    for idx, d in enumerate(discrepancies, 1):
        print(f"\n[{idx}] Scenario: {d['case_id']} ({d['case_family']})")
        print(f"    Clinical Context: {d['description']}")
        print(f"    Expected Partitions: {d['gold']}")
        print(f"    System Partitions:   {d['pred']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChronoMerge Reproducible Benchmark")
    parser.add_argument("--mode", choices=["offline", "full"], default="full", help="Evaluation mode (offline=Part A local only, full=Part A + Groq LLM audit)")
    parser.add_argument("--dataset", default="data/master_exhaustive_benchmark.json", help="Path to benchmark JSON")
    args = parser.parse_args()

    run_reproducible_benchmark(dataset_path=args.dataset, mode=args.mode)

