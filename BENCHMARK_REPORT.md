# ChronoMerge: Synthetic Benchmark Evaluation & Technical Audit Report
**Project**: Clinical Incident Reconciliation & Downstream Evidence Engine  
**Dataset**: Synthetic Multi-Author EHR Benchmark (`data/master_exhaustive_benchmark.json`)  
**Auditor**: Independent Technical Evaluation  

---

## 1. Executive Summary & Claims Boundary

This evaluation measures the performance of **ChronoMerge** on a curated, synthetic benchmark designed to reflect the challenges of long-term care clinical documentation.

### What This Benchmark Proves:
- Demonstrates algorithmic handling of multi-author notes, same-day multiple incidents, retrospective references, and roommate separation under controlled test conditions.
- Demonstrates that structured clustering substantially reduces duplicate counting compared to naive note counting baselines.
- Verifies that large language models (`openai/gpt-oss-120b`) can extract verbatim proof quotes and calibrate clinical uncertainty.

### What This Benchmark Does NOT Prove:
- **This is a synthetic benchmark, not clinical or real-world validation.** Actual EHR records contain regional abbreviations, unstructured templates, non-standard timestamps, and noise that synthetic benchmarks cannot fully replicate.
- **ChronoMerge is a promising research prototype, not a production-proven clinical solution.** The reported metrics (95% count accuracy, 80% perfect grouping across 20 synthetic scenarios) demonstrate prototype feasibility under controlled conditions. Four scenarios produce imperfect groupings (detailed in Section 4), and the system has not been tested in live healthcare facilities.
- **No claim of human-level performance.** Without a formal, double-blinded study comparing licensed nurse auditors against the system on identical unlabelled real-world charts, claims of "human-level equivalence" are ungrounded.

---

## 2. Reproducibility Guide

The benchmark can be reproduced with a single command.

> **Execution Boundary Notice**:  
> - **Task 1: Incident Detection & Reconciliation (Part A)** executes **100% locally on CPU** using open-source NLP libraries (`medspacy`, `sentence-transformers`, `scikit-learn`, `dateparser`). It is completely offline with zero external network or API dependencies.  
> - **Task 2: Downstream Evidence Auditing (Part B)** calls an **external cloud model** (`openai/gpt-oss-120b` hosted on Groq) via API for structured Pydantic validation.  
> - **The pipeline is hybrid, not entirely offline**: incident reconciliation is fully local/offline, while evidence auditing requires internet connectivity and a valid `GROQ_API_KEY`.

### Environment & Pinned Dependencies (Python 3.11+)
```powershell
pip install -r requirements.txt
```
To run the full pipeline including downstream LLM evidence audits, set your Groq API key in `.env`:
```env
GROQ_API_KEY=your_key_here
```

### Run Option A: Offline Deterministic Benchmark (No API Key Required)
```powershell
python run_benchmark.py --mode offline
```
*Executes all 20 resident cases (69 clinical notes) locally on CPU in ~4.41 seconds active elapsed wall-clock time (~9.51 seconds total including 5.10s model initialization into RAM). Runs Part A incident reconciliation and computes the algorithmic baseline comparison without requiring network access or API credentials.*

### Run Option B: Full Pipeline Suite (Includes Groq LLM Evidence Audit)
```powershell
python run_benchmark.py --mode full
```
*Runs Part A local reconciliation (~4.41s active elapsed wall-clock time) plus Part B structured Pydantic compliance audits using `openai/gpt-oss-120b` via Groq (~74.93s across 11 unique targets in 11 dispatched requests, total suite ~89.66 seconds including 10.16s pipeline and client initialization), enforcing programmatic Python substring verification directly on raw text for every returned quote.*

### Run the Instant 60-Second Demo (Single Command)
```powershell
python demo.py TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS
```
*Displays raw input notes, expected ground truth, system prediction, verbatim quotes with exact character offset verification, and independent grouping/evidence audit results.*

---

## 3. Independent Task Performance Metrics

In accordance with clinical system evaluation standards, **Incident Detection (Part A)** is measured independently from **Evidence Validation (Part B)**.

```
================================================================================
TASK 1: INCIDENT DETECTION & RECONCILIATION METRICS (PART A - LOCAL CPU)
================================================================================
Dataset Scale:                        20 Test Cases (69 Clinical Notes)
Total Ground Truth Incidents:         24
Total Predicted Incidents:            25

Case-Level Metrics:
  • Incident Count Accuracy:          19 / 20 cases (95.0%)
  • Perfect Partition Grouping:       16 / 20 cases (80.0%)
  • Part A Active Wall-Clock Time:    4.41 seconds (Measured elapsed wall time via time.perf_counter() across all cases)

Pairwise Reconciliation Errors (Total Note Pairs Evaluated = 124):
  • True Positive Pairs Joined:       47 pairs correctly unified
  • False Merges (Overmerging):        0 pairs  (Zero observed on this synthetic benchmark)
  • Missed Merges (Undermerging):      6 pairs  (Same incident split across records)
  • Net False Incident Detections:     1 event  (Spurious incident from edge case)
================================================================================
```

> **Why False Merges = 0 Matters:**  
> In healthcare compliance, **merging two distinct falls into one (a false merge) is far more dangerous than splitting one fall into two (a missed merge)**. An overmerge silently erases an incident and its required post-fall safety protocol. On this synthetic benchmark, ChronoMerge achieved **zero false merges observed across all 20 test scenarios**. Note that this reflects empirical results on this curated benchmark, not a universal guarantee that false merges will never occur in all clinical documentation.

```
================================================================================
REPRODUCIBLE ALGORITHMIC BASELINE COMPARISON
================================================================================
Method                              | Reported Falls | True Falls | Overcount Error
--------------------------------------------------------------------------------
Naive Note Counter (Any Keyword)    | 61             | 24         | +154.2%
Simple 4-Hour Keyword Window        | 33             | 24         | +37.5%
ChronoMerge Engine                  | 25             | 24         | +4.2%
================================================================================
*(Generated directly by running `python run_benchmark.py --mode offline` on data/master_exhaustive_benchmark.json)*
```

```
================================================================================
TASK 2: DOWNSTREAM EVIDENCE VALIDATION METRICS (PART B - GPT-OSS-120B VIA GROQ)
Method: Greedy 1-to-1 Jaccard Overlap Matching against Gold Standard
================================================================================
Total Gold Audit Requirements:        11
Unique Cluster Audit Targets:         11
Unmatched / Missed Incidents:         0
Total LLM API Requests Dispatched:    11 (11 initial + 0 retries)
Completed Successful Audits:          11
Failed / Crashed Audits:              0
--------------------------------------------------------------------------------
  1. Detector Audit Coverage Rate:    11/11 (100.0%) [Denominator = Total Gold Audit Targets]
  2. Request Completion Rate:         11/11 (100.0%) [Denominator = Dispatched HTTP Requests (11)]
  3. Evidence Finding Accuracy:       11/11 (100.0%) [Denominator = Completed Audits]
  4. Uncertainty Calibration Accuracy:10/11 (90.9%)  [Denominator = Completed Audits]
  5. End-to-End Finding Accuracy:     11/11 (100.0%) [Denominator = Total Gold Targets (Penalizes Missed Incidents)]
  6. Verbatim Quote Verification:     8/8   (100.0%) [Denominator = Positive Evidence Clusters]
     - Literally Identical to Source: 8/8   (100.0% character-for-character match)
     - Normalized Span Aligned:       0/8   (0.0% casing/whitespace resolved to raw span)
================================================================================
SYSTEM EXECUTION PROFILING & HARDWARE TIMING
================================================================================
  • Pipeline and Client Initialization: 10.16 seconds (Models: 5.10s [MedSpaCy + SentenceTransformer into RAM]; Client: 5.06s)
  • Part A Active Inference:          4.41 seconds elapsed wall-clock time (20 Cases / 69 Notes extraction & clustering)
  • Part B Remote LLM Auditing (API): 74.93 seconds (11 unique targets across 11 dispatched requests via Groq)
  • Total End-to-End Suite Runtime:   89.66 seconds
================================================================================
```

---

## 4. The 4 Discrepancies: What Went Wrong & How to Fix Them

To maintain engineering transparency, the table below documents the **four synthetic scenarios that did not achieve perfect partition grouping** (TC_A01 was previously resolved to 100% precision by the generalized temporal parser), the root cause for each, and the engineering path to resolve them.

```
+----------------------------------------------------------------------------------------------------------------------------------------+
| Case ID & Challenge            | Expected vs. Predicted             | Root Cause Analysis                 | Technical Remediation      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| 1. TC_PERIOD_01_NO_FURTHER     | Expected: [101, 102, 103, 104]     | Note 103 stated: "No further falls  | Classification split:      |
|    (Negative_Period_Statements)| Predicted: [101, 102, 104]         | noted overnight." Note 104 is       | Distinguish notes that     |
|                                |                                    | grouped thanks to sentence-scoping, | report an incident from    |
|                                |                                    | but Note 103 contains only negative | monitoring notes that      |
|                                |                                    | monitoring with zero fall trigger.  | confirm zero new events.   |
|                                |                                    | Result: Incident count was exact.   |                            |
|                                |                                    |                                     |                            |
| 2. TC_NOISE_01_EMBEDDED_CARE   | Expected: [103, 104, 105]          | Note 105 was an evening dinner note | Cross-Encounter Entity     |
|    (High_Noise_Environment)    | Predicted: [103, 104]              | mentioning "Ice pack applied to     | Propagation: Care notes    |
|                                |                                    | bruised knee." It lacked a primary  | containing treatment for a |
|                                |                                    | fall term, so it was dropped as     | documented injury inherit  |
|                                |                                    | background noise.                   | the parent incident ID.    |
|                                |                                    |                                     |                            |
| 3. TC_E02_MISSING_ASSESSMENT   | Expected: [201, 202]               | Note 202 recorded general dinner    | Contextual Window Binding:  |
|    (E_Follow_Up_Evidence)      | Predicted: [201]                   | tray delivery after a 16:00 fall.   | All notes within 60 mins   |
|                                |                                    | Because it contained zero clinical  | of an incident should be   |
|                                |                                    | triggers, it was not grouped into   | attached to the cluster as |
|                                |                                    | the incident record.                | audit context.             |
|                                |                                    |                                     |                            |
| 4. TC_C03_RULED_OUT_FALL       | Expected: [] (0 incidents)         | Note 301 logged "sitting on floor"  | Multi-Document Scoping:    |
|    (C_Historical_And_Negated)  | Predicted: [301] (1 incident)      | in one entry; Note 302 ruled out    | When Note B explicitly     |
|                                |                                    | the fall in a subsequent note.      | rules out an incident,     |
|                                |                                    | Denial logic was note-scoped rather | retroactive resolution     |
|                                |                                    | than encounter-scoped.              | must cancel the preceding  |
|                                |                                    | Result: 1 false incident detection. | observation in Note A.     |
+----------------------------------------------------------------------------------------------------------------------------------------+
```

---

## 5. Inspection Demo Output (`TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS`)

When executing `python demo.py TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS`, the system outputs the following complete lifecycle trace:

```text
================================================================================
CHRONOMERGE: 60-SECOND EXECUTIVE INSPECTION DEMO
================================================================================
Case ID:        TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS
Challenge Type: B_Temporal_Proximity
Description:    Resident experiences two separate falls on the same day: one at 06:15 and a second at 18:30.
--------------------------------------------------------------------------------
RAW INPUT CLINICAL NURSING NOTES:
  • [2026-10-23T06:30:00] (Note NB101): "Resident found sitting on floor beside bed at 06:15. Stated slippers caught on rug."
  • [2026-10-23T07:00:00] (Note NB102): "Post-fall assessment completed for 0615 bedside fall. Resident uninjured."
  • [2026-10-23T18:45:00] (Note NB103): "Resident fell near dining room corridor at 18:30 during dinner transport. Assisted by dietary aide. New incident report opened."
  • [2026-10-23T19:15:00] (Note NB104): "Post-fall assessment and neuro check recorded for evening dining room incident. Ice pack applied to left wrist."
--------------------------------------------------------------------------------
EXPECTED GROUND TRUTH ANSWER:
  • Distinct Incidents: 2
    - Incident INC_B01_1: Notes ['NB101', 'NB102']
    - Incident INC_B01_2: Notes ['NB103', 'NB104']
  • Expected Evidence Audit Requirements:
    - Incident INC_B01_1 -> Assessment Found: True (Uncertainty: False)
    - Incident INC_B01_2 -> Assessment Found: True (Uncertainty: False)
--------------------------------------------------------------------------------
RUNNING CHRONOMERGE SYSTEM PIPELINE...

SYSTEM ANSWER (PREDICTED):
  • Identified Incidents: 2
    - Proposed Incident Cluster 0: Notes ['NB101', 'NB102']
      [LLM Compliance Audit]: Evidence Found: True | Uncertainty: False
      [Clinical Reasoning]:   The source notes contain an explicit statement that a post-fall assessment was completed, indicating that the required clinical assessment after the fall was documented. The quote directly references the assessment and its outcome, satisfying the compliance requirement.
      [Verbatim Quote]:       "Post-fall assessment completed for 0615 bedside fall. Resident uninjured."
      [Quote Verification]:   VERIFIED in Note NB102 (Offsets: 0..73, LITERAL EXACT)
    - Proposed Incident Cluster 1: Notes ['NB103', 'NB104']
      [LLM Compliance Audit]: Evidence Found: True | Uncertainty: False
      [Clinical Reasoning]:   The note NB104 explicitly states that a post-fall assessment and neuro check were recorded, which satisfies the requirement for a documented post-fall assessment.
      [Verbatim Quote]:       "Post-fall assessment and neuro check recorded for evening dining room incident."
      [Quote Verification]:   VERIFIED in Note NB104 (Offsets: 0..79, LITERAL EXACT)
--------------------------------------------------------------------------------
EVALUATION RESULT:
  • Incident Grouping: PASS — Predicted partitions match Ground Truth with 100% precision.
  • Evidence Audit:    PASS — All cluster evidence evaluations matched ground truth expectations.
================================================================================
```

> **Evaluation Analysis: Substring Verification & Decoupled Reporting**  
> - **Incident Grouping: PASS**: The deterministic reconciliation engine successfully separated the morning bedside fall (06:15) from the evening dining room fall (18:30) into two distinct incident clusters (`[NB101, NB102]` and `[NB103, NB104]`), matching the expected ground truth partitions with 100% precision.  
> - **Programmatic Quote Verification: PASS**: Every quote returned by the LLM was verified in Python as an authentic verbatim substring directly within the original unnormalized note text, with exact character offsets (`NB102: 0..73`, `NB104: 0..79`) indexed. Any fabricated quote is automatically rejected by the Python validator.  
> - **Separation of Metrics**: Rather than presenting an ambiguous aggregate score, the system reports grouping success and evidence-audit metrics independently so that structural timeline partitioning is never conflated with downstream clinical audit stringency.

---

## 6. Technical Architecture & Component Roles

```
[ Nursing Note Stream ] 
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Dual-Stage Hybrid Extractor [ Local CPU ]     │
│ - MedSpaCy Target Rule Matching (Fast-Path)            │
│ - Zero-Shot Concept Embeddings (all-MiniLM-L6-v2)      │
│ - Sentence-Level Ruling-Out & Predicate Negation       │
└─────────────────────────┬──────────────────────────────┘
                          │ (Active Mentions + Metadata)
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Temporal Reconciliation Engine [ Local CPU ]  │
│ - Fuzzy Clock Normalizer & Relative Day Delta Resolver │
│ - Exponential Shift Decay Kernel (tau = 8 hours)       │
│ - Strict Cross-Resident Distance Isolation (d = inf)   │
│ - Average-Linkage Agglomerative Clustering             │
└─────────────────────────┬──────────────────────────────┘
                          │ (Reconciled Incident Clusters)
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Structured LLM Auditor [ Groq Cloud API ]     │
│ - Remote Call to openai/gpt-oss-120b via Groq          │
│ - Zero-Shot Pydantic Structured Schema Validation      │
│ - Verbatim Substring Proof Extraction                  │
│ - Clinical Uncertainty Flag Calibration                │
└────────────────────────────────────────────────────────┘
```

### Execution Boundary & Hardware Profile
- **Local CPU Execution (Stages 1 & 2 - Incident Matching)**: The entire extraction, temporal normalization, and agglomerative clustering pipeline runs **100% locally on CPU**. It has zero external API dependencies, produces deterministic incident partitions, and completes in ~4.41 seconds active elapsed wall-clock time (~9.51s total including 5.10s model initialization into RAM) on standard consumer hardware. Active runtime is measured as elapsed wall-clock time via `time.perf_counter()`, not CPU execution time.
- **External Cloud API (Stage 3 - Evidence Auditing)**: Downstream compliance verification queries the **Groq Cloud API** hosting `openai/gpt-oss-120b` for Pydantic-constrained structured extraction. It requires an active internet connection and a valid `GROQ_API_KEY`.
- **Pipeline Classification**: ChronoMerge is a **hybrid system**—local for incident reconciliation, cloud-assisted for deep semantic compliance auditing. It must not be characterized as an offline-only product.

