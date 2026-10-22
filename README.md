# ChronoMerge: Clinical Incident Reconciliation & Compliance Evidence Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-MedSpaCy%201.3-green.svg)](https://github.com/medspacy/medspacy)
[![LLM Audit](https://img.shields.io/badge/LLM%20Audit-Groq%20%2F%20GPT--OSS--120B-orange.svg)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-7%20Passed%20(100%25)-brightgreen.svg)](test_clinical_logic.py)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-lightgrey.svg)](EXECUTIVE_PERFORMANCE_REPORT.md)

> **ChronoMerge is an intelligent clinical record reconciliation assistant.** It untangles fragmented, messy nursing notes across hospital shifts to determine the true number of real-world patient incidents (such as falls), eliminates duplicate paperwork, avoids dangerous false merges, and programmatically audits post-incident clinical assessments with 100% verbatim proof in the source notes.

---

## 1. The Real-World Problem (In Plain English)

### What Happens in a Healthcare Facility?
In elderly care facilities, nursing homes, and hospitals, elderly residents require continuous monitoring. When an incident occurs—such as a resident falling—**it is almost never recorded in a single, neat document**. 

Instead, a single real event generates a flurry of notes written by different staff members across multiple nursing shifts:
1. **02:15 AM (Nurse Aide)**: *"Resident found on floor beside bed at approx 0200. Staff notified RN."*
2. **02:40 AM (Night Nurse)**: *"Post-fall assessment completed. Resident denies pain. Returned to bed."*
3. **04:30 AM (Night Nurse)**: *"Neuro check stable. Vitals normal."*
4. **07:15 AM (Day Shift Handover)**: *"Day shift briefed on overnight bedside fall near bed. Monitoring."*
5. **11:00 AM (Attending Doctor)**: *"Reviewed resident following early morning fall. Joints uninjured."*

> **Notice what happened:** Five separate notes were written across 9 hours by 4 different healthcare professionals, but **only ONE real fall occurred**.

---

### Where Existing Software Fails

Standard healthcare software and naive AI search tools make two costly, dangerous mistakes:

```
❌ Mistake A: The Duplicate Epidemic (Overcounting)
The system searches for the keyword "fall", sees it 5 times, and reports:
"This resident had 5 falls today!"
Consequence: Incident numbers are wildly inflated (+154% duplicate error), 
state regulatory compliance reports are inaccurate, and clinical supervisors 
waste dozens of hours investigating phantom duplicates.

❌ Mistake B: The False Merge Disaster (Erasing Real Incidents)
A resident slips in their bedroom at 6:15 AM, and falls AGAIN near the dining hall at 6:30 PM.
A generic AI tool groups them into one incident because they involve the same resident 
and similar vocabulary.
Consequence: A real medical emergency is silently erased from the record, leaving the 
patient without mandatory neuro checks or fall-prevention interventions.
```

---

## 2. What Makes ChronoMerge Special?

Most modern AI projects attempt to solve this problem by taking all messy notes and feeding them into an expensive, slow commercial chatbot prompt (e.g. asking a raw LLM: *"How many falls happened?"*). 

**That approach fails in clinical production because:**
- Generic LLMs hallucinate non-existent quotes and miscount dates.
- LLM API calls are slow and cost hundreds of dollars per day per facility.
- Hospital privacy and air-gapped infrastructure require incident detection to run **locally and deterministically** without sending sensitive data to external clouds.

### The ChronoMerge Innovation: Two-Stage Hybrid Architecture

ChronoMerge divides the problem into what **local deterministic algorithms** do best, and what **large language models** do best:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               CHRONOMERGE PIPELINE                               │
└──────────────────────────────────────────────────────────────────────────────────┘

   [ Messy, Unstructured Multi-Author Clinical Nursing Notes ]
                             │
                             ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ STAGE 1 & 2: DETERMINISTIC CLINICAL NLP (100% Local CPU, Zero API Cost)│
   ├────────────────────────────────────────────────────────────────────────┤
   │ 1. Clinical Concept Extraction (MedSpaCy):                             │
   │    Extracts mentions, observations, and care assessments.              │
   │ 2. Contextual Sentence Scoping:                                        │
   │    Understands clinical rule-outs ("Fall ruled out after RN eval") and │
   │    patient denials ("denies dizziness") without erasing real falls.    │
   │ 3. Clinical Temporal Parser:                                           │
   │    Normalizes shift references ("0200", "at 15:45", "yesterday")       │
   │    without confusing calendar years ("in 2026") for clock times.       │
   │ 4. Nursing Shift Decay (τ = 8 hours):                                  │
   │    Calculates temporal distance based on real hospital shift routines. │
   │ 5. Agglomerative Clustering with Hard Cross-Resident Barriers:         │
   │    Partitions notes into distinct real-world incidents deterministically│
   │    in ~4.41s across 69 notes. Zero false merges.                       │
   └────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼ (Reconciled Incident Clusters)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ STAGE 3: STRUCTURED COMPLIANCE AUDITING (Groq Cloud LLM + Python Guard) │
   ├────────────────────────────────────────────────────────────────────────┤
   │ • Pydantic-Constrained Structured Audit (openai/gpt-oss-120b):          │
   │   Checks: Was a mandatory post-fall physical exam completed?           │
   │ • Programmatic Substring Verification (Anti-Hallucination Guardrail):  │
   │   Python scans the raw note text character-for-character to verify     │
   │   that the quote literally exists in the note.                         │
   │ • Exact Character Offset Attribution (e.g., Note N102, Char 0..73).     │
   │ • Automatic Overrule: If the LLM invents a quote, Python intercepts it,│
   │   quarantines the quote, and forces uncertainty.                       │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Verified Benchmark Results

Under controlled testing across a synthetic benchmark of **20 resident scenarios (69 clinical notes)** representing complex real-world challenges (multi-staff handover, same-day repeat falls, back-to-back incidents, clinical rule-outs, roommate mix-ups, and routine chart noise):

### Algorithmic Baseline Comparison

| Method | Reported Falls | True Falls | Overcount Error | Merge Failures |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Note Counter** (Counts every "fall" keyword) | 61 | 24 | **+154.2%** (Severe inflation) | N/A |
| **Simple 4-Hour Time Window** | 33 | 24 | **+37.5%** | Multiple |
| **ChronoMerge Engine** | **25** | **24** | **+4.2%** (95% Count Accuracy) | **0 False Merges** |

### Verified Performance Scorecard

| Metric | Result | Success Rate | Evaluation Scope |
| :--- | :---: | :---: | :--- |
| **Incident Count Accuracy** | **19 / 20** | **95.0%** | Resident Scenarios (20 cases / 69 notes) |
| **Perfect Note Grouping** | **16 / 20** | **80.0%** | Exact cluster partition match (4 documented edge cases) |
| **False-Merge Prevention** | **100%** | **100.0%** | **Zero** distinct incidents erroneously collapsed |
| **Detector Audit Coverage** | **11 / 11** | **100.0%** | All gold-standard audit targets detected |
| **Evidence Finding Accuracy** | **11 / 11** | **100.0%** | Identified mandatory clinical exams on completed audits |
| **Uncertainty Calibration** | **10 / 11** | **90.9%** | Flagged vague documentation appropriately |
| **Verbatim Quote Verification** | **8 / 8** | **100.0%** | **100%** character-for-character literal match in source text |
| **Active Local CPU Runtime** | **4.41s** | — | Elapsed wall-clock time across all 20 scenarios (69 notes) |

*(Measurements verified via live benchmark execution; active runtime measured via `time.perf_counter()` on CPU).*

---

## 4. 60-Second Quickstart (Try It in One Command)

You can run an executive inspection of any benchmark scenario in seconds:

```powershell
python demo.py TC_B01_SAME_DAY_MORNING_AND_EVENING_FALLS
```

### What the Demo Prints:
1. **Raw Nursing Notes**: The messy, unstructured notes logged across the morning and evening shifts.
2. **Ground Truth Expectation**: The true clinical reality (2 separate falls, 1 at 06:15 and 1 at 18:30).
3. **ChronoMerge System Output**:
   - Accurately splits the notes into 2 distinct clusters without false merges.
   - Audits whether a post-fall assessment was documented for each.
   - Extracts verbatim proof quotes and prints their **exact character offsets** in the raw source notes (e.g., `Offsets: 0..73 in Note NB102`).
   - Flags whether the evaluation passes with 100% precision.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.11+
- Windows, macOS, or Linux
- Standard consumer CPU (no GPU required for local reconciliation)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/ChronoMerge.git
cd ChronoMerge

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# On Windows:
copy .env.example .env
# On Linux/macOS:
cp .env.example .env
```

If you plan to run **Stage 3 LLM Evidence Auditing**, add your free Groq API key to `.env`:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```
*(Note: If you only want to run Stage 1 & 2 local incident reconciliation, no API key is required).*

---

## 6. How to Run the System

### Option A: 100% Offline Local Benchmark (No API Key Needed)
Runs the deterministic extraction, temporal parsing, and agglomerative clustering across all 20 scenarios (69 notes) on CPU in **~4.41 seconds**:
```powershell
python run_benchmark.py --mode offline
```

### Option B: Full Pipeline Suite (Includes Groq LLM Evidence Auditing)
Runs the full end-to-end benchmark including local reconciliation, remote structured compliance audits via `openai/gpt-oss-120b`, programmatic substring quote verification, and full accuracy scorecards:
```powershell
python run_benchmark.py --mode full
```

### Option C: Standalone Pipeline on Custom Data
Run the end-to-end reconciliation and auditing pipeline directly on any formatted JSON dataset:
```powershell
python run_pipeline.py
```

### Option D: Run the Clinical Unit Test Suite
Verify all core clinical logic, temporal parsing, and anti-hallucination guardrails:
```powershell
pytest -v test_clinical_logic.py
```

---

## 7. Repository Structure & File Guide

```
ChronoMerge/
├── nlp_extractor.py               # MedSpaCy clinical concept extraction & sentence negation scoping
├── reconciliation_engine.py       # Clinical temporal parser, shift decay matrix & agglomerative clustering
├── evidence_validator.py          # Instructor/Pydantic LLM auditor & programmatic quote verifier
├── run_benchmark.py               # Comprehensive benchmark suite (offline & full LLM evaluation modes)
├── demo.py                        # Single-case 60-second executive inspection demo
├── run_pipeline.py                # Standalone end-to-end pipeline runner
├── test_clinical_logic.py         # Pytest regression suite (7 comprehensive test cases)
│
├── data/
│   └── master_exhaustive_benchmark.json   # Canonical master dataset: 20 resident scenarios (69 clinical notes)
│
├── BENCHMARK_REPORT.md            # Detailed technical performance evaluation & hardware profiling
├── EXECUTIVE_PERFORMANCE_REPORT.md# Plain-English executive briefing & business impact report
├── requirements.txt               # Locked Python dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git exclusion rules for secrets, caches, and logs
└── README.md                      # This comprehensive project guide
```

---

## 8. What Files to Upload to GitHub

When publishing this repository to GitHub or sharing it with collaborators, use the following clean checklist:

### ✅ Essential Files to Commit and Upload:
- **Core Engine Code**: `nlp_extractor.py`, `reconciliation_engine.py`, `evidence_validator.py`, `run_pipeline.py`
- **Executables & Demonstrations**: `demo.py`, `run_benchmark.py`
- **Automated Tests**: `test_clinical_logic.py`
- **Canonical Benchmark Dataset**: `data/master_exhaustive_benchmark.json` (the only data file required for all benchmarks, demos, and reports)
- **Reports & Documentation**: `README.md`, `BENCHMARK_REPORT.md`, `EXECUTIVE_PERFORMANCE_REPORT.md`
- **Configuration Templates**: `requirements.txt`, `.gitignore`, `.env.example`

### 💡 Optional / Redundant Development Files (Can be Safely Removed):
- `data/benchmark_cases.json`, `data/complete_blueprint_benchmark.json`, `data/heavy_benchmark_cases.json` *(already fully merged into `master_exhaustive_benchmark.json`)*
- `build_master_dataset.py`, `generate_complete_benchmark.py`, `generate_heavy_benchmark.py` *(used during dataset synthesis)*
- `evaluate_master_benchmark.py`, `evaluate_heavy_benchmark.py`, `verify_benchmark.py` *(superseded by `run_benchmark.py`)*

### ❌ Files to EXCLUDE (Never Commit to Git):
- `.env` *(contains your private `GROQ_API_KEY`)*
- `__pycache__/` and `*.pyc` *(compiled Python bytecode)*
- `.pytest_cache/` *(local pytest runtime metadata)*
- `venv/` or `.venv/` *(local virtual environment packages)*
- Any local log files (`*.log`)

*(All exclusions are pre-configured in [.gitignore](.gitignore)).*

---

## 9. Trust Boundaries & Clinical Research Disclaimer

- **Prototype Status**: ChronoMerge is an advanced research and engineering prototype. It demonstrates the technical feasibility of hybrid deterministic-probabilistic clinical reconciliation.
- **Controlled Synthetic Benchmark**: All metrics reported herein were measured under controlled conditions against author-created synthetic nursing scenarios covering 69 notes.
- **Future Work**: No real-world clinical validation or prospective human-in-the-loop trial has yet been conducted in live Electronic Health Record (EHR) environments. Real-world validation remains active future work.
- **Zero Hallucination Tolerance**: In production, LLM-generated compliance conclusions should always be audited through programmatic substring offset verification before inclusion in permanent medical records.

---

## 10. License & Acknowledgments

Built with:
- [MedSpaCy](https://github.com/medspacy/medspacy) for clinical text processing
- [scikit-learn](https://scikit-learn.org/) for hierarchical agglomerative clustering
- [Sentence-Transformers](https://sbert.net/) for semantic embedding representations
- [Groq](https://groq.com/) & [Instructor](https://github.com/jxnl/instructor) for high-speed Pydantic structured output validation
