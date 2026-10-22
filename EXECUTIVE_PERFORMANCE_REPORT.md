# ChronoMerge: Clinical Incident Reconciliation & Evidence Engine
## Executive Briefing & Performance Validation Report

---

## 1. The Real-World Problem (In Plain English)

### What happens in a healthcare facility?
In a nursing home or long-term care facility, elderly residents require continuous care. When an incident occurs—such as a resident falling—it is rarely documented in a single, neat record.

Instead, **one real event creates a flurry of fragmented notes written by different people across different shifts**:
1. **2:15 AM (Nurse Aide)**: *"Resident found on floor beside bed. Notified RN."*
2. **2:40 AM (Night Nurse)**: *"Post-fall check done. Denies pain. Returned to bed."*
3. **4:30 AM (Night Nurse)**: *"Neuro check stable. Vitals normal."*
4. **7:15 AM (Day Shift Handover)**: *"Day shift briefed on overnight fall near bed."*
5. **11:00 AM (Visiting Doctor)**: *"Reviewed resident following early morning fall. Joints uninjured."*

Notice what just happened: **Five separate notes were written across 9 hours by 4 different people, but only ONE fall actually happened.**

---

### Where existing software fails
Standard healthcare analytics software and keyword search tools make two costly mistakes:

```
Mistake A: Overcounting (Duplicate Error)
The computer sees the word "fall" 5 times and reports:
"This resident had 5 falls today!"
Consequence: Incident numbers are inflated, state compliance reports are inaccurate, and clinical staff waste hours investigating "duplicate phantom falls."

Mistake B: Erasing Real Incidents (False-Merge Error)
A resident falls in the bedroom at 6:00 AM, and falls AGAIN near the dining hall at 6:30 PM. 
A naive AI groups them together because they involve the same resident and similar words.
Consequence: A real fall is silently erased from the record, and the required safety follow-up is overlooked.
```

---

## 2. What ChronoMerge Does

**ChronoMerge** is an intelligent clinical record reconciliation assistant. It reads messy, unstructured nursing notes and performs two core jobs:

```
+-----------------------------------------------------------------------------------+
|                               CHRONOMERGE WORKFLOW                                |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Messy Nursing Notes ]                                                          |
|           |                                                                       |
|           v                                                                       |
|  PART 1: INCIDENT RECONCILIATION                                                  |
|  - Groups notes describing the SAME incident into one timeline.                   |
|  - Keeps SEPARATE incidents strictly apart.                                       |
|  - Ignores historical risk ("has a history of falls") & denials ("denies falling").|
|           |                                                                       |
|           v                                                                       |
|  PART 2: CLINICAL EVIDENCE VALIDATION (Audit Check)                               |
|  - Checks whether required follow-up care (e.g., physical assessment) was done.   |
|  - Distinguishes "Comprehensive Exam Done" vs "Vague Check" vs "Missing".         |
|  - Pulls the exact verbatim quote from the note as proof.                         |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 3. What Was in the Test Dataset? (No JSON, Just Real Scenarios)

To evaluate the approach under controlled conditions, I built a synthetic benchmark of 20 resident scenarios covering 69 clinical notes.

Here are the real clinical challenges I tested:

| Test Challenge | What the Scenario Looks Like in Real Life | What ChronoMerge Had to Prove |
| :--- | :--- | :--- |
| **1. Multi-Staff Progression** | An incident generates 5 notes over 9 hours across aides, nurses, and doctors. | Merge all 5 notes into **exactly 1 incident**. |
| **2. Morning vs. Evening Falls** | A resident falls at 6:15 AM by the bed, then falls again at 6:30 PM in the dining room. | Keep them as **2 distinct incidents** (do not erase either one). |
| **3. Back-to-Back Falls** | A resident slips from a wheelchair at 1:00 PM, and falls again 4.5 hours later at 5:45 PM. | Recognize phrases like *"Second fall today"* and keep them separate. |
| **4. Care Plans & Denials** | Notes say *"High fall risk on Morse scale"* and *"Resident denies any falls overnight"*. | Report **zero incidents** (do not count prevention or denials as falls). |
| **5. Ruled-Out Falls** | Note says resident sat on the floor, but nurse writes *"Fall ruled out after evaluation"*. | Suppress the observation because clinical evaluation ruled it out. |
| **6. Disputed Facts** | Aide says *"unwitnessed, hit head"*; Nurse later says *"witnessed, denies hitting head"*. | Group into 1 incident while flagging the disagreement for review. |
| **7. Cross-Day Reviews** | A fall happens Monday; the care team reviews *"the fall earlier this week"* on Wednesday. | Link Wednesday's note back to Monday without inventing a fake Wednesday fall. |
| **8. Roommate Mix-ups** | Alice (Bed A) falls at 3:00 AM; roommate Betty (Bed B) wakes up at 3:15 AM to use the bathroom. | **Never mix up roommates** sharing the same room. |
| **9. Hidden in Routine Notes** | A fall happens in the middle of routine logs (meal trays, pills, physical therapy). | Ignore the daily noise and catch only the real incident and assessment. |

---

## 4. How Did the System Perform? (The Numbers)

I ran the automated engine across all 20 clinical scenarios and 69 notes. Here is the verified score sheet:

### Executive Scorecard

```
+--------------------------------------------------------------------------------------------+
| METRIC                                       | RESULT                   | SUCCESS RATE     |
+--------------------------------------------------------------------------------------------+
| Incident Count Accuracy                      | 19 / 20 cases            | 95.0%            |
| (Did it report the exact true number of falls on this synthetic benchmark?)                |
|                                                                                            |
| Perfect Note-to-Incident Grouping            | 16 / 20 cases            | 80.0%            |
| (Did it attach 100% of related notes correctly without fragmenting?)                       |
|                                                                                            |
| Detector Audit Coverage Rate                 | 11 / 11 audit targets    | 100.0%           |
| (Did detector provide matched clusters for all gold-standard audit targets?)               |
|                                                                                            |
| API Audit Completion Rate                    | 11 / 11 audit targets    | 100.0%           |
| (Did all dispatched LLM audit API requests complete successfully?)                         |
|                                                                                            |
| Follow-Up Evidence Finding (Audit Check)     | 11 / 11 completed audits | 100.0%           |
| (Did it correctly identify whether follow-up care occurred on completed audits?)           |
|                                                                                            |
| Uncertainty Calibration                      | 10 / 11 completed audits | 90.9%            |
| (Did it know when notes were too vague to trust on completed audits?)                      |
|                                                                                            |
| Programmatic Substring Quote Verification    | 8 / 8 quotes             | 100.0%           |
| (Were extracted proof quotes authenticated in raw source notes on positive clusters?)      |
| (All 8 quotes were 100% character-for-character literal matches in source notes)           |
|                                                                                            |
| End-to-End Finding Accuracy                  | 11 / 11 audit targets    | 100.0%           |
| (Overall evidence accuracy across all 11 audited gold-standard target incidents)           |
+--------------------------------------------------------------------------------------------+
```

---

### What would a simple system have done on the exact same data?

| System Type | Total Incidents Reported on This Dataset | Reality (True Incidents) | Error Rate |
| :--- | :---: | :---: | :---: |
| **Naive Note Counter** (Counts every note mentioning "fall") | **61 Falls** | **24 Real Falls** | **+154.2% Overcount** (Completely inflated) |
| **Simple 4-Hour Keyword Window** | **33 Falls** | **24 Real Falls** | **+37.5% Overcount** |
| **ChronoMerge Engine** | **25 Falls** | **24 Real Falls** | **+4.2% Overcount** (95.0% Count Accuracy) |

*(Note: Evaluated strictly against synthetic benchmark scenarios under controlled conditions; no real-world clinical validation or controlled human-auditor trial has been performed.)*

**Key Business Takeaway**: ChronoMerge reduced false duplicate incident overcounts from +154% down to +4.2%, with **zero false merges observed on this synthetic benchmark**.

---

## 5. Live Examples From the Audit

To see how this works in practice, here are two actual outputs generated by the engine:

### Example A: Catching Multi-Author Duplicates
* **Input Notes**:
  - `02:15 AM`: *"Resident found sitting on floor beside bed at approx 0200. Staff notified RN."*
  - `02:40 AM`: *"Post-fall assessment completed. Resident denies pain. No visible injury noted."*
  - `07:10 AM`: *"Resident reviewed this morning following overnight fall. Continue monitoring."*
* **ChronoMerge Output**:
  - **Incidents Identified**: **Exactly 1 Incident** (All 3 notes grouped together).
  - **Estimated Event Time**: `02:00 AM` (Extracted from note text, not logging time).
  - **Evidence Audit**: **Evidence Found: YES** (`Uncertainty: NO`).
  - **Verbatim Proof**: *"Post-fall assessment completed. Resident denies pain. No visible injury noted."*
  - **Programmatic Quote Verification**: **VERIFIED in Note N102 (Offsets: 0..78, LITERAL EXACT)**.

---

### Example B: Separating Real Consecutive Falls
* **Input Notes**:
  - `06:30 AM`: *"Resident found sitting on floor beside bed at 06:15. Slipped on rug."*
  - `07:00 AM`: *"Post-fall assessment completed for 0615 bedside fall."*
  - `06:45 PM`: *"Resident fell near dining room corridor at 18:30. New incident report opened."*
  - `07:15 PM`: *"Post-fall assessment and neuro check recorded for evening dining incident."*
* **ChronoMerge Output**:
  - **Incidents Identified**: **Exactly 2 Separate Incidents**.
    - `Incident 1`: Morning Bedside Fall (Notes 1 & 2).
    - `Incident 2`: Evening Dining Room Fall (Notes 3 & 4).
  - **False Merge Avoided**: Kept the two falls distinct despite identical resident and vocabulary.

---

## 6. How the Pipeline Works (Core Components)

ChronoMerge combines deterministic clinical NLP with structured LLM verification:
1. **Semantic Understanding**: Uses a local AI embedding model (`all-MiniLM-L6-v2`) to recognize clinical concepts (*"slumped off commode"*, *"landed on knees"*, *"tumbled in hallway"*) even if nurses use different words, with sentence-level contextual negation screening.
2. **Clinical Shift Awareness**: Uses an exponential decay function that mirrors hospital nursing shift routines (tau = 8 hour half-life).
3. **Medical Logic & Scoping**: Understands that when a note states *"Fall ruled out"*, prior observations in that sentence context should not create an incident.
4. **Independent Evidence Auditing with Substring Verification**: Uses a large language model (`openai/gpt-oss-120b` via Groq) strictly as an auditor to check clinical documentation, coupled with programmatic Python substring verification that validates and indexes the exact character offsets in the raw source text.
5. **Fast & Modular**: Pipeline and client initialization takes ~10.16 seconds (~5.10s model loading into RAM + ~5.06s client build; clean offline installation requires model weights to be cached beforehand). Active local incident reconciliation runs 100% locally on standard computer hardware in ~4.41 seconds elapsed wall-clock time (measured via `time.perf_counter()`) across all 20 cases (69 notes) with zero API calls. Downstream compliance auditing queries the external LLM API in ~74.93 seconds across all 11 audited gold incidents (~89.66 seconds total suite runtime).

---

## 7. Bottom Line Summary

> **"ChronoMerge is a promising research prototype demonstrating how deterministic clinical NLP and structured LLM verification can reconcile fragmented healthcare documentation. Under controlled synthetic evaluation across 20 resident scenarios (69 notes), it achieved 95% incident-count accuracy (19/20 cases) and 80% perfect grouping accuracy (16/20 cases; 4 edge cases remain imperfect) with zero observed false merges on this benchmark, while programmatically authenticating verbatim audit proof in the source notes. It demonstrates technical feasibility as a prototype, while real-world validation in live clinical facilities remains future work."**


