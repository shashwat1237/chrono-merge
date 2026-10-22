import re
import medspacy
from medspacy.target_matcher import TargetRule
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# Cache embedding model to avoid repeated loads
_EMBED_MODEL = None

def get_embed_model(require_model: bool = False):
    """
    Loads and caches the SentenceTransformer model for zero-shot clinical semantic fallback.
    
    If require_model=True and loading fails, raises RuntimeError with an actionable offline caching message.
    If require_model=False and loading fails, sets _EMBED_MODEL=None and returns None.
    
    NOTE FOR CLEAN / OFFLINE INSTALLATIONS:
    The local pipeline requires pretrained weights for 'all-MiniLM-L6-v2' to be cached
    or pre-downloaded in the HuggingFace cache directory before operating in an offline
    or airgapped environment without active network access.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            _EMBED_MODEL = None
            if require_model:
                raise RuntimeError(
                    f"Failed to initialize SentenceTransformer model 'all-MiniLM-L6-v2': {e}.\n"
                    f"NOTE: A clean installation requires pretrained model artifacts to be cached or "
                    f"downloaded before the local pipeline can operate in offline mode."
                ) from e
    return _EMBED_MODEL

@dataclass
class ExtractedMention:
    note_id: str
    concept_category: str
    span_text: str
    is_negated: bool
    is_historical: bool
    is_hypothetical: bool
    raw_text: str
    timestamp: str
    resident_id: str = "DEFAULT_RESIDENT"
    confidence: float = 1.0
    sentence_text: str = ""

# Pre-computed semantic anchor descriptions for zero-shot clinical generalization
ANCHOR_DESCRIPTIONS = {
    "INCIDENT_FALL": [
        "patient had a fall", "slipped and fell to floor", "loss of balance and dropped",
        "tripped over object", "discovered sitting or lying on ground", "slumped to floor",
        "lost balance and went down", "unscripted loss of footing"
    ],
    "EVIDENCE_ASSESSMENT": [
        "comprehensive post fall evaluation", "neurological exam vitals checked",
        "assessed by registered nurse", "range of motion checked"
    ],
    "REFERENTIAL_AMENDMENT": [
        "correction to prior documentation", "late entry addendum",
        "clarification regarding earlier incident"
    ],
    "PREVENTIVE_PLAN": [
        "fall risk precautions", "preventative care plan", "history of frequent falls"
    ]
}

def build_medspacy_pipeline():
    # Base clinical pipeline with PyRuSH sentence segmenter and ConText
    nlp = medspacy.load()
    target_matcher = nlp.get_pipe("medspacy_target_matcher")

    # High-precision core clinical targets (Fast Path)
    rules = [
        TargetRule(literal="fall precautions", category="PREVENTIVE_PLAN"),
        TargetRule(literal="fall risk", category="PREVENTIVE_PLAN"),
        TargetRule(literal="history of falls", category="PREVENTIVE_PLAN"),
        TargetRule(literal="fall", category="INCIDENT_FALL"),
        TargetRule(literal="falls", category="INCIDENT_FALL"),
        TargetRule(literal="fell", category="INCIDENT_FALL"),
        TargetRule(literal="falling", category="INCIDENT_FALL"),
        TargetRule(literal="slipped", category="INCIDENT_FALL"),
        TargetRule(literal="tripped", category="INCIDENT_FALL"),
        TargetRule(literal="stumbled", category="INCIDENT_FALL"),
        TargetRule(literal="tumbled", category="INCIDENT_FALL"),
        TargetRule(literal="found on floor", category="INCIDENT_OBSERVATION"),
        TargetRule(literal="sitting on floor", category="INCIDENT_OBSERVATION"),
        TargetRule(literal="lying on floor", category="INCIDENT_OBSERVATION"),
        TargetRule(literal="slumped to floor", category="INCIDENT_OBSERVATION"),
        TargetRule(literal="dropped to floor", category="INCIDENT_OBSERVATION"),
        TargetRule(literal="altercation", category="INCIDENT_BEHAVIORAL"),
        TargetRule(literal="combative", category="INCIDENT_BEHAVIORAL"),
        TargetRule(literal="post-fall assessment", category="EVIDENCE_ASSESSMENT"),
        TargetRule(literal="post-fall", category="EVIDENCE_ASSESSMENT"),
        TargetRule(literal="neurological check", category="EVIDENCE_ASSESSMENT"),
        TargetRule(literal="neuro check", category="EVIDENCE_ASSESSMENT"),
        TargetRule(literal="incident report", category="EVIDENCE_REPORT"),
        TargetRule(literal="Correction:", category="REFERENTIAL_AMENDMENT"),
        TargetRule(literal="Late entry", category="REFERENTIAL_AMENDMENT"),
        TargetRule(literal="Addendum to prior note", category="REFERENTIAL_AMENDMENT"),
    ]
    target_matcher.add(rules)
    return nlp

def extract_clinical_mentions(note: dict, nlp) -> List[ExtractedMention]:
    """
    Extracts clinical mentions using a dual-stage architecture:
    1. Fast-path MedSpaCy rules for high-precision extraction.
    2. Zero-shot clinical sentence embedding matcher to capture unscripted nursing variations.
    """
    doc = nlp(note["text"])
    mentions = []
    res_id = note.get("resident_id", "DEFAULT_RESIDENT")

    # Stage 1: Fast-path target matching
    for ent in doc.ents:
        sent_str = ent.sent.text.strip() if hasattr(ent, 'sent') and ent.sent is not None else ""
        mentions.append(
            ExtractedMention(
                note_id=note["note_id"],
                concept_category=ent.label_,
                span_text=ent.text,
                is_negated=ent._.is_negated,
                is_historical=ent._.is_historical,
                is_hypothetical=ent._.is_hypothetical,
                raw_text=note["text"],
                timestamp=note["timestamp"],
                resident_id=res_id,
                confidence=1.0,
                sentence_text=sent_str
            )
        )

    # Stage 2: Zero-shot semantic fallback if no active incident/evidence was captured
    has_primary = any(m.concept_category in ("INCIDENT_FALL", "INCIDENT_OBSERVATION", "EVIDENCE_ASSESSMENT") for m in mentions)
    if not has_primary and len(note["text"].strip()) > 10:
        model = get_embed_model()
        if model is not None:
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 8]
            if sentences:
                sent_embs = model.encode(sentences, normalize_embeddings=True)
                # Compute similarity with core concept anchors
                for sent_idx, sent_text in enumerate(sentences):
                    emb = sent_embs[sent_idx]
                    for cat, anchor_texts in ANCHOR_DESCRIPTIONS.items():
                        anchor_embs = model.encode(anchor_texts, normalize_embeddings=True)
                        sim = float(np.max(np.dot(anchor_embs, emb)))
                        if sim >= 0.50:  # Calibrated semantic alignment threshold (filters routine care <=0.30)
                            s_lower = sent_text.lower()
                            # Targeted incident negation: Negates only when the incident predicate is negated,
                            # NOT incidental negative findings (e.g., 'no dizziness', 'without pain', 'no injury')
                            is_s_neg = bool(re.search(
                                r'\b(?:no|not|denies|denied|without|ruled out|negative for|neither)\s+(?:documented\s+|reported\s+|apparent\s+|witnessed\s+|further\s+|any\s+)?(?:falls?|falling|slips?|trips?|tumbles?|unwitnessed drop|incident)\b'
                                r'|\b(?:fall|incident|slip)\s+(?:was\s+)?(?:ruled out|not found|disconfirmed|denied)\b',
                                s_lower
                            ))
                            is_s_hist = bool(re.search(r'\b(history of|past|previous|prior to admission|baseline|earlier year|chronic)\b', s_lower))
                            is_s_hypo = bool(re.search(r'\b(risk|precaution|potential|if|plan|recommend|consider)\b', s_lower))

                            mentions.append(
                                ExtractedMention(
                                    note_id=note["note_id"],
                                    concept_category=cat,
                                    span_text=sent_text[:40],
                                    is_negated=is_s_neg,
                                    is_historical=is_s_hist,
                                    is_hypothetical=is_s_hypo,
                                    raw_text=note["text"],
                                    timestamp=note["timestamp"],
                                    resident_id=res_id,
                                    confidence=round(sim, 3),
                                    sentence_text=sent_text
                                )
                            )
                            break

    return mentions

def filter_active_incidents(mentions: List[ExtractedMention]) -> List[ExtractedMention]:
    """
    Filters clinical mentions using sentence-scoped probabilistic and discourse logic:
    - Suppresses negated, hypothetical, historical, and preventative mentions.
    - Scopes clinical rule-outs and resident denials specifically to their sentence context,
      preventing a rule-out in one sentence from suppressing genuine incidents in another.
    - Preserves prior incident mentions even when monitoring notes state 'no further falls'.
    """
    if not mentions:
        return []

    filtered = []
    for m in mentions:
        if m.concept_category == "PREVENTIVE_PLAN":
            continue
        if m.is_negated or m.is_hypothetical or m.is_historical:
            continue

        # Scope rule-out and denial checks to the mention's sentence or immediate local context
        context_text = (m.sentence_text or m.span_text).lower()
        is_mention_ruled_out = bool(re.search(
            r'\b(fall ruled out|ruled out after assessment|ruled out after evaluation|determined not a fall|false alarm)\b',
            context_text
        ))
        is_mention_denial = bool(re.search(
            r'\b(denies (any )?(?:falls?|incidents?)|denies having fallen|states (she|he|resident) did not fall|no fall occurred|no incident occurred)\b',
            context_text
        ))

        # Discard incident observations if the specific mention is clinically ruled out or denied
        if (is_mention_ruled_out or is_mention_denial) and m.concept_category in ("INCIDENT_FALL", "INCIDENT_OBSERVATION"):
            continue
        filtered.append(m)

    return filtered

