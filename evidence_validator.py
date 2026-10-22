import os
import re
from dotenv import load_dotenv
import instructor
from groq import Groq
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Tuple

# Load environment variables from .env
load_dotenv()

class PostFallAssessmentValidation(BaseModel):
    evidence_found: bool = Field(..., description="True if a comprehensive post-incident clinical assessment is explicitly documented.")
    exact_quote: Optional[str] = Field(None, description="The authenticated verbatim substring from the source text serving as proof. Only populated when verified.")
    unverified_quote: Optional[str] = Field(None, description="Candidate quote string from LLM that could not be verified in source notes.")
    uncertainty_flag: bool = Field(..., description="True if the text references an assessment vaguely but lacks sufficient detail.")
    reasoning: str = Field(..., description="Step-by-step logical explanation of the finding.")
    quote_verified_in_source: bool = Field(False, description="Programmatically verified by Python as an authentic span in source text.")
    is_literal_match: bool = Field(False, description="True if LLM quote was literally identical to source span without case/whitespace normalization.")
    source_note_id: Optional[str] = Field(None, description="Note ID where the quote was located.")
    char_start: Optional[int] = Field(None, description="Start character offset of the quote within the source note.")
    char_end: Optional[int] = Field(None, description="End character offset of the quote within the source note.")

    @model_validator(mode='after')
    def check_quote_matches_text(self) -> 'PostFallAssessmentValidation':
        if self.evidence_found and not self.uncertainty_flag and not self.exact_quote and not self.unverified_quote:
            raise ValueError("If explicit evidence is found, an exact quote must be provided.")
        return self

def verify_quote_against_sources(quote: Optional[str], cluster_notes: List[dict]) -> Tuple[bool, Optional[str], Optional[int], Optional[int], Optional[str], bool]:
    """
    Programmatic verification: Enforces that the LLM's returned quote maps to an authentic
    span within one of the original source notes.
    
    Returns: (is_verified, note_id, char_start, char_end, verbatim_source_span, is_literal_match)
      - verbatim_source_span: The EXACT slice raw_text[char_start:char_end] from the unnormalized source note.
      - is_literal_match: True if quote was ALREADY character-for-character identical without case/whitespace normalization.
    """
    if not quote or not quote.strip():
        return False, None, None, None, None, False

    clean_quote = quote.strip().strip("\"'")
    if not clean_quote:
        return False, None, None, None, None, False

    for note in cluster_notes:
        raw_text = note.get("raw_text", "")
        note_id = note.get("note_id")

        # Path 1: Exact case-sensitive literal substring match on original raw_text
        idx = raw_text.find(clean_quote)
        if idx != -1:
            verbatim_span = raw_text[idx : idx + len(clean_quote)]
            return True, note_id, idx, idx + len(clean_quote), verbatim_span, (quote == verbatim_span)

        # Path 2: Exact case-insensitive substring match directly on original raw_text
        idx_lower = raw_text.lower().find(clean_quote.lower())
        if idx_lower != -1:
            start = idx_lower
            end = idx_lower + len(clean_quote)
            verbatim_span = raw_text[start:end]
            return True, note_id, start, end, verbatim_span, (quote == verbatim_span)

        # Path 3: Whitespace/newline-tolerant regex match evaluated directly on original raw_text
        # Uses word-boundary tokens to match arbitrary whitespace variations in the original note text,
        # guaranteeing match.start() and match.end() are precise offsets into raw_text.
        words = clean_quote.split()
        if len(words) >= 2:
            pattern = r'\s+'.join(re.escape(w) for w in words)
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                start = match.start()
                end = match.end()
                verbatim_span = raw_text[start:end]
                return True, note_id, start, end, verbatim_span, (quote == verbatim_span)

    return False, None, None, None, None, False

def build_llm_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)

def validate_cluster_evidence(cluster_notes: List[dict], client=None) -> PostFallAssessmentValidation:
    if client is None:
        client = build_llm_client()
    if client is None:
        raise ValueError("GROQ_API_KEY is not configured or client build failed.")

    # Deduplicate notes by note_id to guarantee unique note representations in prompt
    seen_notes = {}
    for n in cluster_notes:
        nid = n.get("note_id")
        if nid and nid not in seen_notes:
            seen_notes[nid] = n
    deduped_notes = list(seen_notes.values()) if seen_notes else cluster_notes

    merged_text = "\n".join([f"[{n['timestamp']}] (Note {n.get('note_id', 'UNKNOWN')}): \"{n['raw_text']}\"" for n in deduped_notes])

    prompt = f"""
    You are a clinical compliance auditor evaluating a set of reconciled nursing notes for a single incident.
    Determine if a post-fall assessment was adequately documented.
    IMPORTANT: The exact_quote must be an EXACT, literal verbatim substring copied directly from the Source Notes.
    Do not paraphrase, edit, or invent text.
    
    Source Notes:
    {merged_text}
    """

    val: PostFallAssessmentValidation = client.chat.completions.create(
        model="openai/gpt-oss-120b", 
        response_model=PostFallAssessmentValidation,
        max_retries=3,
        messages=[
            {"role": "system", "content": "You extract structured clinical compliance data precisely as requested. Never hallucinate or modify source quotes."},
            {"role": "user", "content": prompt}
        ]
    )

    # Programmatic Python-level quote verification against ground truth source notes
    is_verified, note_id, char_start, char_end, verbatim_span, is_literal = verify_quote_against_sources(val.exact_quote, deduped_notes)
    val.quote_verified_in_source = is_verified
    val.is_literal_match = is_literal
    val.source_note_id = note_id
    val.char_start = char_start
    val.char_end = char_end

    # Enforce verbatim identity: Replace LLM string with the exact slice from raw note text
    if is_verified and verbatim_span:
        val.exact_quote = verbatim_span
        val.unverified_quote = None
    else:
        # Never return an unverified quote as if it were usable evidence
        if val.exact_quote:
            val.unverified_quote = val.exact_quote
            val.exact_quote = None
        # Guard against LLM quote hallucination: If LLM claims evidence is found without uncertainty,
        # but the quote cannot be verified in the source text, force uncertainty!
        if val.evidence_found and not val.uncertainty_flag:
            val.uncertainty_flag = True
            val.evidence_found = False
            val.reasoning += " [VALIDATION OVERRULE: LLM quote could not be programmatically verified as an authentic span in source notes. Candidate quote moved to unverified_quote and suppressed from evidence.]"

    return val

