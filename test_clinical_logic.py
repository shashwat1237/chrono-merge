import sys
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from evidence_validator import verify_quote_against_sources, PostFallAssessmentValidation, validate_cluster_evidence
from nlp_extractor import build_medspacy_pipeline, extract_clinical_mentions, filter_active_incidents, ExtractedMention
from reconciliation_engine import extract_event_timestamp, has_recurrence_marker

def test_quote_verification_exact_and_offsets():
    """Validates that character offsets map exactly to the unnormalized original note text and enforce verbatim identity."""
    notes = [
        {"note_id": "N1", "raw_text": "Header info:\n  Resident fell near bed at 0615.\n  Post-fall check normal."}
    ]
    
    # 1. Exact case-sensitive literal match
    quote = "Post-fall check normal."
    verified, note_id, start, end, verbatim_span, is_literal = verify_quote_against_sources(quote, notes)
    assert verified is True
    assert is_literal is True
    assert note_id == "N1"
    assert notes[0]["raw_text"][start:end] == quote
    assert verbatim_span == quote

    # 2. Case-insensitive / whitespace-tolerant match: Must extract authentic original slice and flag non-literal
    quote_nl = "resident fell near bed at 0615. post-fall check normal."
    verified_nl, note_id_nl, s_nl, e_nl, v_nl, is_lit_nl = verify_quote_against_sources(quote_nl, notes)
    assert verified_nl is True
    assert is_lit_nl is False  # LLM had casing/whitespace difference
    assert note_id_nl == "N1"
    # Offsets extract the authentic original raw_text slice
    extracted = notes[0]["raw_text"][s_nl:e_nl]
    assert v_nl == extracted
    assert "Resident fell near bed at 0615." in extracted
    assert "Post-fall check normal." in extracted

    # 3. Deliberately fabricated quote must fail verification
    fake_quote = "Dr. Adams performed a complete orthopedic examination and neuro check."
    v_fake, nid_fake, s_fake, e_fake, vspan_fake, is_lit_fake = verify_quote_against_sources(fake_quote, notes)
    assert v_fake is False
    assert nid_fake is None
    assert vspan_fake is None

    # 4. Strict literal match test: Candidates with added quotes or padding whitespace
    # must verify to the authentic source span, but is_literal_match MUST be False!
    quote_with_quotes = '"Post-fall check normal."'
    vq, nid_q, sq, eq, span_q, lit_q = verify_quote_against_sources(quote_with_quotes, notes)
    assert vq is True
    assert span_q == "Post-fall check normal."
    assert lit_q is False, "Candidate with added quotation marks must not be classified as literal match"

    quote_with_spaces = "  Post-fall check normal.  "
    vs, nid_s, ss, es, span_s, lit_s = verify_quote_against_sources(quote_with_spaces, notes)
    assert vs is True
    assert span_s == "Post-fall check normal."
    assert lit_s is False, "Candidate with padding whitespace must not be classified as literal match"

def test_semantic_fallback_negation_scoping():
    """
    Validates that:
    1. Sentences with unscripted fall phrasing (no MedSpaCy target keywords) explicitly
       trigger the zero-shot semantic fallback stage.
    2. Incidental negative findings (e.g. 'no dizziness', 'without pain') do NOT negate the fallback mention.
    3. Explicit incident denials (e.g. 'no fall occurred') properly suppress the fallback mention.
    """
    nlp = build_medspacy_pipeline()
    
    # Clinical note using phrasing with ZERO MedSpaCy target keywords:
    # "lost balance and went down" is not a target rule in build_medspacy_pipeline()
    unscripted_text = "Resident experienced sudden loss of balance and went down hard onto carpet, denies dizziness or head strike."
    
    # Verify fast-path target rules produce 0 entities
    fast_path_doc = nlp(unscripted_text)
    assert len(fast_path_doc.ents) == 0, "Fast-path must find 0 entities to ensure semantic fallback is exercised"
    
    note_pos = {
        "note_id": "TEST_FALLBACK_01",
        "timestamp": "2026-10-17T10:00:00",
        "text": unscripted_text
    }
    mentions = extract_clinical_mentions(note_pos, nlp)
    # Stage 2 zero-shot fallback must capture this mention
    assert len(mentions) > 0, "Semantic fallback should extract unscripted fall phrasing"
    assert any(m.concept_category == "INCIDENT_FALL" for m in mentions)
    
    # Scoping check: Incidental negation 'denies dizziness' must NOT suppress the fall
    active = filter_active_incidents(mentions)
    assert len(active) > 0, "Active fallback fall should not be suppressed by 'denies dizziness'"

    # Note with genuine unscripted incident denial (contains ZERO MedSpaCy target keywords: no "fall", "slip", "trip", etc.)
    unscripted_denial_text = "Resident experienced sudden loss of balance and went down hard, but nurse confirmed no incident occurred."
    fast_path_doc_neg = nlp(unscripted_denial_text)
    assert len(fast_path_doc_neg.ents) == 0, "Fast-path must find 0 entities in denial text to prove semantic fallback is exercised"

    note_neg = {
        "note_id": "TEST_FALLBACK_02",
        "timestamp": "2026-10-17T10:00:00",
        "text": unscripted_denial_text
    }
    mentions_neg = extract_clinical_mentions(note_neg, nlp)
    # Stage 2 zero-shot fallback must run and capture the mention
    assert len(mentions_neg) > 0, "Semantic fallback stage must run on unscripted text lacking MedSpaCy keywords"
    assert any(m.concept_category == "INCIDENT_FALL" for m in mentions_neg)
    assert mentions_neg[0].confidence < 1.0, "Confidence < 1.0 proves mention originated from Stage 2 semantic fallback"
    assert mentions_neg[0].is_negated is True, "Semantic fallback targeted predicate negation must flag mention as negated"

    active_neg = filter_active_incidents(mentions_neg)
    assert len(active_neg) == 0, "Negated semantic fallback mention must be suppressed by filter_active_incidents"

def test_semantic_fallback_explicit_verification_and_denial():
    """
    Separately and definitively verifies:
    1. The fast-path target matcher finds 0 entities on unscripted text, guaranteeing that
       the mention is not generated by MedSpaCy rules.
    2. The semantic fallback stage explicitly runs and extracts an active unscripted mention (confidence < 1.0).
    3. When an explicit unscripted denial is present ("nurse confirmed no incident occurred"),
       the fallback stage still runs, correctly tags is_negated=True, and filter_active_incidents suppresses it.
    """
    nlp = build_medspacy_pipeline()

    # Step 1: Positive unscripted text without any MedSpaCy keywords
    pos_text = "Patient experienced sudden loss of footing and dropped to carpeted ground."
    assert len(nlp(pos_text).ents) == 0, "Fast-path must find 0 entities on positive unscripted text"

    pos_note = {"note_id": "FALLBACK_POS", "timestamp": "2026-10-17T14:00:00", "text": pos_text}
    pos_mentions = extract_clinical_mentions(pos_note, nlp)
    assert len(pos_mentions) > 0, "Semantic fallback must generate a mention"
    assert pos_mentions[0].confidence < 1.0, "Mention must originate from semantic fallback"
    assert pos_mentions[0].is_negated is False, "Positive mention must not be negated"
    pos_active = filter_active_incidents(pos_mentions)
    assert len(pos_active) == 1, "Positive fallback mention must survive filtering as active"

    # Step 2: Negative unscripted text with explicit incident denial (no "fall" keyword)
    neg_text = "Patient experienced sudden loss of footing and dropped to carpeted ground, but charge RN determined incident was ruled out."
    assert len(nlp(neg_text).ents) == 0, "Fast-path must find 0 entities on unscripted denial text"

    neg_note = {"note_id": "FALLBACK_NEG", "timestamp": "2026-10-17T14:30:00", "text": neg_text}
    neg_mentions = extract_clinical_mentions(neg_note, nlp)
    assert len(neg_mentions) > 0, "Semantic fallback must run and generate mention for unscripted phrasing"
    assert neg_mentions[0].confidence < 1.0, "Mention must originate from Stage 2 semantic fallback"
    assert neg_mentions[0].is_negated is True, "Semantic fallback predicate negation must mark mention as negated"
    neg_active = filter_active_incidents(neg_mentions)
    assert len(neg_active) == 0, "Negated semantic fallback mention must be cleanly suppressed"

def test_sentence_level_rule_out_scoping():
    """
    Validates that a clinical rule-out in sentence 2 does not erase a genuine incident in sentence 1.
    """
    nlp = build_medspacy_pipeline()
    multi_sentence_note = {
        "note_id": "TEST_03",
        "timestamp": "2026-10-17T12:00:00",
        "text": (
            "Resident found on floor beside bed at 0200 with abrasion on right knee. "
            "Separate incident at 0800 was evaluated by RN and fall ruled out after assessment."
        )
    }
    mentions = extract_clinical_mentions(multi_sentence_note, nlp)
    active = filter_active_incidents(mentions)
    
    # The 0200 bedside fall in sentence 1 must be preserved
    active_spans = [m.span_text for m in active]
    assert any("floor" in s or "fell" in s for s in active_spans), "Sentence 1 fall should survive Sentence 2 rule-out"

def test_generalized_clinical_temporal_parser():
    """
    Validates generalized clinical temporal parsing without overfitting to benchmark-specific strings:
    1. Composite relative day + explicit clock time ('yesterday at 21:00').
    2. Elapsed hours ago offset ('approx 3 hours ago').
    3. Relative day counts ('2 days ago').
    4. Retrospective named weekdays ('reviewed monday's fall' written on Thursday).
    5. Clinical shift markers ('noc shift' -> 02:00, 'morning shift' -> 08:00).
    6. Midnight documentation rollover (event at 23:30 logged at 01:15).
    7. Generalized recurrence markers ('fell again', 'fell a second time', 'subsequent fall').
    """
    # 1. Composite date offset + clock format
    note_ts = "2026-11-12T10:00:00"  # Thursday
    txt_composite = "Resident slipped yesterday at 21:00 while walking to bathroom."
    dt_comp, retro_comp = extract_event_timestamp(txt_composite, note_ts)
    assert retro_comp is True
    assert dt_comp.year == 2026 and dt_comp.month == 11 and dt_comp.day == 11
    assert dt_comp.hour == 21 and dt_comp.minute == 0

    # 2. Elapsed hour offset
    txt_hr = "Fall occurred approx 3 hours ago, resident assisted to bed."
    dt_hr, retro_hr = extract_event_timestamp(txt_hr, note_ts)
    assert retro_hr is True
    assert dt_hr.hour == 7 and dt_hr.day == 12

    # 3. Relative day counts
    txt_days = "Incident reported 2 days ago during transport."
    dt_days, retro_days = extract_event_timestamp(txt_days, note_ts)
    assert retro_days is True
    assert dt_days.day == 10 and dt_days.hour == 14

    # 4. Retrospective named weekday
    txt_wday = "Follow-up note: reviewed monday's fall with therapy team."
    dt_wday, retro_wday = extract_event_timestamp(txt_wday, note_ts)  # Thursday to Monday = 3 days prior (Nov 9)
    assert retro_wday is True
    assert dt_wday.day == 9 and dt_wday.hour == 14

    # 5. Clinical shift markers
    txt_noc = "Resident found on floor during noc shift, uninjured."
    dt_noc, retro_noc = extract_event_timestamp(txt_noc, note_ts)
    assert retro_noc is True
    assert dt_noc.hour == 2

    # 6. Midnight rollover
    ts_early_am = "2026-11-12T01:15:00"
    txt_rollover = "Resident lost footing near bed at 23:30."
    dt_roll, retro_roll = extract_event_timestamp(txt_rollover, ts_early_am)
    assert dt_roll.day == 11 and dt_roll.hour == 23 and dt_roll.minute == 30

    # 7. Generalized recurrence detection
    assert has_recurrence_marker("Resident fell again in dining room.") is True
    assert has_recurrence_marker("Resident fell a second time today.") is True
    assert has_recurrence_marker("Subsequent fall noted at 19:00.") is True
    assert has_recurrence_marker("Routine shift rounding completed.") is False

    # 8. Context-aware military times vs date strings, bare years & prepositional year phrases
    # Must NOT misinterpret year 2026 as 20:26 clock time in date strings, bare years, or temporal phrases
    dt_iso_date, _ = extract_event_timestamp("Fall documented on 2026-10-18.", "2026-10-18T10:00:00")
    assert dt_iso_date.hour == 10 and dt_iso_date.minute == 0, f"Expected 10:00 default, got {dt_iso_date.hour}:{dt_iso_date.minute}"

    dt_slash_date, _ = extract_event_timestamp("Fall documented on 10/18/2026.", "2026-10-18T10:00:00")
    assert dt_slash_date.hour == 10 and dt_slash_date.minute == 0

    dt_bare_yr, _ = extract_event_timestamp("Resident had a fall in 2026.", "2026-10-18T10:00:00")
    assert dt_bare_yr.hour == 10 and dt_bare_yr.minute == 0

    dt_yr_rev, _ = extract_event_timestamp("Annual fall review 2024.", "2026-10-18T10:00:00")
    assert dt_yr_rev.hour == 10 and dt_yr_rev.minute == 0

    # Temporal prepositional phrases with years (e.g. 'for 2026', 'since 2026') must NOT parse as clock times
    dt_for_2026, _ = extract_event_timestamp("Care plan for 2026.", "2026-10-18T10:00:00")
    assert dt_for_2026.hour == 10 and dt_for_2026.minute == 0, f"Expected 10:00 default for 'for 2026', got {dt_for_2026.hour}:{dt_for_2026.minute}"

    dt_since_2026, _ = extract_event_timestamp("Resident has resided in facility since 2026.", "2026-10-18T10:00:00")
    assert dt_since_2026.hour == 10 and dt_since_2026.minute == 0, f"Expected 10:00 default for 'since 2026', got {dt_since_2026.hour}:{dt_since_2026.minute}"

    dt_for_fall_prev, _ = extract_event_timestamp("Goals for 2026 fall prevention program.", "2026-10-18T10:00:00")
    assert dt_for_fall_prev.hour == 10 and dt_for_fall_prev.minute == 0, f"Expected 10:00 default for 'for 2026 fall prevention', got {dt_for_fall_prev.hour}:{dt_for_fall_prev.minute}"

    dt_since_prev_yr, _ = extract_event_timestamp("No falls recorded since 2025.", "2026-10-18T10:00:00")
    assert dt_since_prev_yr.hour == 10 and dt_since_prev_yr.minute == 0, f"Expected 10:00 default for 'since 2025', got {dt_since_prev_yr.hour}:{dt_since_prev_yr.minute}"

    # Legitimate military times MUST still parse accurately
    dt_0200, _ = extract_event_timestamp("Resident found sitting on floor 0200.", "2026-10-18T05:00:00")
    assert dt_0200.hour == 2 and dt_0200.minute == 0

    dt_approx, _ = extract_event_timestamp("Resident found beside bed at approx 0300.", "2026-10-18T05:00:00")
    assert dt_approx.hour == 3 and dt_approx.minute == 0

    dt_at1545, _ = extract_event_timestamp("Resident fell at 1545 in corridor.", "2026-10-18T16:00:00")
    assert dt_at1545.hour == 15 and dt_at1545.minute == 45

    dt_for0615, _ = extract_event_timestamp("Post-fall assessment completed for 0615 bedside fall.", "2026-10-18T07:00:00")
    assert dt_for0615.hour == 6 and dt_for0615.minute == 15

    dt_hrs, _ = extract_event_timestamp("Incident occurred at 2000 hrs.", "2026-10-18T21:00:00")
    assert dt_hrs.hour == 20 and dt_hrs.minute == 0

def test_unverified_quote_quarantined_from_usable_evidence():
    """
    Validates Correction #4: An unverified quote must NEVER be returned as if it were usable evidence.
    Tests directly via validate_cluster_evidence() using a mocked LLM client.
    When programmatic verification fails:
    1. exact_quote must be cleared to None.
    2. unverified_quote must store the rejected string separately.
    3. evidence_found must be suppressed to False.
    4. uncertainty_flag must be forced to True.
    """
    from unittest.mock import MagicMock

    notes = [
        {
            "note_id": "N201",
            "raw_text": "Resident assisted to chair after slip. Vitals taken. Dr. informed.",
            "timestamp": "2026-10-12T14:00:00"
        }
    ]
    hallucinated_quote = "Dr. performed full neurological examination and post-fall assessment."

    # Build a mock LLM client returning a candidate validation with the hallucinated quote
    mock_client = MagicMock()
    mock_candidate_val = PostFallAssessmentValidation(
        evidence_found=True,
        exact_quote=hallucinated_quote,
        uncertainty_flag=False,
        reasoning="LLM claimed post-fall assessment was documented."
    )
    mock_client.chat.completions.create.return_value = mock_candidate_val

    # Execute validate_cluster_evidence directly through the real pipeline function
    result = validate_cluster_evidence(notes, client=mock_client)

    # Strict assertion checks:
    # 1. exact_quote (usable evidence) must be None
    assert result.exact_quote is None
    # 2. unverified_quote must preserve the candidate string for quarantined audit review
    assert result.unverified_quote == hallucinated_quote
    # 3. evidence_found must be suppressed to False
    assert result.evidence_found is False
    # 4. uncertainty_flag must be forced to True
    assert result.uncertainty_flag is True
    # 5. reasoning must explain the validation overrule
    assert "[VALIDATION OVERRULE" in result.reasoning
    # 6. quote_verified_in_source must be False
    assert result.quote_verified_in_source is False

def test_embedding_model_initialization_fail_visibly():
    """
    Validates Correction #5: get_embed_model(require_model=True) must fail visibly
    with an informative error message if the model cannot be loaded.
    Uses a mocked model loader to test failure behavior without corrupting the local environment.
    """
    from unittest.mock import patch
    import nlp_extractor
    from nlp_extractor import get_embed_model

    # Save original model reference to preserve test environment state
    original_model = nlp_extractor._EMBED_MODEL

    try:
        # 1. Test failure behavior when SentenceTransformer cannot load (e.g., offline cache missing)
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("Simulated offline weights missing")):
            nlp_extractor._EMBED_MODEL = None

            # require_model=True MUST raise RuntimeError with an actionable message
            threw_expected = False
            try:
                get_embed_model(require_model=True)
            except RuntimeError as e:
                threw_expected = True
                err_msg = str(e)
                assert "Failed to initialize SentenceTransformer model" in err_msg
                assert "A clean installation requires pretrained model artifacts" in err_msg
            assert threw_expected, "Expected RuntimeError when require_model=True and model loading fails"

            # require_model=False must return None gracefully without raising
            nlp_extractor._EMBED_MODEL = None
            fallback_res = get_embed_model(require_model=False)
            assert fallback_res is None

        # 2. Test successful initialization & caching behavior using a mocked loader (100% offline & deterministic)
        from unittest.mock import MagicMock
        dummy_model = MagicMock(name="DummySentenceTransformer")
        with patch("sentence_transformers.SentenceTransformer", return_value=dummy_model) as mock_st:
            nlp_extractor._EMBED_MODEL = None
            # First call should initialize and cache dummy_model
            loaded_model = get_embed_model(require_model=True)
            assert loaded_model is dummy_model
            assert nlp_extractor._EMBED_MODEL is dummy_model
            mock_st.assert_called_once_with('all-MiniLM-L6-v2')

            # Second call should return cached object without re-instantiating
            cached_model = get_embed_model(require_model=False)
            assert cached_model is dummy_model
            assert mock_st.call_count == 1
    finally:
        # Restore original model reference
        nlp_extractor._EMBED_MODEL = original_model


if __name__ == "__main__":
    test_quote_verification_exact_and_offsets()
    print("✓ test_quote_verification_exact_and_offsets passed")
    test_semantic_fallback_negation_scoping()
    print("✓ test_semantic_fallback_negation_scoping passed")
    test_semantic_fallback_explicit_verification_and_denial()
    print("✓ test_semantic_fallback_explicit_verification_and_denial passed")
    test_sentence_level_rule_out_scoping()
    print("✓ test_sentence_level_rule_out_scoping passed")
    test_generalized_clinical_temporal_parser()
    print("✓ test_generalized_clinical_temporal_parser passed")
    test_unverified_quote_quarantined_from_usable_evidence()
    print("✓ test_unverified_quote_quarantined_from_usable_evidence passed")
    test_embedding_model_initialization_fail_visibly()
    print("✓ test_embedding_model_initialization_fail_visibly passed")
    print("\nALL CLINICAL REGRESSION TESTS PASSED SUCCESSFULLY!")
