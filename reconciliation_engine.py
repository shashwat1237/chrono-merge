import re
import numpy as np
from datetime import datetime, timedelta
import dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

def extract_event_timestamp(note_text: str, note_timestamp_str: str) -> tuple:
    """
    Generalized clinical temporal parser:
    1. Resolves relative calendar day offsets and elapsed intervals:
       - 'yesterday', 'prior day', 'previous day', 'the day before', 'last night'
       - 'N days ago', 'N d ago', 'N hours ago', 'N hrs ago'
       - 'earlier this week', 'a few days ago', 'several days ago', 'past week'
       - Named weekday references ('on monday', 'reviewed monday fall', 'prior tuesday incident')
    2. Extracts explicit clock and military times (e.g. '0200', '15:00', '2am', '0200h', 'approx 0300').
    3. Resolves shift-relative markers ('overnight', 'noc shift', 'night shift', 'day shift', 'pm shift', 'morning shift').
    4. Composes date offsets with explicit clock times (e.g. 'yesterday at 21:00').
    5. Accommodates midnight documentation rollovers.
    6. Defaults gracefully to note creation timestamp.

    Returns: (event_datetime, is_retrospective_flag)
    """
    note_dt = dateparser.parse(note_timestamp_str) or dateparser.parse("now")
    t = note_text.lower()

    # Step 1: Detect relative day/date offsets
    day_offset = 0
    is_retrospective = False
    explicit_hour = None
    explicit_minute = None

    # 1a. Elapsed hour offsets (e.g. 'approx 3 hours ago', '2 hrs ago', '4h ago')
    m_hr_ago = re.search(r'\b(?:approx|around)?\s*([0-9]{1,2})\s*(?:hours?|hrs?|h)\s*ago\b', t)
    if m_hr_ago:
        hr_val = int(m_hr_ago.group(1))
        if 0 < hr_val <= 72:
            return note_dt - timedelta(hours=hr_val), True

    # 1b. Relative calendar days
    if re.search(r'\b(yesterday|prior day|previous day|the day before)\b', t):
        day_offset = 1
        is_retrospective = True
    elif re.search(r'\b(last night)\b', t):
        # 'last night' implies prior day if note is written during the daytime/morning
        day_offset = 1 if note_dt.hour >= 6 else 0
        is_retrospective = True
    else:
        # Check 'N days ago' (e.g. '2 days ago', '3 d ago')
        m_days = re.search(r'\b([1-9]|1[0-4])\s*(?:days?|d)\s*ago\b', t)
        if m_days:
            day_offset = int(m_days.group(1))
            is_retrospective = True
        elif re.search(r'\b(earlier this week|a few days ago|several days ago|past week)\b', t):
            day_offset = 2
            is_retrospective = True
        else:
            # Check retrospective named weekdays (e.g. 'last monday', 'reviewed monday's fall', 'prior tuesday incident')
            weekdays = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            m_wday = re.search(
                r'\b(?:last|prior|previous|reviewed|following)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
                r'|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\'?s?\s+(?:fall|incident|slip|event|drop)\b',
                t
            )
            if m_wday:
                w_name = m_wday.group(1) or m_wday.group(2)
                target_w = weekdays[w_name]
                curr_w = note_dt.weekday()
                w_diff = (curr_w - target_w) % 7
                matched_str = m_wday.group(0)
                if 'last' in matched_str or 'prior' in matched_str or 'previous' in matched_str:
                    if w_diff == 0:
                        w_diff = 7
                if 1 <= w_diff <= 7:
                    day_offset = w_diff
                    is_retrospective = True

            # 1c. Explicit calendar date in text (e.g. '2026-10-18', '10/18/2026')
            m_iso = re.search(r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b', t)
            if m_iso:
                y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
                try:
                    target_date = datetime(y, mo, d)
                    diff_days = (note_dt.date() - target_date.date()).days
                    if diff_days >= 0:
                        day_offset = diff_days
                        if diff_days > 0:
                            is_retrospective = True
                except ValueError:
                    pass
            else:
                m_us = re.search(r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](20\d{2})\b', t)
                if m_us:
                    mo, d, y = int(m_us.group(1)), int(m_us.group(2)), int(m_us.group(3))
                    try:
                        target_date = datetime(y, mo, d)
                        diff_days = (note_dt.date() - target_date.date()).days
                        if diff_days >= 0:
                            day_offset = diff_days
                            if diff_days > 0:
                                is_retrospective = True
                    except ValueError:
                        pass

    # Step 2: Extract explicit clock and military times
    # 2a. Colon formats: 15:00, 02:15, 2:30pm, at approx 06:15
    m_colon = re.search(r'\b(?:at|approx\.?|around|about|@)?\s*([0-2]?[0-9]):([0-5][0-9])\s*(am|pm)?\b', t)
    if m_colon:
        hh = int(m_colon.group(1))
        mm = int(m_colon.group(2))
        meridiem = m_colon.group(3)
        if meridiem == 'pm' and hh < 12:
            hh += 12
        elif meridiem == 'am' and hh == 12:
            hh = 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            explicit_hour = hh
            explicit_minute = mm

    # 2b. 12-hour am/pm format: 2am, 2pm, at 8 am
    if explicit_hour is None:
        m_ampm = re.search(r'\b(?:at|approx\.?|around|about|@)?\s*([1-9]|1[0-2])\s*(am|pm)\b', t)
        if m_ampm:
            hh = int(m_ampm.group(1))
            mm = 0
            meridiem = m_ampm.group(2)
            if meridiem == 'pm' and hh < 12:
                hh += 12
            elif meridiem == 'am' and hh == 12:
                hh = 0
            if 0 <= hh <= 23:
                explicit_hour = hh
                explicit_minute = mm

    # 2c. Context-aware four-digit military time (e.g. 0200, approx 0300, at 1545, 1830 hrs, for 0615 fall)
    # Strictly guards against date strings (e.g. 2026-10-18) and bare years (e.g. 2026, 2024)
    if explicit_hour is None:
        # Category A: Leading-zero military time (0000 - 0959, e.g. 0200, 0615, approx 0300)
        # Years never start with 0; must not be adjacent to date delimiters
        m_mil_zero = re.search(
            r'(?<![-/\d])(?<!\d\.)\b(?:(?:at|approx\.?|around|about|@|ca\.?|for|since|following|prior to)\s+)?(0[0-9])([0-5][0-9])(?:\s*(?:h|hr|hrs|hours))?\b(?![-/\d])(?!\.\d)',
            t
        )
        if m_mil_zero:
            hh = int(m_mil_zero.group(1))
            mm = int(m_mil_zero.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                explicit_hour = hh
                explicit_minute = mm
        else:
            # Category B: Non-leading-zero military time (1000 - 2359, e.g. at 1545, approx 1830, 2000 hrs)
            # Tightened: Only unambiguous clock prepositions allowed (general prepositions like for, since, by, from excluded).
            # Modern calendar years (1900-2099) are strictly guarded: they must never be parsed as clock times
            # unless accompanied by an explicit time unit (e.g. '2000 hrs').
            m_mil_ctx = re.search(
                r'(?<![-/\d])(?<!\d\.)\b(?:at|approx\.?|around|about|@|ca\.?|time(?:\s+of)?(?:\s*:)?)\s+([12][0-9])([0-5][0-9])(?:\s*(?:h|hr|hrs|hours))?\b(?![-/\d])(?!\.\d)'
                r'|(?<![-/\d])(?<!\d\.)\b([12][0-9])([0-5][0-9])\s*(?:h|hr|hrs|hours)\b(?![-/\d])(?!\.\d)'
                r'|(?<![-/\d])(?<!\d\.)\b([12][0-9])([0-5][0-9])\s+(?:fall|incident|event|slip|slide|drop)\b(?![-/\d])(?!\.\d)',
                t
            )
            if m_mil_ctx:
                matched_text = m_mil_ctx.group(0)
                has_time_unit = bool(re.search(r'\b(?:h|hr|hrs|hours)\b', matched_text))
                m_digits = re.search(r'\b([12][0-9])([0-5][0-9])\b', matched_text)
                if m_digits:
                    hh = int(m_digits.group(1))
                    mm = int(m_digits.group(2))
                    val = hh * 100 + mm
                    if 1900 <= val <= 2099 and not has_time_unit:
                        pass
                    elif 0 <= hh <= 23 and 0 <= mm <= 59:
                        explicit_hour = hh
                        explicit_minute = mm

    # Step 3: Extract shift-relative and time-of-day markers if no explicit clock time
    shift_hour = None
    if explicit_hour is None:
        if re.search(r'\b(overnight|night shift|noc shift|noc\b|graveyard|third shift|late night)\b', t):
            shift_hour = 2
            is_retrospective = True
        elif re.search(r'\b(morning shift|early morning|day shift|first shift|am shift|this morning)\b', t):
            shift_hour = 8
            is_retrospective = True
        elif re.search(r'\b(evening shift|swing shift|second shift|pm shift|this evening|dinner|dinnertime)\b', t):
            shift_hour = 18
            is_retrospective = True
        elif re.search(r'\b(afternoon|midday|lunchtime)\b', t):
            shift_hour = 14
            is_retrospective = True

    # Step 4: Compose target datetime
    base_date = note_dt - timedelta(days=day_offset)

    if explicit_hour is not None:
        try:
            event_dt = base_date.replace(hour=explicit_hour, minute=explicit_minute, second=0, microsecond=0)
            # Rollover check: If no day offset was detected, but event time is > 1 hour in the future of note_dt
            # (e.g. event documented at 01:00 AM regarding a 23:30 PM event)
            if day_offset == 0 and event_dt > note_dt + timedelta(hours=1):
                event_dt -= timedelta(days=1)
                is_retrospective = True
            return event_dt, is_retrospective
        except (ValueError, OverflowError):
            pass

    if shift_hour is not None:
        event_dt = base_date.replace(hour=shift_hour, minute=0, second=0, microsecond=0)
        return event_dt, True

    if day_offset > 0:
        # Default to 14:00 (mid-day clinical default) when only a relative day was specified
        event_dt = base_date.replace(hour=14, minute=0, second=0, microsecond=0)
        return event_dt, True

    # Fallback: defaults gracefully to note creation timestamp
    return note_dt, False

def parse_temporal_window(timestamp_str: str, raw_text: str = "", fuzzy_hours: int = 2):
    event_dt, _ = extract_event_timestamp(raw_text, timestamp_str)
    return event_dt - timedelta(hours=fuzzy_hours), event_dt + timedelta(hours=fuzzy_hours)

def calculate_temporal_distance(window_a, window_b, max_gap_hours: int = 48, shift_tau: float = 8.0):
    """
    Asymmetric clinical shift decay function:
    Uses an exponential half-life calibrated to hospital shift boundaries (tau = 8 hours).
    Seamlessly accommodates delayed physician reviews across a shift while separating independent days.
    """
    a_min, a_max = window_a
    b_min, b_max = window_b
    
    if (a_min <= b_max) and (a_max >= b_min):
        return 0.0
    
    gap = min(abs((a_min - b_max).total_seconds()), abs((b_min - a_max).total_seconds()))
    gap_hours = gap / 3600.0
    
    if gap_hours > max_gap_hours:
        return np.inf
    
    return 1.0 - np.exp(-gap_hours / shift_tau)

def has_recurrence_marker(text: str) -> bool:
    """
    General semantic recurrence detector:
    Identifies clinical language stating an explicit incident boundary or repeat event.
    """
    pattern = r'\b(second fall|another fall|new incident|new fall|re-fall|recurrence of (?:fall|incident)|repeat fall|tumble again|fell again|fallen again|fell a second time|subsequent fall)\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

def group_mentions_by_note(mentions):
    """
    Aggregates active mentions into distinct note documents.
    Preserves resident identity to enforce strict multi-resident boundary isolation.
    """
    notes_dict = {}
    for m in mentions:
        if m.note_id not in notes_dict:
            notes_dict[m.note_id] = {
                "note_id": m.note_id,
                "timestamp": m.timestamp,
                "raw_text": m.raw_text,
                "resident_id": getattr(m, "resident_id", "DEFAULT_RESIDENT"),
                "mentions": []
            }
        notes_dict[m.note_id]["mentions"].append(m)
    return list(notes_dict.values())

def compute_note_distance_matrix(note_records, w_time=0.6, w_sem=0.4):
    n = len(note_records)
    distance_matrix = np.zeros((n, n))
    
    if n == 0:
        return distance_matrix
        
    texts = [nr["raw_text"] for nr in note_records]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    sem_distances = cosine_distances(tfidf_matrix)
    windows = [parse_temporal_window(nr["timestamp"], nr["raw_text"]) for nr in note_records]
    resident_ids = [nr.get("resident_id", "DEFAULT_RESIDENT") for nr in note_records]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0.0
                continue

            # Challenge 8: Multi-Resident Isolation
            # If notes refer to different residents (e.g. roommates in same room), they must NEVER merge
            if resident_ids[i] != resident_ids[j]:
                distance_matrix[i, j] = np.inf
                continue
                
            # Disjunction check: If either note states a recurrence and times are distinct
            if (has_recurrence_marker(texts[i]) or has_recurrence_marker(texts[j])):
                t_diff = abs((windows[i][0] - windows[j][0]).total_seconds()) / 3600.0
                if t_diff >= 2.0:
                    distance_matrix[i, j] = np.inf
                    continue

            d_time = calculate_temporal_distance(windows[i], windows[j])
            if d_time == np.inf:
                distance_matrix[i, j] = np.inf
                continue
            
            # Combine temporal distance and semantic distance
            distance_matrix[i, j] = (w_time * d_time) + (w_sem * sem_distances[i, j])
            
    return distance_matrix

def reconcile_incidents(mentions, distance_threshold=0.55):
    """
    Reconciles mentions into coherent real-world incident clusters at the note level.
    Guarantees cross-resident isolation, temporal retrospective linkage, and average-linkage chain stability.
    """
    if not mentions:
        return {}
        
    note_records = group_mentions_by_note(mentions)
    if len(note_records) == 1:
        return {0: mentions}
        
    dist_matrix = np.nan_to_num(compute_note_distance_matrix(note_records), posinf=9999.0)
    
    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        metric='precomputed', 
        linkage='average', 
        distance_threshold=distance_threshold
    )
    
    labels = clusterer.fit_predict(dist_matrix)
    
    clusters = {}
    for idx, cid in enumerate(labels):
        clusters.setdefault(int(cid), []).extend(note_records[idx]["mentions"])
        
    return clusters
