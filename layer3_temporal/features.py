"""
layer3_temporal/features.py
------------------------------
Pure, deterministic temporal feature functions. Each takes the active
window's entries (oldest first) plus a candidate's ontology-defined
trigger/conflict words and returns a single [0, 1] score. No learned
parameters, no NLP — every score is directly traceable to which words
appeared, when, and in what order.

"Relevant words" are determined via the ontology's own context_triggers /
context_conflicts fields (already part of Layer 1's IntentCandidate
schema) — this is the mechanism the Phase 2 spec pointed to: "relevant
previous words from the intent ontology can contribute evidence toward
a candidate."
"""
from typing import List

from intent.conversation import ConversationEntry


def linear_decay(elapsed_seconds: float, window_seconds: float) -> float:
    """1.0 at elapsed=0, decaying linearly to 0.0 at elapsed>=window_seconds."""
    if window_seconds <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (elapsed_seconds / window_seconds)))


def matched_entries(entries: List[ConversationEntry], words: List[str]) -> List[ConversationEntry]:
    """Entries (oldest first) whose word is in `words` (already uppercased)."""
    word_set = set(w.upper() for w in words)
    return [e for e in entries if e.word in word_set]


def compute_recency(matches: List[ConversationEntry], now: float, window_seconds: float) -> float:
    """
    FEATURE 1 — RECENCY.
    How fresh is the single most-recent relevant word, relative to now?
    0.0 if there are no matches.
    """
    if not matches:
        return 0.0
    most_recent = max(e.timestamp for e in matches)
    return linear_decay(now - most_recent, window_seconds)


def compute_sequence_strength(matches: List[ConversationEntry],
                               all_entries: List[ConversationEntry]) -> float:
    """
    FEATURE 3 — SEQUENCE RELATIONSHIP.
    Rewards relevant words that occurred IMMEDIATELY before the current word
    more than ones further back. `all_entries` is the full window
    (oldest first); a match at the very end of the list (immediately prior
    to the current word) gets weight 1.0, two back gets 0.5, three back
    gets 0.33, etc. Multiple matches sum (capped at 1.0), so a coherent
    sequence of several relevant words scores higher than a single one.
    """
    if not matches or not all_entries:
        return 0.0
    n = len(all_entries)
    match_ids = {id(e) for e in matches}
    total = 0.0
    for i, e in enumerate(all_entries):
        if id(e) in match_ids:
            position_back = n - i   # last entry -> 1, one before -> 2, ...
            total += 1.0 / position_back
    return min(1.0, total)


def compute_frequency(matches: List[ConversationEntry], saturation_count: int) -> float:
    """
    FEATURE 2 — WORD FREQUENCY.
    How often have relevant words appeared in the window, normalized so
    `saturation_count`+ occurrences saturate to 1.0.
    """
    if saturation_count <= 0:
        return 0.0
    return min(1.0, len(matches) / saturation_count)


def compute_time_gap_coherence(matches: List[ConversationEntry], window_seconds: float) -> float:
    """
    FEATURE 4 — TIME GAP.
    How tightly clustered ARE the relevant words themselves (not vs. now)?
    With 2+ matches: the largest gap between consecutive relevant words is
    the weakest link — a single wide gap in an otherwise tight sequence
    still weakens overall coherence.
    With exactly 1 match, there is no internal gap to measure -> neutral 1.0.
    With 0 matches -> 0.0 (nothing to be coherent about).
    """
    if not matches:
        return 0.0
    if len(matches) == 1:
        return 1.0
    ts = sorted(e.timestamp for e in matches)
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    worst_gap = max(gaps)
    return linear_decay(worst_gap, window_seconds)


def compute_repetition(entries: List[ConversationEntry], current_word: str,
                        saturation_count: int) -> float:
    """
    FEATURE 5 — REPETITION.
    How many prior occurrences of the CURRENT word are already in the
    window? This is a global (not candidate-specific) fact about the
    conversation, saturating at `saturation_count` prior repeats.
    """
    if saturation_count <= 0:
        return 0.0
    count = sum(1 for e in entries if e.word == current_word.upper())
    return min(1.0, count / saturation_count)


def conversation_length(entries: List[ConversationEntry]) -> int:
    """FEATURE 6 — CONVERSATION LENGTH: words currently in the active window."""
    return len(entries)
