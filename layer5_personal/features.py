"""
layer5_personal/features.py
-------------------------------
Pure, deterministic personal-context feature functions. Every score is
computed directly from stored SQLite counts/timestamps — nothing here is
a trained parameter, and nothing is word- or intent-specific in the code
(no `if word == "HELP"` branches anywhere).
"""
import math
from typing import Optional


def personal_prior(occurrence_count: int, total_for_word: int) -> float:
    """FEATURE 2 — PERSONAL PRIOR. This candidate's share of the user's
    historical intents for this specific word. 0.0 if the word has never
    been seen for this user at all."""
    if total_for_word <= 0:
        return 0.0
    return occurrence_count / total_for_word


def recency_weight(last_seen: Optional[float], now: float, half_life_days: float) -> float:
    """FEATURE 3 — RECENCY OF PERSONAL PATTERN.
    Exponential half-life decay measured in DAYS (long-term habit), NOT
    Layer 3's linear 120-second window (current exchange) — deliberately
    a different mechanism so the two layers never duplicate each other.
    0.0 if this candidate has never been observed for this user."""
    if last_seen is None or half_life_days <= 0:
        return 0.0
    days_elapsed = max(0.0, (now - last_seen) / 86400.0)
    return 0.5 ** (days_elapsed / half_life_days)


def observation_sufficiency(occurrence_count: int, saturation_count: int) -> float:
    """FEATURE 4 — EXPERIENCE / DATA SUFFICIENCY for this specific
    word->intent pair. Saturates at `saturation_count` observations."""
    if saturation_count <= 0:
        return 0.0
    return min(1.0, occurrence_count / saturation_count)


def profile_maturity(total_user_observations: int, saturation_count: int) -> float:
    """FEATURE 5 — PERSONAL PROFILE MATURITY. Global (not candidate- or
    word-specific): how much history exists for this user AT ALL, across
    every word they've ever produced. A brand-new user scores 0.0
    regardless of which word/candidate is being evaluated."""
    if saturation_count <= 0:
        return 0.0
    return min(1.0, total_user_observations / saturation_count)


def concentration(prior: float, num_intents_for_word: int, has_row: bool) -> float:
    """How dominated is this word's history by a single intent, vs. split
    across competing ones? 1.0 = fully consistent (only one intent ever
    used, or no competition at all); 0.0 = perfectly ambiguous (this
    candidate's share matches what pure chance across the seen intents
    would predict). Used as a reliability factor, distinct from `prior`
    itself, which drives signal."""
    if num_intents_for_word <= 1:
        return 1.0 if has_row else 0.0
    uniform_baseline = 1.0 / num_intents_for_word
    if uniform_baseline >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (prior - uniform_baseline) / (1.0 - uniform_baseline)))
