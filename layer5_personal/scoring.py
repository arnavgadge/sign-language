"""
layer5_personal/scoring.py
------------------------------
Combines the raw personal features (features.py) into two independent,
per-candidate scores:

  SIGNAL       "How strongly does this user's history support this
               candidate for this word?" — driven by their historical
               share of intents for this word (personal_prior) plus how
               recently that association was reinforced.

  RELIABILITY  "How much should the future Reasoning Engine trust this
               user's historical pattern right now?" — driven by whether
               this user has a profile at all (profile_maturity), whether
               there's enough data on this specific pair (sufficiency),
               whether their history is consistent or split across
               competing intents (concentration), and recency.

Deliberately different weighted combinations, so signal and reliability
never collapse into the same number by construction.
"""
from typing import List

from intent.ontology import IntentCandidate

from .config import Layer5Config
from .db import WordIntentRow
from .schema import PersonalEvidence
from .features import (
    personal_prior,
    recency_weight,
    observation_sufficiency,
    profile_maturity,
    concentration,
)


def score_candidate(candidate: IntentCandidate,
                     word_rows: List[WordIntentRow],
                     now: float,
                     cfg: Layer5Config,
                     total_user_observations: int) -> PersonalEvidence:

    total_for_word = sum(r.occurrence_count for r in word_rows)
    num_intents_for_word = len(word_rows)

    row = next((r for r in word_rows if r.intent_id == candidate.id), None)
    has_row = row is not None
    occurrence_count = row.occurrence_count if row else 0
    last_seen = row.last_seen if row else None

    prior = personal_prior(occurrence_count, total_for_word)
    recency = recency_weight(last_seen, now, cfg.recency_half_life_days)
    sufficiency = observation_sufficiency(occurrence_count, cfg.observation_saturation_count)
    maturity = profile_maturity(total_user_observations, cfg.profile_saturation_count)
    conc = concentration(prior, num_intents_for_word, has_row)

    signal = max(0.0, min(1.0,
        cfg.signal_weight_prior * prior +
        cfg.signal_weight_recency * recency
    ))

    reliability = max(0.0, min(1.0,
        cfg.reliability_weight_profile_maturity * maturity +
        cfg.reliability_weight_sufficiency * sufficiency +
        cfg.reliability_weight_concentration * conc +
        cfg.reliability_weight_recency * recency
    ))

    evidence = {
        "historical_count": occurrence_count,
        "total_word_intent_count": total_for_word,
        "personal_prior": round(prior, 4),
        "last_seen": last_seen,
        "recency_weight": round(recency, 4),
        "observation_sufficiency": round(sufficiency, 4),
        "profile_maturity": round(maturity, 4),
        "concentration": round(conc, 4),
        "num_competing_intents": num_intents_for_word,
        "profile_observations": total_user_observations,
    }

    return PersonalEvidence(
        candidate=candidate.id,
        signal=round(signal, 4),
        reliability=round(reliability, 4),
        evidence=evidence,
    )
