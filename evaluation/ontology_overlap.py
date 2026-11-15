"""
evaluation/ontology_overlap.py
----------------------------------
PHASE 9 -- ontology robustness report.

Read-only analysis of data/intent_ontology.json: for every trigger word
that appears anywhere, which intents (across ALL words, not just one) it
is a context_trigger for. A word that triggers many different intents
across different target words is a structural ambiguity risk -- this
explains WHY certain failure patterns happen (e.g. HELP triggering both
STOP's pain_stop and DOCTOR's medical_care_request, which is exactly
what Phase 9's adversarial category exploits).

This module does NOT modify the ontology. It only reads and reports.
"""
from collections import defaultdict
from typing import Dict, List

from intent.ontology import IntentOntology


def build_trigger_index() -> Dict[str, List[dict]]:
    """trigger_word -> [{word, intent_id}, ...] for every (word, intent)
    pair where trigger_word appears in that intent's context_triggers."""
    ontology = IntentOntology()
    index = defaultdict(list)
    for word in ontology._data.get("words", {}):
        for candidate in ontology.lookup(word):
            for trigger in candidate.context_triggers:
                index[trigger].append({"word": word, "intent_id": candidate.id})
    return dict(index)


def ambiguity_report() -> List[dict]:
    """One row per trigger word: how many DISTINCT intents it's associated
    with (across all target words), and an ambiguity_level label."""
    index = build_trigger_index()
    rows = []
    for trigger_word, associations in sorted(index.items()):
        distinct_intents = sorted({a["intent_id"] for a in associations})
        n = len(distinct_intents)
        if n == 1:
            level = "none"
        elif n == 2:
            level = "low"
        elif n <= 4:
            level = "moderate"
        else:
            level = "high"
        rows.append({
            "trigger_word": trigger_word,
            "n_associated_intents": n,
            "associated_intents": distinct_intents,
            "ambiguity_level": level,
        })
    return sorted(rows, key=lambda r: -r["n_associated_intents"])


def sparse_intent_report(min_triggers: int = 3) -> List[dict]:
    """Intents with very few context_triggers -- fragile, easy to under-evidence."""
    ontology = IntentOntology()
    rows = []
    for word in ontology._data.get("words", {}):
        for candidate in ontology.lookup(word):
            if len(candidate.context_triggers) < min_triggers:
                rows.append({
                    "word": word, "intent_id": candidate.id,
                    "n_triggers": len(candidate.context_triggers),
                    "triggers": candidate.context_triggers,
                })
    return sorted(rows, key=lambda r: r["n_triggers"])


def summary() -> dict:
    amb = ambiguity_report()
    sparse = sparse_intent_report()
    high_ambiguity = [r for r in amb if r["ambiguity_level"] == "high"]
    return {
        "n_distinct_trigger_words": len(amb),
        "n_high_ambiguity_triggers": len(high_ambiguity),
        "high_ambiguity_triggers": high_ambiguity,
        "n_sparse_intents": len(sparse),
        "sparse_intents": sparse,
    }
