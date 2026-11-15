"""
evaluation/cold_start_experiment.py
---------------------------------------
EXPERIMENT 5 — Cold start.

Sweeps a single user's HELP -> medical_assistance observation count
through {0, 1, 2, 10, 30} and records the REAL PersonalContextEngine
signal/reliability at each level, to verify reliability grows with
evidence rather than being overconfident from sparse data.
"""
from intent.ontology import IntentOntology
from layer5_personal.engine import PersonalContextEngine

NOW = 1_700_000_000.0
DAY = 86400.0

OBSERVATION_LEVELS = [0, 1, 2, 10, 30]


def run() -> list:
    ontology = IntentOntology()
    candidates = ontology.lookup("HELP")
    rows = []

    for count in OBSERVATION_LEVELS:
        engine = PersonalContextEngine(db_path=":memory:")
        user_id = f"cold_start_user_{count}"
        for _ in range(count):
            engine.update(user_id, "HELP", "medical_assistance", timestamp=NOW - 5 * DAY)

        resolution = engine.evaluate(user_id, "HELP", candidates, now=NOW)
        target = next(ev for ev in resolution.candidates if ev.candidate == "medical_assistance")

        rows.append({
            "observation_count": count,
            "signal": target.signal,
            "reliability": target.reliability,
            "profile_maturity": target.evidence["profile_maturity"],
            "observation_sufficiency": target.evidence["observation_sufficiency"],
        })
        engine.close()

    return rows


def check_monotonic_reliability(rows: list) -> bool:
    """Sanity check: reliability should not decrease as evidence accumulates."""
    reliabilities = [r["reliability"] for r in rows]
    return all(reliabilities[i] <= reliabilities[i + 1] for i in range(len(reliabilities) - 1))
