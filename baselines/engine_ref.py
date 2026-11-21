"""
baselines/engine_ref.py
---------------------------
Reference row for the proposed engine: the `full_reliability_weighted`
variant, reproduced with the same components, per-case in-memory seeding
helpers, NOW and label mapping that evaluation/final_variants.py uses
(imported, not copied).

Differences from final_variants.py, both deliberate:
  * user_id is a constant placeholder. user_id is not an allowed input, and
    every case gets fresh engines, so it is only a key into empty stores.
  * Timing covers lookup + Layer 3/5 evaluate + resolve + gate, AFTER
    seeding (final_variants.py times construction and seeding too).
"""
from types import SimpleNamespace

from decision_gate.gate import ConfidenceDecisionGate
from layer3_temporal.engine import TemporalContextEngine
from layer5_personal.engine import PersonalContextEngine
from reasoning.resolver import IntentResolver
from reasoning.strategies import ReliabilityWeightedStrategy
from evaluation.final_variants import (
    NOW, _predicted_label, _seed_personal, _seed_temporal_history,
)

from .common import ONTOLOGY, CaseInput, Method

_PLACEHOLDER_USER = "eval-user"


class EngineRef(Method):
    name = "engine"
    abstains = True

    def __init__(self):
        self.gate = ConfidenceDecisionGate()
        self.resolver = IntentResolver(strategy=ReliabilityWeightedStrategy())
        self._temporal = self._personal = None
        self._extras = {}

    def prepare(self, inp: CaseInput) -> None:
        seedable = SimpleNamespace(
            user_id=_PLACEHOLDER_USER,
            conversation_sequence=list(inp.conversation_sequence),
            sequence_offsets_seconds=list(inp.sequence_offsets_seconds),
            personal_seed=list(inp.personal_seed),
        )
        self._temporal = TemporalContextEngine()
        _seed_temporal_history(self._temporal, seedable)
        self._personal = PersonalContextEngine(db_path=":memory:")
        _seed_personal(self._personal, seedable)

    def predict(self, inp: CaseInput):
        word = inp.conversation_sequence[-1]
        candidates = ONTOLOGY.lookup(word)
        temporal = self._temporal.evaluate(_PLACEHOLDER_USER, word, candidates, now=NOW)
        personal = self._personal.evaluate(_PLACEHOLDER_USER, word, candidates, now=NOW)
        reasoning = self.resolver.resolve(word, candidates, temporal=temporal, personal=personal, now=NOW)
        decision = self.gate.evaluate(reasoning, now=NOW)

        ranking = [(r["candidate"], r["final_score"]) for r in decision.evidence["full_ranking"]]
        gated = _predicted_label(decision.mode, decision.resolved_candidate)
        self._extras = {
            "gated_label": gated,
            "gated_status": gated if gated in ("AMBIGUOUS", "LITERAL") else "RESOLVED",
            "conflict": decision.conflict_detected,
        }
        top, score = ranking[0] if ranking else ("LITERAL", 0.0)
        return top, score, ranking

    def extras(self):
        return dict(self._extras)

    def cleanup(self) -> None:
        self._personal.close()
        self._temporal = self._personal = None
