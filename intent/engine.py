"""
IntentEngine — Main orchestrator for the Context-Aware Intent Resolution Engine.

Accepts a recognized word from the upstream ASL recognition system (which is
treated as a completed black box) and returns a resolved communicative intent.

Pipeline:
  recognized word
    → Stage 1: CandidateGenerator (ontology lookup + context filter + personal prune)
    → Stage 2: Four context layers (visual, temporal, device, personal) — dual scores
    → Stage 3: IntentResolver (reliability-weighted scoring + conflict detection)
    → Stage 4: ConfidenceGate (threshold decision + fallback modes)
    → IntentOutput
"""
from typing import Optional

from .ontology import IntentOntology
from .candidate_generator import CandidateGenerator
from .context_layers import (
    VisualContextLayer,
    TemporalContextLayer,
    DeviceContextLayer,
    PersonalContextLayer,
)
from .resolver import IntentResolver
from .confidence_gate import ConfidenceGate, IntentOutput
from .conversation import ConversationHistory
from .user_model import UserModel


class IntentEngine:
    def __init__(
        self,
        user_id: str = "default",
        ontology_path: str = "data/intent_ontology.json",
        users_dir: str = "data/users",
    ):
        self.user_model = UserModel(user_id=user_id, data_dir=users_dir)
        self.ontology = IntentOntology(ontology_path=ontology_path)
        self.history = ConversationHistory(window_seconds=120, max_entries=20)

        self.candidate_gen = CandidateGenerator(
            ontology=self.ontology,
            history=self.history,
            user_model=self.user_model,
        )

        self.visual_layer = VisualContextLayer()
        self.temporal_layer = TemporalContextLayer()
        self.device_layer = DeviceContextLayer()
        self.personal_layer = PersonalContextLayer()

        self.resolver = IntentResolver()
        self.gate = ConfidenceGate()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        word: str,
        frame=None,
        motion_speed: float = 0.0,
    ) -> IntentOutput:
        """
        Resolve the communicative intent of a recognized word.

        Args:
            word          Recognized word from the ASL recognition system
            frame         Current BGR camera frame (optional, for visual layer)
            motion_speed  Hand motion speed in px/frame (optional, for urgency scoring)

        Returns:
            IntentOutput with resolved intent, confidence, output text, and debug info
        """
        word = word.upper().strip()
        if not word:
            return self._empty_output()

        # ── Stage 1: Candidate generation ─────────────────────────────────────
        candidates, gen_debug = self.candidate_gen.generate(word)
        candidates_map = {c.id: c for c in candidates}

        # ── Stage 2: Context layers ────────────────────────────────────────────
        visual_out   = self.visual_layer.compute(candidates, frame=frame, motion_speed=motion_speed)
        temporal_out = self.temporal_layer.compute(candidates, history=self.history)
        device_out   = self.device_layer.compute(candidates)
        personal_out = self.personal_layer.compute(candidates, word=word, user_model=self.user_model)

        layers = [visual_out, temporal_out, device_out, personal_out]

        # ── Stage 3: Reliability-weighted resolution ───────────────────────────
        resolution = self.resolver.resolve(candidates, layers)

        # ── Stage 4: Confidence gate ───────────────────────────────────────────
        output = self.gate.evaluate(word, resolution, candidates_map)

        # Record word in conversation history (with resolved intent for future filtering)
        self.history.add(word, resolved_intent=output.intent_id)

        return output

    def record_confirmed_intent(self, word: str, intent_id: str) -> None:
        """
        Passive feedback: call when subsequent user behavior confirms an intent.
        Updates the personal model without requiring explicit user ratings.
        """
        self.user_model.record_intent(word, intent_id)

    def new_session(self) -> None:
        """Reset conversation history at the start of a new session."""
        self.history.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _empty_output(self) -> IntentOutput:
        return IntentOutput(
            word_input="",
            resolved=False,
            intent_id="empty",
            intent_label="Empty Input",
            output_text="",
            lcd_text="",
            confidence=0.0,
            conflict_detected=False,
            emergency=False,
            fallback_mode="literal",
            top_n_candidates=[],
        )
