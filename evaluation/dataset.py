"""
evaluation/dataset.py
-------------------------
THE GROUND-TRUTH EVALUATION DATASET (Phase 8).

Methodology / how this was constructed
---------------------------------------
Every sample was hand-authored for this evaluation by grounding it in
the ACTUAL `data/intent_ontology.json` content: each preceding-context
word was chosen because it is already listed as a `context_triggers`
entry for the intended ground-truth candidate (or, for adversarial
samples, because it is a trigger for a DIFFERENT candidate than the
intended one, to test whether the engine gets misled). No sample uses a
word/intent that isn't already in the ontology.

This is disclosed as a LIMITATION, not hidden: because ground truth was
assigned using the same trigger relationships the engine itself reads,
this dataset primarily measures whether the engine's MECHANISM (window,
decay, reliability weighting, conflict detection, gating) correctly
implements the ontology's own designed semantics — not whether those
semantics match real, unconstrained human communication. A genuine user
study would be needed for the latter and does not exist yet.

Ground truth values are one of:
  - a specific intent_id (e.g. "medical_assistance") when the sequence
    should unambiguously point to one candidate,
  - "AMBIGUOUS" when evidence sources are DESIGNED to conflict and the
    correct system behavior is to decline to pick one,
  - "LITERAL" when there is deliberately NO usable context (cold start)
    and the correct system behavior is to fall back to the literal word.

`personal_seed` entries are (word, intent_id, count, days_ago) tuples —
how many times, and how long ago, to record that word/intent pair in
PersonalContextEngine BEFORE evaluating the sample. Empty list = no
personal history for that user (a fresh profile).
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    category: str
    user_id: str
    sequence: List[str]                       # preceding words, target word LAST
    ground_truth: str                          # intent_id | "AMBIGUOUS" | "LITERAL"
    personal_seed: List[Tuple[str, str, int, float]] = field(default_factory=list)
    notes: str = ""


DATASET: List[EvalSample] = [

    # ── medical_context: temporal evidence alone should resolve these ──────
    EvalSample("M01", "medical_context", "u_med_01", ["CHEST", "PAIN", "HELP"],
               "medical_assistance",
               notes="CHEST, PAIN are both context_triggers of medical_assistance for HELP."),
    EvalSample("M02", "medical_context", "u_med_02", ["BLOOD", "FALL", "HELP"],
               "medical_assistance",
               notes="BLOOD, FALL are both context_triggers of medical_assistance for HELP."),
    EvalSample("M03", "medical_context", "u_med_03", ["CHEST", "BREATHE", "DOCTOR"],
               "emergency_medical",
               notes="CHEST, BREATHE are context_triggers of emergency_medical for DOCTOR."),
    EvalSample("M04", "medical_context", "u_med_04", ["SICK", "PAIN", "DOCTOR"],
               "medical_care_request",
               notes="SICK, PAIN are context_triggers of medical_care_request for DOCTOR."),

    # ── task_context ─────────────────────────────────────────────────────
    EvalSample("T01", "task_context", "u_task_01", ["PROJECT", "DEADLINE", "HELP"],
               "task_assistance",
               notes="PROJECT, DEADLINE are context_triggers of task_assistance for HELP."),
    EvalSample("T02", "task_context", "u_task_02", ["WORK", "TASK", "HELP"],
               "task_assistance",
               notes="WORK, TASK are context_triggers of task_assistance for HELP."),
    EvalSample("T03", "task_context", "u_task_03", ["SHOW", "HOW", "HELP"],
               "task_assistance",
               notes="SHOW, HOW are context_triggers of task_assistance for HELP."),

    # ── emergency_context ────────────────────────────────────────────────
    EvalSample("E01", "emergency_context", "u_emg_01", ["DANGER", "URGENT", "HELP"],
               "emergency_support",
               notes="DANGER, URGENT are context_triggers of emergency_support for HELP."),
    EvalSample("E02", "emergency_context", "u_emg_02", ["FIRE", "QUICK", "HELP"],
               "emergency_support",
               notes="FIRE, QUICK are context_triggers of emergency_support for HELP."),

    # ── information_context ──────────────────────────────────────────────
    EvalSample("I01", "information_context", "u_info_01", ["WHERE", "EXPLAIN", "HELP"],
               "information_request",
               notes="WHERE, EXPLAIN are context_triggers of information_request for HELP."),

    # ── cold_start: no usable context at all -> must NOT guess ─────────────
    EvalSample("C01", "cold_start", "u_cold_01", ["HELP"], "LITERAL",
               notes="Single word, fresh user, no preceding context, no personal history."),
    EvalSample("C02", "cold_start", "u_cold_02", ["PAIN"], "LITERAL",
               notes="Single word, fresh user, no preceding context, no personal history."),
    EvalSample("C03", "cold_start", "u_cold_03", ["DOCTOR"], "LITERAL",
               notes="Single word, fresh user, no preceding context, no personal history."),
    EvalSample("C04", "cold_start", "u_cold_04", ["HELP", "HELP", "HELP"], "LITERAL",
               notes="Repetition alone does not identify WHICH intent -> still no usable evidence."),

    # ── personalized: same word, different long-term histories ─────────────
    EvalSample("P01", "personalized", "u_pers_A", ["HELP"], "medical_assistance",
               personal_seed=[("HELP", "medical_assistance", 18, 5),
                               ("HELP", "emergency_support", 3, 5),
                               ("HELP", "task_assistance", 1, 5)],
               notes="User A's long-term history strongly favors medical_assistance for HELP."),
    EvalSample("P02", "personalized", "u_pers_B", ["HELP"], "task_assistance",
               personal_seed=[("HELP", "task_assistance", 15, 5),
                               ("HELP", "medical_assistance", 2, 5)],
               notes="User B's long-term history strongly favors task_assistance for HELP."),
    EvalSample("P03", "personalized", "u_pers_C", ["PAIN"], "medical_symptom",
               personal_seed=[("PAIN", "medical_symptom", 12, 5)],
               notes="User C's long-term history strongly favors medical_symptom for PAIN."),
    EvalSample("P04", "personalized", "u_pers_D", ["PAIN"], "emotional_distress",
               personal_seed=[("PAIN", "emotional_distress", 12, 5)],
               notes="User D's long-term history strongly favors emotional_distress for PAIN."),

    # ── conflict: temporal and personal DELIBERATELY disagree ──────────────
    EvalSample("X01", "conflict", "u_conf_01", ["CHEST", "PAIN", "HELP"], "AMBIGUOUS",
               personal_seed=[("HELP", "task_assistance", 20, 10)],
               notes="Temporal points to medical_assistance; personal history points to "
                     "task_assistance just as strongly. Correct behavior: flag ambiguity/conflict, "
                     "not silently pick one."),
    EvalSample("X02", "conflict", "u_conf_02", ["PROJECT", "DEADLINE", "HELP"], "AMBIGUOUS",
               personal_seed=[("HELP", "medical_assistance", 20, 10)],
               notes="Temporal points to task_assistance; personal history points to "
                     "medical_assistance just as strongly."),

    # ── adversarial: keyword overlap that could mislead the engine ─────────
    EvalSample("A01", "adversarial", "u_adv_01", ["TASK", "DO", "HELP", "STOP"],
               "halt_action",
               notes="Intended meaning: 'stop doing the task.' But HELP immediately precedes "
                     "STOP, and HELP is literally a context_trigger of STOP's 'pain_stop' "
                     "candidate — the engine may be misled toward a medical reading. Genuine "
                     "adversarial test; result reported as-is, not adjusted."),
    EvalSample("A02", "adversarial", "u_adv_02", ["APPOINTMENT", "SCHEDULE", "DOCTOR"],
               "appointment_inquiry",
               notes="Intended meaning: a routine scheduling question, not urgent care. "
                     "APPOINTMENT/SCHEDULE are appointment_inquiry's own triggers, and are "
                     "NOT triggers of emergency_medical, so this should resolve correctly — "
                     "included as a check that superficially medical words (DOCTOR) don't "
                     "always default to an emergency reading."),

    # ── everyday / non-medical words, broader ontology coverage ───────────
    EvalSample("D01", "daily_needs", "u_day_01", ["HUNGRY", "EAT", "FOOD"],
               "hunger_request",
               notes="HUNGRY, EAT are context_triggers of hunger_request for FOOD."),
    EvalSample("D02", "daily_needs", "u_day_02", ["ALLERGY", "SICK", "FOOD"],
               "dietary_restriction",
               notes="ALLERGY, SICK are context_triggers of dietary_restriction for FOOD."),
    EvalSample("D03", "daily_needs", "u_day_03", ["THIRSTY", "HOT", "WATER"],
               "thirst_request",
               notes="THIRSTY, HOT are context_triggers of thirst_request for WATER."),
    EvalSample("D04", "daily_needs", "u_day_04", ["URGENT", "HURRY", "BATHROOM"],
               "restroom_need",
               notes="URGENT, HURRY are context_triggers of restroom_need for BATHROOM."),
    EvalSample("D05", "daily_needs", "u_day_05", ["ALONE", "PRIVATE", "BATHROOM"],
               "privacy_request",
               notes="ALONE, PRIVATE are context_triggers of privacy_request for BATHROOM."),
    EvalSample("D06", "daily_needs", "u_day_06", ["APPOINTMENT", "SCHEDULE", "HOSPITAL"],
               "appointment_destination",
               notes="APPOINTMENT, SCHEDULE are context_triggers of appointment_destination "
                     "for HOSPITAL, and are explicit context_conflicts of the emergency "
                     "reading — a good non-emergency-defaulting check."),
    EvalSample("D07", "daily_needs", "u_day_07", ["AMBULANCE", "URGENT", "HOSPITAL"],
               "emergency_transport",
               notes="AMBULANCE, URGENT are context_triggers of emergency_transport for HOSPITAL."),
    EvalSample("D08", "daily_needs", "u_day_08", ["MOMENT", "PLEASE", "WAIT"],
               "patience_request",
               notes="MOMENT, PLEASE are context_triggers of patience_request for WAIT."),
    EvalSample("D09", "daily_needs", "u_day_09", ["BYE", "THANK", "STOP"],
               "end_conversation",
               notes="BYE, THANK are context_triggers of end_conversation for STOP."),
]


def load_dataset() -> List[EvalSample]:
    return list(DATASET)


def dataset_summary() -> dict:
    ds = load_dataset()
    words = set()
    users = set()
    intents = set()
    categories = {}
    for s in ds:
        words.update(s.sequence)
        users.add(s.user_id)
        intents.add(s.ground_truth)
        categories[s.category] = categories.get(s.category, 0) + 1
    return {
        "n_samples": len(ds),
        "n_distinct_words": len(words),
        "n_distinct_users": len(users),
        "n_distinct_ground_truth_classes": len(intents),
        "ground_truth_classes": sorted(intents),
        "categories": categories,
    }
