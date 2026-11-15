"""
evaluation/user_pool.py
---------------------------
PHASE 9 -- a persistent pool of simulated users (NOT one user per sample,
per the explicit Phase 9 requirement). Each user has a stable "bias" --
which intent they tend to mean for a given word -- used to seed
Personal Context history consistently across every case that references
them, so personalization is tested with real, reusable profiles rather
than one-off single-sample users.

This is a CONTROLLED SIMULATION of persistent user identity, not real
user data. Labelled as such throughout the Phase 9 report.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SimulatedUser:
    user_id: str
    tag: str                          # "medical_leaning" | "task_leaning" | "neutral" | "cold_start_probe"
    bias_intent_by_word: Dict[str, str] = field(default_factory=dict)


def _medical_leaning(n: int) -> List[SimulatedUser]:
    return [
        SimulatedUser(f"U{100+i:03d}", "medical_leaning", {
            "HELP": "medical_assistance", "PAIN": "medical_symptom",
            "DOCTOR": "medical_care_request", "SICK": "illness_statement",
            "HURT": "pain_complaint", "STOP": "pain_stop",
        }) for i in range(n)
    ]


def _task_leaning(n: int) -> List[SimulatedUser]:
    return [
        SimulatedUser(f"U{200+i:03d}", "task_leaning", {
            "HELP": "task_assistance", "PAIN": "location_pain",
            "DOCTOR": "appointment_inquiry", "STOP": "halt_action",
            "GO": "destination_request", "WAIT": "patience_request",
        }) for i in range(n)
    ]


def _neutral(n: int) -> List[SimulatedUser]:
    return [SimulatedUser(f"U{300+i:03d}", "neutral", {}) for i in range(n)]


def _cold_start_probes(n: int) -> List[SimulatedUser]:
    return [SimulatedUser(f"U{400+i:03d}", "cold_start_probe", {
        "HELP": "medical_assistance",
    }) for i in range(n)]


MEDICAL_LEANING = _medical_leaning(10)
TASK_LEANING = _task_leaning(10)
NEUTRAL = _neutral(12)
COLD_START_PROBES = _cold_start_probes(8)

USER_POOL: List[SimulatedUser] = MEDICAL_LEANING + TASK_LEANING + NEUTRAL + COLD_START_PROBES


def get_user(user_id: str) -> SimulatedUser:
    for u in USER_POOL:
        if u.user_id == user_id:
            return u
    raise KeyError(user_id)


def pool_summary() -> dict:
    tags = {}
    for u in USER_POOL:
        tags[u.tag] = tags.get(u.tag, 0) + 1
    return {"n_users": len(USER_POOL), "by_tag": tags}
