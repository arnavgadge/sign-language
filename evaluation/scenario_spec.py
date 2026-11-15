"""
evaluation/scenario_spec.py
-------------------------------
PHASE 9 -- INDEPENDENT GROUND-TRUTH SPECIFICATION.

This is the direct fix for Phase 8's core methodological flaw. In Phase
8, ground truth was effectively computed as "this word is a
context_trigger of intent X in the ontology, therefore ground_truth =
X" -- circular, because the engine reads the same trigger lists to make
its own decision.

Here, each ScenarioTemplate is authored as a plain-English MEANING
first (what a person is actually trying to communicate), with a
vocabulary list chosen to express that meaning -- and ONLY THEN checked
against the ontology (via `verify_representable`) to confirm the engine
CAN represent this scenario at all (i.e. the target word has an
ontology entry with that intent_id among its candidates). The ontology
is used to check representability, never to assign the label. The
label is fixed at authoring time, in this file, independent of any
ontology.lookup() call.

HONEST LIMITATION (still true after this fix): I designed both this
specification and the ontology's trigger lists. There is no second,
unaffiliated annotator and no real human communication data behind
either. This narrows the circularity Phase 8 had (label is no longer
MECHANICALLY derived from live trigger lookups) but does not eliminate
single-author bias. A real user study remains the only way to close
that gap -- stated plainly in the Phase 9 report, not hidden.
"""
from dataclasses import dataclass
from typing import List

from intent.ontology import IntentOntology

# Confirmed (by direct grep of data/intent_ontology.json) to appear in
# ZERO context_triggers/context_conflicts anywhere in the ontology --
# genuinely neutral filler, cannot bias any candidate's score.
SAFE_DISTRACTORS = ["OK", "UM", "THEN", "SO", "MAYBE", "ALSO", "JUST"]


@dataclass(frozen=True)
class ScenarioTemplate:
    template_id: str
    target_word: str
    intent_id: str              # the INDEPENDENTLY declared ground truth
    meaning: str                 # plain-English description, authored first
    core_vocabulary: List[str]   # words expressing this meaning


TEMPLATES: List[ScenarioTemplate] = [
    ScenarioTemplate("SC01", "HELP", "medical_assistance",
                      "Urgent need for medical attention due to a physical symptom.",
                      ["CHEST", "PAIN", "BLOOD", "FALL", "DIZZY", "BREATHE", "HURT", "SICK"]),
    ScenarioTemplate("SC02", "HELP", "task_assistance",
                      "Need for help completing a work task or project.",
                      ["WORK", "TASK", "PROJECT", "DEADLINE", "FIX", "MAKE", "SHOW"]),
    ScenarioTemplate("SC03", "HELP", "emergency_support",
                      "Immediate danger, need urgent non-medical help.",
                      ["DANGER", "FIRE", "URGENT", "QUICK", "POLICE"]),
    ScenarioTemplate("SC04", "HELP", "information_request",
                      "Need an explanation or piece of information.",
                      ["WHERE", "WHAT", "EXPLAIN", "TELL", "KNOW"]),
    ScenarioTemplate("SC05", "PAIN", "medical_symptom",
                      "Reporting a physical pain symptom.",
                      ["CHEST", "BREATHE", "DIZZY", "FALL", "BLOOD"]),
    ScenarioTemplate("SC06", "PAIN", "emotional_distress",
                      "Expressing emotional or psychological pain.",
                      ["SAD", "CRY", "HEART", "MISS", "ALONE", "UPSET"]),
    ScenarioTemplate("SC07", "PAIN", "location_pain",
                      "Indicating or asking about the location of pain.",
                      ["WHERE", "POINT", "HERE", "BACK", "HEAD", "LEG", "ARM"]),
    ScenarioTemplate("SC08", "DOCTOR", "medical_care_request",
                      "Non-urgent request to see a doctor because of illness.",
                      ["SICK", "PAIN", "HURT", "MEDICINE"]),
    ScenarioTemplate("SC09", "DOCTOR", "emergency_medical",
                      "Urgent need for a doctor right now.",
                      ["URGENT", "QUICK", "NOW", "CHEST", "BREATHE"]),
    ScenarioTemplate("SC10", "DOCTOR", "appointment_inquiry",
                      "Routine scheduling question about a doctor's appointment.",
                      ["APPOINTMENT", "WHEN", "TIME", "SCHEDULE"]),
    ScenarioTemplate("SC11", "HOSPITAL", "emergency_transport",
                      "Needs to get to the hospital urgently.",
                      ["URGENT", "AMBULANCE", "PAIN", "QUICK"]),
    ScenarioTemplate("SC12", "HOSPITAL", "appointment_destination",
                      "Has a scheduled, non-urgent hospital appointment.",
                      ["APPOINTMENT", "SCHEDULE", "TOMORROW", "TODAY"]),
    ScenarioTemplate("SC13", "HOSPITAL", "direction_to_hospital",
                      "Asking for directions to the hospital.",
                      ["WHERE", "WHICH", "GO"]),
    ScenarioTemplate("SC14", "SICK", "illness_statement",
                      "General statement of feeling unwell.",
                      ["PAIN", "HURT", "DOCTOR", "MEDICINE"]),
    ScenarioTemplate("SC15", "SICK", "nausea_complaint",
                      "Feeling nauseous, likely food-related.",
                      ["STOMACH", "FOOD", "EAT", "DIZZY"]),
    ScenarioTemplate("SC16", "SICK", "severe_illness",
                      "Seriously unwell, needs urgent attention.",
                      ["VERY", "CHEST", "BREATHE", "DIZZY", "FALL"]),
    ScenarioTemplate("SC17", "HURT", "pain_complaint",
                      "Simple statement of being in pain.",
                      ["OW", "OUCH", "BODY"]),
    ScenarioTemplate("SC18", "HURT", "injury_emergency",
                      "Serious injury, needs immediate help.",
                      ["FALL", "BLOOD", "QUICK", "URGENT", "SERIOUS"]),
    ScenarioTemplate("SC19", "HURT", "emotional_hurt",
                      "Emotionally hurt, feelings-related.",
                      ["FEELINGS", "SAD", "CRY", "UPSET", "HEART"]),
    ScenarioTemplate("SC20", "STOP", "halt_action",
                      "Wants an ongoing activity to stop -- not medical.",
                      ["PLEASE", "NOW", "THAT", "DO"]),
    ScenarioTemplate("SC21", "STOP", "pain_stop",
                      "Wants something to stop BECAUSE it is causing pain.",
                      ["PAIN", "HURT", "OW", "OUCH"]),
    ScenarioTemplate("SC22", "STOP", "end_conversation",
                      "Wants to end the conversation politely.",
                      ["BYE", "GOODBYE", "FINISHED", "DONE", "THANK"]),
    ScenarioTemplate("SC23", "WATER", "thirst_request",
                      "Thirsty, wants water to drink.",
                      ["DRINK", "THIRSTY", "HOT"]),
    ScenarioTemplate("SC24", "WATER", "hygiene_request",
                      "Wants water to wash or clean.",
                      ["CLEAN", "WASH", "BATH", "DIRTY"]),
    ScenarioTemplate("SC25", "FOOD", "hunger_request",
                      "Hungry, wants food.",
                      ["HUNGRY", "EAT", "MEAL", "LUNCH"]),
    ScenarioTemplate("SC26", "FOOD", "dietary_restriction",
                      "Cannot eat something due to allergy/restriction.",
                      ["ALLERGY", "SICK", "MEDICINE", "CANNOT"]),
    ScenarioTemplate("SC27", "BATHROOM", "restroom_need",
                      "Urgently needs the restroom.",
                      ["URGENT", "NOW", "QUICK", "HURRY"]),
    ScenarioTemplate("SC28", "BATHROOM", "privacy_request",
                      "Wants privacy, not urgent.",
                      ["ALONE", "PRIVATE", "PLEASE", "ROOM"]),
    ScenarioTemplate("SC29", "GO", "destination_request",
                      "Wants to know where to go.",
                      ["WHERE", "HOSPITAL", "HOME"]),
    ScenarioTemplate("SC30", "GO", "urgency_go",
                      "Needs to leave right now, urgently.",
                      ["QUICK", "NOW", "HURRY", "FAST", "URGENT"]),
    ScenarioTemplate("SC31", "WAIT", "patience_request",
                      "Politely asking for a short wait.",
                      ["PLEASE", "MOMENT", "SECOND", "SOON"]),
    ScenarioTemplate("SC32", "WAIT", "pause_activity",
                      "Wants an activity paused, not ended.",
                      ["STOP", "PAUSE", "HOLD"]),
]

# Intents NOT represented by a template, and why -- documented, not hidden.
UNCOVERED_INTENTS_NOTE = (
    "YES/NO/SORRY/MORE/FINISHED/HOME words, and a handful of their "
    "candidates (e.g. 'confirmation', 'negative_response', 'apology'), "
    "were not templated: their meanings are too context-free / too short "
    "to construct an independently distinguishable multi-word scenario "
    "for (e.g. 'YES' meaning 'confirmation' has no natural preceding "
    "context that isn't circular with the ontology's own YES-adjacent "
    "candidates). 32 templates cover 32 of the ontology's 56 intents (57%)."
)


def verify_all_representable() -> dict:
    """Sanity check ONLY -- confirms each template's declared intent_id
    actually exists among the ontology's candidates for that word. Does
    NOT assign or change any label; if a template failed this check it
    would mean the scenario isn't representable by the current system,
    not that the label is wrong."""
    ontology = IntentOntology()
    report = []
    for t in TEMPLATES:
        candidates = {c.id for c in ontology.lookup(t.target_word)}
        report.append({
            "template_id": t.template_id, "target_word": t.target_word,
            "intent_id": t.intent_id, "representable": t.intent_id in candidates,
        })
    return {"all_representable": all(r["representable"] for r in report), "details": report}
