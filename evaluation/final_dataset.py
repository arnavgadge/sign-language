"""
evaluation/final_dataset.py
-------------------------------
PHASE 9 -- the ~500-case final evaluation dataset generator.

Ground truth for every case traces back to evaluation/scenario_spec.py's
independently-authored ScenarioTemplates or evaluation/user_pool.py's
independently-declared user "personalities" -- NEVER to a live
ontology.lookup() trigger check at generation time (that was Phase 8's
circularity problem). The ontology is only consulted afterward, by
evaluation/scenario_spec.verify_all_representable(), as a
representability sanity check.

Deterministic: everything is driven by random.Random(SEED), so the same
500 cases are produced every run -- no manual editing, no re-rolling
until numbers look good.

Each case carries explicit per-word timing (`sequence_offsets_seconds`)
so window-boundary behavior (Category B, some of Category I) is actually
testable -- a word placed at offset > 120s genuinely falls outside
Layer 3's temporal window when the evaluation harness records it at
that timestamp, not just "implied" by category label.
"""
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from .scenario_spec import TEMPLATES, SAFE_DISTRACTORS, ScenarioTemplate
from .user_pool import MEDICAL_LEANING, TASK_LEANING, NEUTRAL, COLD_START_PROBES

SEED = 20240521
NOW = 1_700_000_000.0
DAY = 86400.0
WINDOW_SECONDS = 120.0

AMBIGUOUS = "AMBIGUOUS"
LITERAL = "LITERAL"
RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class FinalEvalCase:
    sample_id: str
    scenario_category: str
    recognized_word: str
    conversation_sequence: List[str]           # preceding words + target word LAST
    sequence_offsets_seconds: List[float]        # seconds before "now" each word occurred; last entry is 0.0
    user_id: str
    ground_truth: str                            # intent_id | AMBIGUOUS | LITERAL
    expected_status: str                          # RESOLVED | AMBIGUOUS | LITERAL
    personal_seed: List[Tuple[str, str, int, float]] = field(default_factory=list)
    notes: str = ""


def _templates_for_word(word: str) -> List[ScenarioTemplate]:
    return [t for t in TEMPLATES if t.target_word == word]


def _default_offsets(n_preceding: int, spacing: float = 10.0) -> List[float]:
    """n_preceding words spaced `spacing` seconds apart, most recent closest
    to the target word, target word itself at offset 0.0."""
    offsets = [spacing * (n_preceding - i) for i in range(n_preceding)]
    return offsets + [0.0]


def _build_sequence(rng: random.Random, template: ScenarioTemplate,
                     n_relevant: int, n_distractors: int) -> List[str]:
    relevant = rng.sample(template.core_vocabulary, min(n_relevant, len(template.core_vocabulary)))
    distractors = [rng.choice(SAFE_DISTRACTORS) for _ in range(n_distractors)]
    body = relevant + distractors
    rng.shuffle(body)
    if relevant and body[-1] not in relevant:
        body.remove(relevant[0])
        body.append(relevant[0])
    return body + [template.target_word]


def _gen_clear_context(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category A -- clear contextual intent."""
    cases = []
    for i in range(n):
        template = rng.choice(TEMPLATES)
        user = rng.choice(NEUTRAL)
        seq = _build_sequence(rng, template, n_relevant=rng.randint(2, 3), n_distractors=rng.randint(0, 2))
        cases.append(FinalEvalCase(
            sample_id=f"A{start_id + i:04d}", scenario_category="clear_context",
            recognized_word=template.target_word, conversation_sequence=seq,
            sequence_offsets_seconds=_default_offsets(len(seq) - 1), user_id=user.user_id,
            ground_truth=template.intent_id, expected_status=RESOLVED,
            notes=f"template={template.template_id}: {template.meaning}",
        ))
    return cases


def _gen_temporal_dependent(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category B -- temporal-context-dependent, including REAL window-boundary cases
    (relevant words placed at an offset > WINDOW_SECONDS, not just implied)."""
    cases = []
    n_outside_window = max(1, n // 5)   # ~20% deliberately push relevant context outside 120s
    for i in range(n):
        template = rng.choice(TEMPLATES)
        user = rng.choice(NEUTRAL)
        outside_window = i < n_outside_window
        n_relevant = rng.randint(1, 4)
        relevant = rng.sample(template.core_vocabulary, min(n_relevant, len(template.core_vocabulary)))
        distractors = [rng.choice(SAFE_DISTRACTORS) for _ in range(rng.randint(0, 3))]
        preceding = relevant + distractors
        rng.shuffle(preceding)
        seq = preceding + [template.target_word]

        if outside_window:
            # All preceding words placed well beyond the 120s window.
            spacing = rng.uniform(15, 25)
            base = WINDOW_SECONDS + 30.0   # guarantees every preceding word is outside the window
            offsets = [base + spacing * (len(preceding) - j) for j in range(len(preceding))] + [0.0]
            gt, status = LITERAL, LITERAL
            note = (f"ALL {len(preceding)} preceding word(s) placed {offsets[0]:.0f}s-{offsets[-2]:.0f}s "
                    f"before target -- genuinely outside the {WINDOW_SECONDS:.0f}s window")
        else:
            offsets = _default_offsets(len(preceding), spacing=rng.uniform(5, 15))
            gt, status = template.intent_id, RESOLVED
            note = f"relevant context inside window, template={template.template_id}"

        cases.append(FinalEvalCase(
            sample_id=f"B{start_id + i:04d}", scenario_category="temporal_dependent",
            recognized_word=template.target_word, conversation_sequence=seq,
            sequence_offsets_seconds=offsets, user_id=user.user_id,
            ground_truth=gt, expected_status=status, notes=note,
        ))
    return cases


def _gen_personalization_dependent(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category C -- relies on persistent user bias, no temporal help."""
    cases = []
    biased_users = MEDICAL_LEANING + TASK_LEANING
    for i in range(n):
        user = rng.choice(biased_users)
        word = rng.choice(list(user.bias_intent_by_word.keys()))
        intent = user.bias_intent_by_word[word]
        count = rng.choice([15, 18, 20, 25])
        days_ago = rng.choice([2, 5, 10])
        cases.append(FinalEvalCase(
            sample_id=f"C{start_id + i:04d}", scenario_category="personalization_dependent",
            recognized_word=word, conversation_sequence=[word], sequence_offsets_seconds=[0.0],
            user_id=user.user_id, ground_truth=intent, expected_status=RESOLVED,
            personal_seed=[(word, intent, count, days_ago)],
            notes=f"user personality={user.tag}, no temporal context -- tests personal signal alone",
        ))
    return cases


def _gen_combined(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category D -- temporal AND personal reinforce the SAME intent."""
    cases = []
    biased_users = MEDICAL_LEANING + TASK_LEANING
    for i in range(n):
        user = rng.choice(biased_users)
        word = rng.choice(list(user.bias_intent_by_word.keys()))
        intent = user.bias_intent_by_word[word]
        matching_templates = [t for t in TEMPLATES if t.target_word == word and t.intent_id == intent]
        if not matching_templates:
            continue
        template = matching_templates[0]
        seq = _build_sequence(rng, template, n_relevant=rng.randint(1, 3), n_distractors=rng.randint(0, 2))
        count = rng.choice([10, 15, 20])
        cases.append(FinalEvalCase(
            sample_id=f"D{start_id + i:04d}", scenario_category="combined_temporal_personal",
            recognized_word=word, conversation_sequence=seq,
            sequence_offsets_seconds=_default_offsets(len(seq) - 1), user_id=user.user_id,
            ground_truth=intent, expected_status=RESOLVED,
            personal_seed=[(word, intent, count, 5)],
            notes=f"temporal and personal both support {intent}",
        ))
    return cases


def _gen_ambiguous(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category E -- genuinely comparable support for 2+ candidates."""
    cases = []
    words_with_multiple_templates = {}
    for t in TEMPLATES:
        words_with_multiple_templates.setdefault(t.target_word, []).append(t)
    eligible_words = [w for w, ts in words_with_multiple_templates.items() if len(ts) >= 2]

    for i in range(n):
        word = rng.choice(eligible_words)
        t1, t2 = rng.sample(words_with_multiple_templates[word], 2)
        user = rng.choice(NEUTRAL)
        v1 = rng.sample(t1.core_vocabulary, min(2, len(t1.core_vocabulary)))
        v2 = rng.sample(t2.core_vocabulary, min(2, len(t2.core_vocabulary)))
        preceding = v1 + v2
        rng.shuffle(preceding)
        seq = preceding + [word]
        cases.append(FinalEvalCase(
            sample_id=f"E{start_id + i:04d}", scenario_category="genuinely_ambiguous",
            recognized_word=word, conversation_sequence=seq,
            sequence_offsets_seconds=_default_offsets(len(preceding)), user_id=user.user_id,
            ground_truth=AMBIGUOUS, expected_status=AMBIGUOUS,
            notes=f"balanced vocabulary from {t1.template_id} ({t1.intent_id}) "
                  f"and {t2.template_id} ({t2.intent_id})",
        ))
    return cases


def _gen_conflict(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category F -- temporal supports A, personal supports B. Vary relative strength."""
    cases = []
    words_with_multiple_templates = {}
    for t in TEMPLATES:
        words_with_multiple_templates.setdefault(t.target_word, []).append(t)
    eligible_words = [w for w, ts in words_with_multiple_templates.items() if len(ts) >= 2]

    strengths = (["temporal_dominant"] * (n // 3) + ["personal_dominant"] * (n // 3)
                 + ["balanced"] * (n - 2 * (n // 3)))
    rng.shuffle(strengths)

    for i, strength in enumerate(strengths):
        word = rng.choice(eligible_words)
        t_temporal, t_personal = rng.sample(words_with_multiple_templates[word], 2)
        user = rng.choice(NEUTRAL)

        if strength == "temporal_dominant":
            seq = _build_sequence(rng, t_temporal, n_relevant=3, n_distractors=0)
            offsets = _default_offsets(len(seq) - 1, spacing=8.0)
            personal_seed = [(word, t_personal.intent_id, 3, 20)]   # sparse, old
            gt, status = t_temporal.intent_id, RESOLVED
        elif strength == "personal_dominant":
            filler = rng.choice(SAFE_DISTRACTORS)
            seq = [filler, word]
            offsets = _default_offsets(1, spacing=60.0)   # distant, weak temporal signal
            personal_seed = [(word, t_personal.intent_id, 25, 3)]   # extensive, fresh
            gt, status = t_personal.intent_id, RESOLVED
        else:
            seq = _build_sequence(rng, t_temporal, n_relevant=2, n_distractors=0)
            offsets = _default_offsets(len(seq) - 1, spacing=8.0)
            personal_seed = [(word, t_personal.intent_id, 15, 5)]
            gt, status = AMBIGUOUS, AMBIGUOUS

        cases.append(FinalEvalCase(
            sample_id=f"F{start_id + i:04d}", scenario_category="conflict",
            recognized_word=word, conversation_sequence=seq, sequence_offsets_seconds=offsets,
            user_id=user.user_id, ground_truth=gt, expected_status=status, personal_seed=personal_seed,
            notes=f"strength={strength}: temporal->{t_temporal.intent_id}, personal->{t_personal.intent_id}",
        ))
    return cases


def _gen_cold_start(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category G -- longitudinal simulation of personal-history accumulation."""
    levels = [0, 1, 2, 5, 10, 20, 30]
    # Below this many observations, the CORRECT expected behavior is to
    # decline confident resolution -- sparse history should not produce
    # false confidence. This is an independently-stated safety principle,
    # not derived from the ontology.
    CONFIDENT_THRESHOLD = 10

    cases = []
    words = ["HELP", "PAIN", "DOCTOR"]
    i = 0
    for probe_index, probe in enumerate(COLD_START_PROBES):
        for level in levels:
            word = words[i % len(words)]
            matching = _templates_for_word(word)
            template = matching[probe_index % len(matching)]
            personal_seed = [(word, template.intent_id, level, 5)] if level > 0 else []

            if level == 0:
                gt, status = LITERAL, LITERAL
            elif level < CONFIDENT_THRESHOLD:
                gt, status = AMBIGUOUS, AMBIGUOUS
            else:
                gt, status = template.intent_id, RESOLVED

            cases.append(FinalEvalCase(
                sample_id=f"G{start_id + i:04d}", scenario_category="cold_start",
                recognized_word=word, conversation_sequence=[word], sequence_offsets_seconds=[0.0],
                user_id=probe.user_id, ground_truth=gt, expected_status=status, personal_seed=personal_seed,
                notes=f"observation_count={level}, no temporal context",
            ))
            i += 1
            if len(cases) >= n:
                return cases
    return cases


def _gen_literal(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category H -- genuinely no usable context."""
    cases = []
    words = sorted({t.target_word for t in TEMPLATES})
    for i in range(n):
        word = words[i % len(words)]
        user = rng.choice(NEUTRAL)
        cases.append(FinalEvalCase(
            sample_id=f"H{start_id + i:04d}", scenario_category="literal_fallback",
            recognized_word=word, conversation_sequence=[word], sequence_offsets_seconds=[0.0],
            user_id=user.user_id, ground_truth=LITERAL, expected_status=LITERAL,
            notes="single word, fresh user, zero context",
        ))
    return cases


# Confirmed adversarial (misleading-trigger) relationships, verified by
# direct inspection of data/intent_ontology.json:
#   HELP is a context_trigger of STOP's 'pain_stop' candidate.
#   HELP is a context_trigger of DOCTOR's 'medical_care_request' candidate.
_ADVERSARIAL_PAIRS = [
    ("halt_action", ["TASK", "DO"], "HELP", "STOP"),
    ("appointment_inquiry", ["APPOINTMENT", "SCHEDULE"], "HELP", "DOCTOR"),
]


def _gen_adversarial(rng: random.Random, n: int, start_id: int) -> List[FinalEvalCase]:
    """Category I -- systematically varied misleading-trigger cases."""
    cases = []
    variations = [
        {"desc": "misleading trigger immediately before target", "gap_before": 0, "n_irrelevant": 0, "repeat": 1, "far": False},
        {"desc": "misleading trigger 2 positions before target", "gap_before": 2, "n_irrelevant": 0, "repeat": 1, "far": False},
        {"desc": "misleading trigger 4 positions before target", "gap_before": 4, "n_irrelevant": 0, "repeat": 1, "far": False},
        {"desc": "misleading trigger with irrelevant words interleaved", "gap_before": 1, "n_irrelevant": 3, "repeat": 1, "far": False},
        {"desc": "misleading trigger repeated twice", "gap_before": 0, "n_irrelevant": 0, "repeat": 2, "far": False},
        {"desc": "misleading trigger placed OUTSIDE the temporal window", "gap_before": 0, "n_irrelevant": 0, "repeat": 1, "far": True},
    ]
    i = 0
    while len(cases) < n:
        intended_intent, intended_vocab, misleading_word, target_word = \
            _ADVERSARIAL_PAIRS[i % len(_ADVERSARIAL_PAIRS)]
        variation = variations[i % len(variations)]
        user = rng.choice(NEUTRAL)

        preceding = list(intended_vocab)
        for _ in range(variation["gap_before"]):
            preceding.append(rng.choice(SAFE_DISTRACTORS))
        for _ in range(variation["n_irrelevant"]):
            preceding.append(rng.choice(SAFE_DISTRACTORS))
        for _ in range(variation["repeat"]):
            preceding.append(misleading_word)
        seq = preceding + [target_word]

        if variation["far"]:
            # Misleading word pushed outside the window -> should NOT mislead.
            offsets = [WINDOW_SECONDS + 40.0 if w == misleading_word else 10.0 for w in preceding] + [0.0]
        else:
            offsets = _default_offsets(len(preceding), spacing=8.0)

        cases.append(FinalEvalCase(
            sample_id=f"I{start_id + i:04d}", scenario_category="adversarial",
            recognized_word=target_word, conversation_sequence=seq, sequence_offsets_seconds=offsets,
            user_id=user.user_id, ground_truth=intended_intent, expected_status=RESOLVED,
            notes=f"{variation['desc']}; intended={intended_intent}, misleading_word={misleading_word}",
        ))
        i += 1
    return cases[:n]


def generate_final_dataset() -> List[FinalEvalCase]:
    """Deterministic generator -- same SEED always produces the same 500 cases."""
    rng = random.Random(SEED)

    plan = [
        ("clear_context", _gen_clear_context, 120),
        ("temporal_dependent", _gen_temporal_dependent, 80),
        ("personalization_dependent", _gen_personalization_dependent, 80),
        ("combined_temporal_personal", _gen_combined, 70),
        ("genuinely_ambiguous", _gen_ambiguous, 60),
        ("conflict", _gen_conflict, 40),
        ("cold_start", _gen_cold_start, 20),
        ("literal_fallback", _gen_literal, 20),
        ("adversarial", _gen_adversarial, 10),
    ]

    all_cases: List[FinalEvalCase] = []
    for category, fn, count in plan:
        cases = fn(rng, count, start_id=1)
        all_cases.extend(cases)

    return all_cases


def dataset_composition(cases: List[FinalEvalCase]) -> dict:
    by_category = {}
    by_class = {}
    words = set()
    users = set()
    for c in cases:
        by_category[c.scenario_category] = by_category.get(c.scenario_category, 0) + 1
        by_class[c.ground_truth] = by_class.get(c.ground_truth, 0) + 1
        words.add(c.recognized_word)
        users.add(c.user_id)
    return {
        "n_total": len(cases),
        "by_category": by_category,
        "n_distinct_words": len(words),
        "n_distinct_users": len(users),
        "n_ground_truth_classes": len(by_class),
        "samples_per_class": by_class,
        "min_class_count": min(by_class.values()) if by_class else 0,
        "max_class_count": max(by_class.values()) if by_class else 0,
    }
