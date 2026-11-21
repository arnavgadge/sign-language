"""
baselines/classifier_features.py
------------------------------------
Feature dicts for the trained-classifier baseline, built ONLY from a
CaseInput (recognized_word, conversation_sequence, sequence_offsets_seconds,
personal_seed) plus the word's candidate list from IntentOntology.lookup().

Feature keys are strings; a DictVectorizer fitted on the training fold turns
them into columns, so the vocabulary comes from the training fold only and
unseen keys at test time are dropped.

  w=<WORD>                    target word one-hot
  ctx=<W>                     sum over earlier occurrences of W of exp(-offset / tau)
  wx=<WORD>|<W>               same value, keyed by (target word, earlier word)
  seq_len                     number of earlier words
  has_prev, last_offset       whether there is an earlier word; offset (s) of the most recent one
  hist_any                    any personal_seed entry for the target word
  p_count=<intent>            seeded count for that candidate intent (target word only)
  p_decay=<intent>            count * exp(-days_ago / 30)
  p_share=<intent>            share of the word's total seeded count
"""
import math
from typing import Dict

from .common import ONTOLOGY, CaseInput

PERSONAL_DECAY_DAYS = 30.0


def featurize(inp: CaseInput, tau: float) -> Dict[str, float]:
    word = inp.recognized_word.upper()
    earlier = inp.conversation_sequence[:-1]
    offsets = inp.sequence_offsets_seconds[:-1]

    f: Dict[str, float] = {f"w={word}": 1.0, "seq_len": float(len(earlier))}

    for w, off in zip(earlier, offsets):
        v = math.exp(-off / tau)
        f[f"ctx={w}"] = f.get(f"ctx={w}", 0.0) + v
        f[f"wx={word}|{w}"] = f.get(f"wx={word}|{w}", 0.0) + v

    f["has_prev"] = 1.0 if earlier else 0.0
    f["last_offset"] = float(offsets[-1]) if earlier else 0.0

    candidate_ids = {c.id for c in ONTOLOGY.lookup(word)}
    counts, decayed = {}, {}
    for seed_word, intent, count, days_ago in inp.personal_seed:
        if seed_word.upper() == word and intent in candidate_ids:
            counts[intent] = counts.get(intent, 0) + count
            decayed[intent] = decayed.get(intent, 0.0) + count * math.exp(-days_ago / PERSONAL_DECAY_DAYS)
    total = sum(counts.values())
    f["hist_any"] = 1.0 if total else 0.0
    for intent, cnt in counts.items():
        f[f"p_count={intent}"] = float(cnt)
        f[f"p_decay={intent}"] = decayed[intent]
        f[f"p_share={intent}"] = cnt / total
    return f


# Features that track HOW the synthetic dataset was generated rather than what
# the user meant (see baselines/artifact_ablation.py): sequence shape, and the
# neutral filler words the generator inserts everywhere except AMBIGUOUS cases.
SEQUENCE_SHAPE_FEATURES = ("seq_len", "has_prev", "last_offset")


def _filler_words():
    from evaluation.scenario_spec import SAFE_DISTRACTORS   # read-only constant
    return tuple(SAFE_DISTRACTORS)


def featurize_no_artifacts(inp: CaseInput, tau: float) -> Dict[str, float]:
    fillers = _filler_words()
    return {k: v for k, v in featurize(inp, tau).items()
            if k not in SEQUENCE_SHAPE_FEATURES
            and not (k.startswith("ctx=") and k[4:] in fillers)
            and not (k.startswith("wx=") and k.split("|", 1)[1] in fillers)}
