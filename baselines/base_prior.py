"""
baselines/base_prior.py
---------------------------
Base-prior baseline: rank the word's ontology candidates by
base_probability; ties broken by priority_weight, then by order in
data/intent_ontology.json. Confidence = base_probability. Ignores all context.
"""
from .common import ONTOLOGY, CaseInput, Method


def prior_ranking(word: str):
    candidates = ONTOLOGY.lookup(word)
    order = sorted(enumerate(candidates),
                   key=lambda ic: (-ic[1].base_probability, -ic[1].priority_weight, ic[0]))
    return [(c.id, c.base_probability) for _, c in order]


class BasePrior(Method):
    name = "base_prior"

    def predict(self, inp: CaseInput):
        ranking = prior_ranking(inp.recognized_word)
        top, score = ranking[0]
        return top, score, ranking
