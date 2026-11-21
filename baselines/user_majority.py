"""
baselines/user_majority.py
------------------------------
Per-user majority baseline: count intents in THIS case's personal_seed for
the recognized word only (weighted by count; never other cases). Confidence =
the top intent's share of the seed count. With no seed for the word, fall back
to the base-prior ranking (confidence = base_probability) and flag it.

Seeded intents that are not candidates of the word are ignored. Count ties
and the unseeded remainder of the ranking follow base-prior order.
"""
from collections import Counter

from .base_prior import prior_ranking
from .common import CaseInput, Method


class UserMajority(Method):
    name = "user_majority"

    def __init__(self):
        self._used_fallback = False

    def predict(self, inp: CaseInput):
        word = inp.recognized_word.upper()
        prior = prior_ranking(word)
        prior_pos = {cid: i for i, (cid, _) in enumerate(prior)}

        counts = Counter()
        for seed_word, intent, count, _days_ago in inp.personal_seed:
            if seed_word.upper() == word and intent in prior_pos:
                counts[intent] += count
        total = sum(counts.values())

        self._used_fallback = total == 0
        if self._used_fallback:
            top, score = prior[0]
            return top, score, prior

        ranking = sorted(((cid, counts[cid] / total) for cid in prior_pos),
                         key=lambda cs: (-cs[1], prior_pos[cs[0]]))
        top, score = ranking[0]
        return top, score, ranking

    def extras(self):
        return {"used_fallback": self._used_fallback}
