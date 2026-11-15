"""
Intent Ontology — maps recognized words to their possible communicative intents.

The ontology is a graph: each word node connects to a set of IntentCandidate
nodes.  Stored as JSON and loaded at runtime.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class IntentCandidate:
    id: str
    label: str
    output_text: str        # Natural-language output shown on screen
    lcd_text: str           # ≤16-char version for Arduino LCD
    emergency: bool         # True → eligible for Emergency Override pathway
    priority_weight: float  # Ontology-level prior boost (default 1.0)
    context_triggers: List[str]    # Recent words that raise this intent's probability
    context_conflicts: List[str]   # Recent words that lower this intent's probability
    location_priors: dict          # location_tag -> probability adjustment
    time_priors: dict              # time_period -> probability adjustment
    base_probability: float        # Uniform prior within this word's candidate set


class IntentOntology:
    def __init__(self, ontology_path: str = "data/intent_ontology.json"):
        self.path = ontology_path
        self._data = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": "1.0", "words": {}}
        with open(self.path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def lookup(self, word: str) -> List[IntentCandidate]:
        word = word.upper().strip()
        entry = self._data.get("words", {}).get(word)
        if not entry:
            return [self._literal_candidate(word)]

        n = max(1, len(entry["intents"]))
        return [
            IntentCandidate(
                id=i["id"],
                label=i["label"],
                output_text=i.get("output_text", word),
                lcd_text=i.get("lcd_text", word[:16]),
                emergency=i.get("emergency", False),
                priority_weight=i.get("priority_weight", 1.0),
                context_triggers=[t.upper() for t in i.get("context_triggers", [])],
                context_conflicts=[c.upper() for c in i.get("context_conflicts", [])],
                location_priors=i.get("location_priors", {}),
                time_priors=i.get("time_priors", {}),
                base_probability=i.get("base_probability", 1.0 / n),
            )
            for i in entry["intents"]
        ]

    def has_word(self, word: str) -> bool:
        return word.upper() in self._data.get("words", {})

    def _literal_candidate(self, word: str) -> IntentCandidate:
        return IntentCandidate(
            id="literal_translation",
            label=f"Literal: {word}",
            output_text=word,
            lcd_text=word[:16],
            emergency=False,
            priority_weight=1.0,
            context_triggers=[],
            context_conflicts=[],
            location_priors={},
            time_priors={},
            base_probability=1.0,
        )
