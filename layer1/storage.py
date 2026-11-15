"""
layer1/storage.py
-------------------
Append-only JSONL persistence for InteractionContext records. Chosen over
a database per the Phase 1 "avoid unnecessary technology" principle: at
this data volume (single-machine, low-concurrency research prototype),
one JSON object per line is sufficient, human-inspectable, and diffable.
"""
import os

from .config import INTERACTIONS_DIR
from .schema import InteractionContext


class InteractionLogger:
    """
    Usage
    -----
    logger = InteractionLogger(session_id="session-1")
    logger.append(ctx)                  # write
    records = logger.read_all()         # read back this session
    """

    def __init__(self, session_id: str, base_dir: str = INTERACTIONS_DIR):
        self.session_id = session_id
        self.base_dir = base_dir
        self.path = os.path.join(base_dir, f"{session_id}.jsonl")

    def append(self, ctx: InteractionContext) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(ctx.to_json() + "\n")

    def read_all(self) -> list:
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(InteractionContext.from_json(line))
        return records
