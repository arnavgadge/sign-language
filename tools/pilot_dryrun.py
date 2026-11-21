"""
tools/pilot_dryrun.py -- dry run of the pilot logging path with SIMULATED
recognized words (no webcam, no recognizer). Calls the real IntentEngine
with in-memory databases, writes a temp CSV, prints it, then deletes it.
Run:  python tools/pilot_dryrun.py
"""
import os
import sys
import tempfile
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from resolution.engine import IntentEngine
from utils.pilot_logger import PilotSession

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(prefix="pilot_dryrun_")

# in-memory DBs so the real data/*.db files are NOT touched
engine = IntentEngine(personal_db_path=":memory:", resolution_db_path=":memory:")
session = PilotSession("DRYRUN", os.path.join(base, "pilot_items.csv"), tmp)

def resolve(word):
    r = engine.resolve(user_id="dryrun", recognized_word=word)
    session.log(word, r)

resolve("HELP")            # item 1, attempt 1
session.retry()
resolve("HELP")            # item 1, attempt 2
session.next_item()        # -> item 2
resolve("PAIN")            # item 2, attempt 1
session.next_item(); session.next_item(); session.next_item(); session.next_item()
session.next_item(); session.next_item(); session.next_item(); session.next_item()
session.next_item(); session.next_item(); session.next_item()   # -> item 13 (PAIN > HELP)
resolve("PAIN"); resolve("HELP")
session.close(); engine.close()

print("\n===== CSV CONTENTS =====")
print(open(session.log_path, encoding="utf-8").read())
os.remove(session.log_path); os.rmdir(tmp)
print("temp log deleted:", not os.path.exists(session.log_path))
