"""
utils/pilot_logger.py
-------------------------
Optional pilot-study logging, used ONLY when main.py is started with
`--pilot <participant_id>`. It is a passive observer: it never touches the
recognizer, voting, persistence gate, IntentEngine, ontology or dashboard.
It only reads the word that was passed to IntentEngine.resolve() and the
ResolutionResult that came back, and appends one CSV row.

The current test item / intended meaning is printed to the TERMINAL only.
Nothing here draws on the participant-facing window.

No names, video or images are stored; only the CSV columns below.
"""
import csv
import os
import re
import sys
import time

COLUMNS = [
    "timestamp", "item_number", "intended_words", "intended_meaning",
    "attempt_number", "recognized_word", "engine_status", "engine_intent",
    "engine_confidence",
]
ITEM_COLUMNS = ("item_number", "intended_words", "intended_meaning")

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def validate_participant_id(participant_id: str) -> str:
    if not _ID_RE.match(participant_id or ""):
        sys.exit("[PILOT][ERROR] participant_id must be 1-40 characters of letters, "
                 "digits, '_' or '-' (use a code like P01, not a name).")
    return participant_id


def load_items(path: str) -> list:
    if not os.path.isfile(path):
        sys.exit(f"[PILOT][ERROR] Test-item file not found: {path}\n"
                 f"        Create it with columns: {','.join(ITEM_COLUMNS)}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        # tolerate '#' comment lines so the file can carry an "edit before use" note
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or tuple(reader.fieldnames) != ITEM_COLUMNS:
        sys.exit(f"[PILOT][ERROR] {path} must have the header: {','.join(ITEM_COLUMNS)} "
                 f"(found: {reader.fieldnames})")
    items = [{k: (row[k] or "").strip() for k in ITEM_COLUMNS} for row in reader]
    if not items:
        sys.exit(f"[PILOT][ERROR] {path} has a header but no test items.")
    return items


class PilotSession:
    def __init__(self, participant_id: str, items_path: str, log_dir: str):
        self.participant_id = validate_participant_id(participant_id)
        self.items = load_items(items_path)
        self.index = 0
        self.attempt = 1

        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{self.participant_id}_{stamp}.csv")
        self._fh = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)
        self._fh.flush()

        print(f"[PILOT] Participant '{self.participant_id}' | {len(self.items)} items "
              f"| logging to {self.log_path}")
        print("[PILOT] Keys: N = next item   R = count a retry   "
              "(C clears the word before a retry; items are shown here only)")
        self._print_item()

    # ── item navigation ──────────────────────────────────────────────
    def _current(self):
        return self.items[self.index] if self.index < len(self.items) else None

    def _print_item(self) -> None:
        item = self._current()
        if item is None:
            print("[PILOT] *** No more items. Further resolutions are logged with an empty item_number. ***")
            return
        print(f"[PILOT] ITEM {item['item_number']} ({self.index + 1}/{len(self.items)}) | "
              f"words: {item['intended_words']} | meaning: {item['intended_meaning']} | "
              f"attempt {self.attempt}")

    def next_item(self) -> None:
        if self.index < len(self.items):
            self.index += 1
        self.attempt = 1
        self._print_item()

    def retry(self) -> None:
        self.attempt += 1
        print(f"[PILOT] Retry counted -> attempt {self.attempt}")

    # ── logging ──────────────────────────────────────────────────────
    def log(self, recognized_word: str, result) -> None:
        """Append one row for one IntentEngine.resolve() call. `recognized_word`
        is the exact string passed to resolve(); `result` is its return value.
        Never raises into the main loop and never modifies `result`."""
        try:
            item = self._current() or {"item_number": "", "intended_words": "", "intended_meaning": ""}
            row = [
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                item["item_number"], item["intended_words"], item["intended_meaning"],
                self.attempt, recognized_word,
                result.status, result.selected_intent or "",
                f"{result.final_score:.4f}",
            ]
            self._writer.writerow(row)
            self._fh.flush()
            os.fsync(self._fh.fileno())
            print(f"[PILOT] logged: item {row[1] or '-'} attempt {self.attempt} "
                  f"word={recognized_word} status={result.status} "
                  f"intent={row[7] or '-'} conf={row[8]}")
        except Exception as exc:   # logging must never break the session
            print(f"[PILOT][WARNING] Could not write log row: {exc}")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        print(f"[PILOT] Log saved: {self.log_path}")
