# pilot_items.csv — EDIT BEFORE USE

`pilot_items.csv` is a **template**. Review and edit every row before the session.

- Columns (header must stay exactly): `item_number,intended_words,intended_meaning`
- Every word must exist in `data/intent_ontology.json`.
- Multi-word items use ` > ` between words (e.g. `PAIN > HELP`). Each word is
  signed and resolved in order with Enter; each resolve logs its own row under
  the same `item_number`.
- Lines starting with `#` are ignored.
- Run: `python main.py --pilot P01` (use a participant code, not a name).
- Items and meanings are printed in the terminal only, never on the participant window.
- Keys in pilot mode: `N` = next item (attempt resets to 1), `R` = count a retry.
  Press `C` to clear the word before a retry (the engine will not re-resolve an unchanged word).
  N and R are ignored while `S` collection mode is on.
- Logs go to `pilot_logs/<participant_id>_<YYYYMMDD_HHMMSS>.csv`.
