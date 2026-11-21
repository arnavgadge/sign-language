"""
baselines/llm.py
--------------------
Zero-shot LLM baselines with ONE shared prompt (baselines/llm_prompt.txt), one
shared JSON schema and one shared validation path:

  llm_cloud -- Gemini via the google-genai SDK. Model id: env GEMINI_MODEL,
               key: env GEMINI_API_KEY (never printed, logged or written).
  llm_local -- qwen3:14b via the local Ollama HTTP API.

Prompt inputs are ONLY: recognized_word, earlier words with offsets, this case's
personal_seed for the target word, and the word's candidates from
IntentOntology.lookup() (id, label, output_text). No forbidden field and no
engine knowledge (triggers, conflicts, priors, weights) is ever rendered.

Validation: answer must be a candidate id, AMBIGUOUS or LITERAL; best_intent
must be a candidate id; confidence a number in [0, 1]. Invalid -> one retry
(attempt=1, a separate cache entry). Still invalid -> answer=LITERAL,
best_intent=first candidate, confidence=0, invalid=True.

Gemini quota protection (free tier): every API call, retries included, is
appended to baselines/results/gemini_call_log.jsonl. Before each call the
rolling 60 s window is kept below GEMINI_RPM, and the run stops cleanly if the
call would exceed GEMINI_RPD - 5 calls in the last 24 h. HTTP 429/503 -> up to
3 retries with backoff, then a clean stop. Raw responses are cached in
baselines/results/llm_cache_<method>.jsonl keyed by sha256(model, prompt,
settings, attempt), so re-runs resume and never re-request.

Pilot:  python -m baselines.llm --pilot 10
"""
import argparse
import hashlib
import json
import os
import re
import string
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .common import NON_INTENT_LABELS, ONTOLOGY, RESULTS_DIR, CaseInput, Method

SEED = 20240521
MAX_OUTPUT_TOKENS = 60
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_prompt.txt")
CALL_LOG = os.path.join(RESULTS_DIR, "gemini_call_log.jsonl")
TMP_DIR = os.path.join(RESULTS_DIR, "tmp")
DAILY_SAFETY_MARGIN = 5
RETRYABLE = (429, 503)
BACKOFF_S = (15, 30, 60)
REQUEST_TIMEOUT_MS = 60_000
OUTPUT_AFFECTING_SETTINGS = ("backend", "model", "temperature", "seed", "max_output_tokens",
                             "response_mime_type", "response_json_schema", "thinking_config",
                             "num_predict", "format", "think")


class QuotaStop(Exception):
    """Raised to stop a run cleanly (quota, persistent rate limiting, overload)."""


# ── prompt, schema, validation (shared by both backends) ───────────────────

def render_prompt(inp: CaseInput) -> str:
    word = inp.recognized_word.upper()
    earlier = list(zip(inp.conversation_sequence[:-1], inp.sequence_offsets_seconds[:-1]))
    earlier_txt = "\n".join(f"- {w} ({off:.0f} seconds ago)" for w, off in earlier) or "none"
    history = [(i, c, d) for w, i, c, d in inp.personal_seed if w.upper() == word]
    history_txt = "\n".join(f"- {i}: chosen {c} times, most recently {d:g} days ago"
                            for i, c, d in history) or "no history"
    cands_txt = "\n".join(f'{c.id} | {c.label} | "{c.output_text}"' for c in ONTOLOGY.lookup(word))
    with open(TEMPLATE_PATH, encoding="utf-8") as fh:
        template = string.Template(fh.read())
    return template.substitute(target_word=word, earlier_words=earlier_txt,
                               history=history_txt, candidates=cands_txt)


def candidate_ids(inp: CaseInput) -> List[str]:
    return [c.id for c in ONTOLOGY.lookup(inp.recognized_word)]


def json_schema(cands: List[str]) -> Dict:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": cands + list(NON_INTENT_LABELS)},
            "best_intent": {"type": "string", "enum": cands},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "best_intent", "confidence"],
    }


def validate(text: Optional[str], cands: List[str]):
    """-> (parsed or None, reason or None)."""
    try:
        d = json.loads(text or "")
    except (json.JSONDecodeError, TypeError):
        return None, "not valid JSON"
    if not isinstance(d, dict):
        return None, "not a JSON object"
    if d.get("answer") not in cands + list(NON_INTENT_LABELS):
        return None, f"answer {d.get('answer')!r} not allowed"
    if d.get("best_intent") not in cands:
        return None, f"best_intent {d.get('best_intent')!r} not a candidate"
    conf = d.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None, f"confidence {conf!r} not in [0, 1]"
    return d, None


# ── backends ────────────────────────────────────────────────────────────────

class GeminiBackend:
    name = "llm_cloud"

    def __init__(self, cache_only: bool = False):
        """cache_only=True: replay cached responses only (no key needed, never calls the API);
        the model id comes from GEMINI_MODEL or, if unset, from the cache file."""
        from google import genai
        from google.genai import types
        self._types = types
        self.cache_only = cache_only
        self.rpm = int(os.environ.get("GEMINI_RPM", "15"))
        self.rpd = int(os.environ.get("GEMINI_RPD", "500"))
        if cache_only:
            self._key, self._client = "", None
            self.model = os.environ.get("GEMINI_MODEL") or self._model_from_cache()
        else:
            missing = [v for v in ("GEMINI_API_KEY", "GEMINI_MODEL") if not os.environ.get(v)]
            if missing:
                raise SystemExit(f"[LLM] Missing environment variable(s): {', '.join(missing)}. "
                                 f"Set them and re-run.")
            self._key = os.environ["GEMINI_API_KEY"]
            self.model = os.environ["GEMINI_MODEL"]
            # SDK-internal retries disabled so that every request is counted here.
            self._client = genai.Client(api_key=self._key, http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))
        self.settings = {
            "backend": "google-genai", "sdk_version": genai.__version__, "model": self.model,
            "temperature": 0, "seed": SEED, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json", "response_json_schema": "per-case enum schema",
            "thinking_config": {"thinking_level": "MINIMAL"},
            "sdk_retries": "disabled (attempts=1)", "request_timeout_ms": REQUEST_TIMEOUT_MS,
            "rpm_limit": self.rpm, "rpd_limit": self.rpd,
        }
        self.calls_made = 0

    # call log ---------------------------------------------------------------
    def _log_times(self) -> List[float]:
        if not os.path.exists(CALL_LOG):
            return []
        with open(CALL_LOG, encoding="utf-8") as fh:
            return [json.loads(l)["epoch"] for l in fh if l.strip()]

    def calls_last_24h(self) -> int:
        now = time.time()
        return sum(1 for t in self._log_times() if now - t < 86400)

    def _log_call(self, status: str) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(CALL_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 "model": self.model, "status": status}) + "\n")
        self.calls_made += 1

    def _wait_for_quota(self) -> None:
        times = self._log_times()
        now = time.time()
        if sum(1 for t in times if now - t < 86400) + 1 > self.rpd - DAILY_SAFETY_MARGIN:
            raise QuotaStop(f"daily budget reached ({self.rpd - DAILY_SAFETY_MARGIN} calls in 24 h)")
        recent = sorted(t for t in times if now - t < 60)
        wait = 0.0
        if len(recent) >= self.rpm - 1:                      # stay strictly under RPM
            wait = 60.0 - (now - recent[-(self.rpm - 1)]) + 0.5
        if recent:
            wait = max(wait, 60.0 / self.rpm - (now - recent[-1]))
        if wait > 0:
            time.sleep(wait)

    def _scrub(self, msg: str) -> str:
        return msg.replace(self._key, "<KEY>")

    # generation -------------------------------------------------------------
    @staticmethod
    def _model_from_cache() -> str:
        path = os.path.join(RESULTS_DIR, f"llm_cache_{GeminiBackend.name}.jsonl")
        models = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                models = {json.loads(l)["model"] for l in fh if l.strip()}
        if len(models) != 1:
            raise SystemExit(f"[LLM] cache-only Gemini: expected one model in {path}, found {sorted(models)}")
        return models.pop()

    def generate(self, prompt: str, schema: Dict) -> Dict:
        if self.cache_only:
            raise QuotaStop("cache-only mode (no API calls); run with --llm cloud to fetch the rest")
        from google.genai import errors
        t = self._types
        config = t.GenerateContentConfig(
            temperature=0, seed=SEED, max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json", response_json_schema=schema,
            thinking_config=t.ThinkingConfig(thinking_level=t.ThinkingLevel.MINIMAL))
        for attempt in range(len(BACKOFF_S) + 1):
            self._wait_for_quota()
            t0 = time.perf_counter()
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
            except errors.APIError as e:
                self._log_call(f"http_{e.code}")
                if e.code in RETRYABLE and attempt < len(BACKOFF_S):
                    time.sleep(BACKOFF_S[attempt])
                    continue
                if e.code in RETRYABLE:
                    raise QuotaStop(f"HTTP {e.code} persisted after {len(BACKOFF_S)} retries") from None
                raise RuntimeError(self._scrub(f"Gemini HTTP {e.code}: {e.message}")) from None
            except Exception as e:                  # timeouts / connection errors: retry like a 503
                self._log_call(f"network_{type(e).__name__}")
                if attempt < len(BACKOFF_S):
                    time.sleep(BACKOFF_S[attempt])
                    continue
                raise QuotaStop(f"network error persisted after {len(BACKOFF_S)} retries: "
                                f"{self._scrub(type(e).__name__)}") from None
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._log_call("ok")
            cand = resp.candidates[0] if resp.candidates else None
            usage = resp.usage_metadata
            return {
                "text": resp.text,
                "finish_reason": str(cand.finish_reason) if cand else None,
                "model_version": resp.model_version,
                "usage": {"prompt": usage.prompt_token_count, "output": usage.candidates_token_count,
                          "thoughts": usage.thoughts_token_count} if usage else None,
                "latency_ms": latency_ms,
            }
        raise RuntimeError("unreachable")


class OllamaBackend:
    name = "llm_local"
    model = "qwen3:14b"
    URL = "http://localhost:11434/api/chat"

    def __init__(self):
        self.calls_made = 0
        self.settings = {
            "backend": "ollama", "model": self.model, "temperature": 0, "seed": SEED,
            "num_predict": MAX_OUTPUT_TOKENS, "format": "per-case JSON schema", "think": False,
            "ollama_version": self._run(["ollama", "--version"]),
            "ollama_show": self._run(["ollama", "show", self.model]),
        }
        self.settings.update(self._parse_show(self.settings["ollama_show"]))

    @staticmethod
    def _parse_show(text: str) -> Dict:
        """Pull architecture / parameters / quantization / context length out of `ollama show`."""
        wanted = {"architecture": "ollama_architecture", "parameters": "ollama_parameters",
                  "quantization": "ollama_quantization", "context length": "ollama_context_length"}
        out = {}
        for line in text.splitlines():
            for label, key in wanted.items():
                if line.strip().lower().startswith(label) and key not in out:
                    out[key] = line.strip()[len(label):].strip()
        return out

    @staticmethod
    def _run(cmd) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    @classmethod
    def available(cls) -> bool:
        out = cls._run(["ollama", "list"])
        return any(line.split()[0] == cls.model for line in out.splitlines()[1:] if line.split())

    def generate(self, prompt: str, schema: Dict) -> Dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "format": schema, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": SEED, "num_predict": MAX_OUTPUT_TOKENS}}
        if not getattr(self, "_warmed", False):
            # load the model into memory once, untimed and uncached, so case 1's latency
            # doesn't include the model-load time
            warm = {"model": self.model, "messages": [{"role": "user", "content": "ok"}], "stream": False,
                    "think": False, "options": {"num_predict": 1}, "keep_alive": "30m"}
            urllib.request.urlopen(urllib.request.Request(
                self.URL, data=json.dumps(warm).encode(), method="POST",
                headers={"content-type": "application/json"}), timeout=600).read()
            self._warmed = True
        body["keep_alive"] = "30m"
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        t0 = time.perf_counter()
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        self.calls_made += 1
        return {"text": resp.get("message", {}).get("content"), "finish_reason": resp.get("done_reason"),
                "model_version": self.model,
                "usage": {"prompt": resp.get("prompt_eval_count"), "output": resp.get("eval_count")},
                "latency_ms": (time.perf_counter() - t0) * 1000.0}


# ── method ──────────────────────────────────────────────────────────────────

class LLMMethod(Method):
    abstains = True

    def __init__(self, backend, write_to_temp: bool = False):
        """write_to_temp=True: new responses go to baselines/results/tmp/ (used while another
        run owns the main results); they are merged into the main cache by merge_temp_caches().
        Reads always see both the main and the temp cache."""
        self.backend = backend
        self.name = backend.name
        main_path = os.path.join(RESULTS_DIR, f"llm_cache_{self.name}.jsonl")
        tmp_path = os.path.join(TMP_DIR, f"llm_cache_{self.name}.jsonl")
        self.cache_path = tmp_path if write_to_temp else main_path
        self._cache = {}
        for path in (main_path, tmp_path):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rec = json.loads(line)
                            self._cache[rec["key"]] = rec
        self._results: Dict[str, Dict] = {}
        self.cache_reads = 0
        self.stop_reason = None
        self._extras = {}

    def _key(self, prompt: str, attempt: int) -> str:
        """Hash of everything that can change the model's output -- and nothing else,
        so operational settings (timeouts, rate limits, versions) never invalidate the cache."""
        settings = {k: v for k, v in self.backend.settings.items() if k in OUTPUT_AFFECTING_SETTINGS}
        blob = json.dumps({"model": self.backend.model, "prompt": prompt, "settings": settings,
                           "attempt": attempt}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _response(self, prompt: str, schema: Dict, attempt: int) -> Dict:
        k = self._key(prompt, attempt)
        if k in self._cache:
            self.cache_reads += 1
            return self._cache[k]
        out = self.backend.generate(prompt, schema)
        rec = {"key": k, "attempt": attempt, "model": self.backend.model, "prompt": prompt,
               "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **out}
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._cache[k] = rec
        return rec

    def resolve(self, inp: CaseInput) -> Dict:
        prompt = render_prompt(inp)
        if prompt in self._results:
            return self._results[prompt]
        cands = candidate_ids(inp)
        schema = json_schema(cands)
        records, parsed, reason = [], None, None
        for attempt in (0, 1):
            rec = self._response(prompt, schema, attempt)
            records.append(rec)
            parsed, reason = validate(rec.get("text"), cands)
            if parsed:
                break
        invalid = parsed is None
        if invalid:
            parsed = {"answer": "LITERAL", "best_intent": cands[0], "confidence": 0.0}
        res = {"answer": parsed["answer"], "best_intent": parsed["best_intent"],
               "confidence": float(parsed["confidence"]), "invalid": invalid,
               "invalid_reason": reason if invalid else None, "attempts": len(records),
               "latency_ms": sum(r["latency_ms"] for r in records),
               "raw": [r.get("text") for r in records]}
        self._results[prompt] = res
        return res

    def run_all(self, inputs: List[CaseInput]) -> bool:
        """Resolve every input (cache first). Returns True if complete."""
        for n, inp in enumerate(inputs, 1):
            try:
                self.resolve(inp)
            except QuotaStop as e:
                self.stop_reason = str(e)
                print(f"[LLM] {self.name}: stopped cleanly after {n - 1}/{len(inputs)} cases: {e}")
                return False
            if n % 25 == 0:
                print(f"[LLM] {self.name}: {n}/{len(inputs)} "
                      f"(API calls this run: {self.backend.calls_made}, cache reads: {self.cache_reads})")
        return True

    def predict(self, inp: CaseInput):
        res = self._results[render_prompt(inp)]   # never calls the API: run_all() must have completed
        self._extras = {
            "gated_label": res["answer"],
            "gated_status": res["answer"] if res["answer"] in NON_INTENT_LABELS else "RESOLVED",
            "latency_ms": res["latency_ms"], "llm_invalid": res["invalid"], "llm_attempts": res["attempts"],
        }
        return res["best_intent"], res["confidence"], [(res["best_intent"], res["confidence"])]

    def extras(self):
        return dict(self._extras)

    def run_report(self, inputs: List[CaseInput]) -> Dict:
        done = [self._results.get(render_prompt(i)) for i in inputs]
        rep = {
            "method": self.name, "model": self.backend.model, "settings": self.backend.settings,
            "run_date": time.strftime("%Y-%m-%d"), "n_cases": len(inputs),
            "n_cases_done": sum(d is not None for d in done),
            "n_unique_prompts": len({render_prompt(i) for i in inputs}),
            "api_calls_this_run": self.backend.calls_made, "cache_reads_this_run": self.cache_reads,
            "n_invalid": sum(1 for d in done if d and d["invalid"]),
            "n_needed_retry": sum(1 for d in done if d and d["attempts"] > 1),
            "complete": self.stop_reason is None and all(d is not None for d in done),
            "stop_reason": self.stop_reason,
            "model_versions": sorted({r.get("model_version") for r in self._cache.values() if r.get("model_version")}),
        }
        if isinstance(self.backend, GeminiBackend):
            rep["gemini_calls_last_24h"] = self.backend.calls_last_24h()
        return rep


def other_run_active() -> List[int]:
    """PIDs of other baselines runs (run_baselines or the LLM pilot), excluding this
    process and its ancestors (e.g. the shell that launched it)."""
    mine, pid = {os.getpid()}, os.getpid()
    while pid > 1:
        try:
            pid = int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True,
                                     text=True).stdout.strip() or 0)
        except ValueError:
            break
        mine.add(pid)
    out = subprocess.run(["pgrep", "-f", r"python.*-m baselines\.(run_baselines|llm)"],
                         capture_output=True, text=True).stdout.split()
    active = []
    for p in (int(x) for x in out):
        if p in mine:
            continue
        # only real Python processes count -- not shells/monitors whose command text
        # merely mentions a baselines run (that caused a wait-on-the-watcher deadlock)
        comm = subprocess.run(["ps", "-o", "comm=", "-p", str(p)], capture_output=True, text=True).stdout
        if "python" in comm.lower():
            active.append(p)
    return active


def merge_temp_caches() -> Dict[str, int]:
    """Fold baselines/results/tmp/llm_cache_*.jsonl into the main caches (dedup by key),
    then delete the temp files. Only safe when no other run is active."""
    merged = {}
    if not os.path.isdir(TMP_DIR):
        return merged
    for fname in sorted(os.listdir(TMP_DIR)):
        if not (fname.startswith("llm_cache_") and fname.endswith(".jsonl")):
            continue
        tmp_path, main_path = os.path.join(TMP_DIR, fname), os.path.join(RESULTS_DIR, fname)
        keys = set()
        if os.path.exists(main_path):
            with open(main_path, encoding="utf-8") as fh:
                keys = {json.loads(l)["key"] for l in fh if l.strip()}
        added = 0
        with open(tmp_path, encoding="utf-8") as src, open(main_path, "a", encoding="utf-8") as dst:
            for line in src:
                if line.strip() and json.loads(line)["key"] not in keys:
                    dst.write(line if line.endswith("\n") else line + "\n")
                    keys.add(json.loads(line)["key"])
                    added += 1
        os.remove(tmp_path)
        merged[fname] = added
    if not os.listdir(TMP_DIR):
        os.rmdir(TMP_DIR)
    return merged


def build_llm_methods(which: str, write_to_temp: bool = False) -> List[LLMMethod]:
    methods = []
    if which in ("cloud", "all"):
        methods.append(LLMMethod(GeminiBackend()))
    elif which == "local" and os.path.exists(os.path.join(RESULTS_DIR, f"llm_cache_{GeminiBackend.name}.jsonl")):
        # keep the Gemini rows in metrics.json without needing the key or spending quota
        methods.append(LLMMethod(GeminiBackend(cache_only=True)))
    if which in ("local", "all"):
        if OllamaBackend.available():
            methods.append(LLMMethod(OllamaBackend(), write_to_temp=write_to_temp))
        else:
            print(f"[LLM] SKIPPING llm_local: {OllamaBackend.model} is not in `ollama list` yet.")
    return methods


# ── pilot ───────────────────────────────────────────────────────────────────

def pilot(n: int, which: str) -> None:
    from .common import load_cases, to_input
    cases = load_cases()
    step = max(1, len(cases) // n)
    picked = cases[::step][:n]                  # spread across all categories
    busy = other_run_active()
    if busy:
        print(f"[PILOT] another run is active (pid {busy}); new responses go to {TMP_DIR}")
    if which == "local":
        if not OllamaBackend.available():
            print(f"[PILOT] {OllamaBackend.model} is not in `ollama list` yet.")
            return
        methods = [LLMMethod(OllamaBackend(), write_to_temp=bool(busy))]
    else:
        methods = build_llm_methods(which)
    for m in methods:
        t0 = time.time()
        complete = m.run_all([to_input(c) for c in picked])
        wall = time.time() - t0
        res = [m._results.get(render_prompt(to_input(c))) for c in picked]
        res_done = [r for r in res if r]
        rep = m.run_report([to_input(c) for c in picked])
        api_s = [r["latency_ms"] / 1000 for r in res_done]
        print(f"\n[PILOT] {m.name} ({m.backend.model}): {len(res_done)}/{n} done, complete={complete}")
        print(f"[PILOT] wall time {wall:.1f}s; API calls this run {rep['api_calls_this_run']}, "
              f"cache reads {rep['cache_reads_this_run']}")
        if api_s:
            per_case = wall / max(1, rep["api_calls_this_run"]) if rep["api_calls_this_run"] else None
            print(f"[PILOT] mean API time/case {sum(api_s) / len(api_s):.2f}s")
            if isinstance(m.backend, GeminiBackend):
                pace = max(sum(api_s) / len(api_s), 60.0 / (m.backend.rpm - 1))
                print(f"[PILOT] estimated total for 500 cases at <{m.backend.rpm} RPM: "
                      f"~{500 * pace / 60:.0f} min (rate-limit bound)")
            elif per_case:
                print(f"[PILOT] estimated total for 500 cases: ~{500 * per_case / 60:.0f} min")
        print(f"[PILOT] invalid after retry: {rep['n_invalid']}, needed a retry: {rep['n_needed_retry']}")
        if "gemini_calls_last_24h" in rep:
            print(f"[PILOT] Gemini calls in the last 24 h: {rep['gemini_calls_last_24h']}")
        for c, r in zip(picked, res):
            if r and (r["invalid"] or r["attempts"] > 1):
                print(f"[PILOT] RAW ({r['invalid_reason'] or 'retried'}): {r['raw']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=10)
    ap.add_argument("--llm", choices=("cloud", "local", "all"), default="cloud")
    args = ap.parse_args()
    pilot(args.pilot, args.llm)
