"""
evaluation/latency_experiment.py
------------------------------------
EXPERIMENT 9 — Latency.

Measures real wall-clock time for each stage in isolation and for the
complete pipeline, over many repeated calls. Does NOT include webcam/
MediaPipe/RF inference time — that's the locked recognition system,
measured and reported separately in earlier phases; this measures only
what happens AFTER a word is already recognized.
"""
import statistics
import time

from intent.ontology import IntentOntology
from layer3_temporal.engine import TemporalContextEngine
from layer5_personal.engine import PersonalContextEngine
from reasoning.resolver import IntentResolver
from decision_gate.gate import ConfidenceDecisionGate
from resolution.engine import IntentEngine

NOW = 1_700_000_000.0
N_ITERATIONS = 300


def _stats(times_ms: list) -> dict:
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "n": len(times_ms),
    }


def run() -> dict:
    ontology = IntentOntology()
    temporal_engine = TemporalContextEngine()
    temporal_engine.record("latency_user", "CHEST", timestamp=NOW - 20)
    temporal_engine.record("latency_user", "PAIN", timestamp=NOW - 10)

    personal_engine = PersonalContextEngine(db_path=":memory:")
    for _ in range(10):
        personal_engine.update("latency_user", "HELP", "medical_assistance", timestamp=NOW - 5 * 86400)

    candidates = ontology.lookup("HELP")
    temporal_result = temporal_engine.evaluate("latency_user", "HELP", candidates, now=NOW)
    personal_result = personal_engine.evaluate("latency_user", "HELP", candidates, now=NOW)

    resolver = IntentResolver()
    gate = ConfidenceDecisionGate()

    results = {}

    # Layer 1
    times = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        ontology.lookup("HELP")
        times.append((time.perf_counter() - t0) * 1000.0)
    results["layer1_candidate_generation"] = _stats(times)

    # Layer 3
    times = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        temporal_engine.evaluate("latency_user", "HELP", candidates, now=NOW)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["layer3_temporal"] = _stats(times)

    # Layer 5
    times = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        personal_engine.evaluate("latency_user", "HELP", candidates, now=NOW)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["layer5_personal"] = _stats(times)

    # Reasoning Engine
    times = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        resolver.resolve("HELP", candidates, temporal=temporal_result, personal=personal_result, now=NOW)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["reasoning_engine"] = _stats(times)

    # Decision Gate
    reasoning_result = resolver.resolve("HELP", candidates, temporal=temporal_result, personal=personal_result, now=NOW)
    times = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        gate.evaluate(reasoning_result, now=NOW)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["decision_gate"] = _stats(times)

    # Complete IntentEngine (fresh, isolated per iteration to avoid
    # personal-profile-write side effects skewing later iterations)
    times = []
    for i in range(N_ITERATIONS):
        engine = IntentEngine(personal_db_path=":memory:", resolution_db_path=":memory:")
        engine.temporal_engine.record("latency_user", "CHEST", timestamp=NOW - 20)
        engine.temporal_engine.record("latency_user", "PAIN", timestamp=NOW - 10)
        t0 = time.perf_counter()
        engine.resolve("latency_user", "HELP", now=NOW)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["complete_intent_engine"] = _stats(times)

    personal_engine.close()
    return results
