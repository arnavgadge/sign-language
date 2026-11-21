# Baseline Comparison: Paper Text and Reference Details

All numbers below are taken from `baselines/results/` (run date 2026-11-21). Part A is written for direct use in the paper. Part B is background for co-authors.

---

# PART A: PAPER-READY TEXT

## A1. Baselines

To test whether the engine's performance reflects contextual reasoning rather than simple regularities, it was compared with six baselines on the same 500 cases. Every method received only the recognized word, the preceding words with their elapsed times, the user's interaction history for that word, and the word's candidate intents from the ontology; case identifiers, user identifiers and scenario metadata were withheld.

Most-Frequent-Intent (MFI), analogous to the most-frequent-sense baseline [CITE: most-frequent-sense baseline, WSD survey], selected the candidate with the highest ontology prior, breaking ties by priority weight and then by ontology order. Per-User Majority selected the intent most often recorded in the user's history for the word, resolving equal counts with the MFI ordering and falling back to MFI when no history existed (293 of 500 cases).

Two supervised classifiers, logistic regression and random forest [CITE: random forests, Breiman 2001], were trained on recency-weighted context features, word–context interaction features and history features over all 33 labels, including AMBIGUOUS and LITERAL; their outputs were restricted to the word's candidates and the two abstention labels. They were evaluated out-of-fold with stratified 5-fold cross-validation (seed 20240521), with hyperparameters and the recency time constant selected by a nested 3-fold search within each training fold [CITE: nested cross-validation, Varma & Simon 2006].

Two large language models received an identical zero-shot prompt and output schema: Qwen3 14B (qwen3:14b, Q4_K_M quantization) [CITE: Qwen3 technical report], run locally through Ollama as an offline alternative deployable on the same hardware, and Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite) [CITE: Gemini model documentation], a lightweight cloud model. Both were run on 21 November 2026 at temperature 0 with a fixed seed and JSON-schema-constrained output, with thinking disabled (Qwen) or set to its minimum level (Gemini). Each returned a decision, a best intent and a self-reported confidence; none of the 1,000 outputs was invalid.

## A2. Evaluation Protocol

Forced-choice accuracy was computed on the 378 intent-gold cases from each method's top-ranked candidate, regardless of abstention. Full accuracy was computed on all 500 cases from each method's final output, counted as correct only on an exact match with the gold label, including AMBIGUOUS and LITERAL; methods that never abstain are therefore limited to 378/500 = 0.756. The wrong-when-resolved rate is the fraction of committed intent outputs that were incorrect. Confidence intervals (95%) used a paired percentile bootstrap with 1,000 resamples [CITE: bootstrap confidence intervals, Efron & Tibshirani 1993]. Baselines were compared with the engine by exact two-sided McNemar tests [CITE: McNemar test for comparing classifiers, Dietterich 1998]. Confidence ranking was summarized by AURC [CITE: selective classification / risk-coverage, Geifman & El-Yaniv 2017]. Latency covered inference only.

## A3. Main Results Table

### (i) LaTeX (requires `\usepackage{booktabs}`)

```latex
\begin{table*}[t]
\centering
\caption{Comparison with baselines on the 500-case benchmark. Forced: top-ranked intent on the 378 intent-gold cases. Full: final output on all 500 cases. Latency is the median per decision. The best value in each column is in bold; McNemar $p$ is against the Intent Engine and is not ranked.}
\label{tab:baselines}
\begin{tabular}{lcccccc}
\toprule
Method & Forced acc [95\% CI] & Full acc [95\% CI] & Wrong-when-resolved & AURC & McNemar $p$ & Latency (ms) \\
\midrule
Intent Engine (ours)          & 0.976 [0.960, 0.989]          & 0.838 [0.804, 0.870]               & 0.135                 & 0.003                     & ---                   & 0.096 \\
MFI                           & 0.481 [0.429, 0.529]          & 0.364$^{a}$ [0.322, 0.406]          & 0.636$^{d}$           & 0.499                     & $2.4\times10^{-50}$  & \textbf{0.006} \\
Per-User Majority             & 0.627 [0.579, 0.672]          & 0.474$^{a}$ [0.428, 0.516]          & 0.526$^{d}$           & 0.162                     & $1.7\times10^{-34}$  & 0.008 \\
Logistic regression           & \textbf{0.981} [0.968, 0.995] & \textbf{0.844} [0.812, 0.874]       & 0.129                 & 0.013                     & 0.804                 & 0.179 \\
Random forest                 & 0.976 [0.960, 0.989]          & 0.814 [0.780, 0.846]               & \textbf{0.128}        & \textbf{0.002}            & 1.000                 & 7.867 \\
Qwen3 14B (local)             & 0.780 [0.738, 0.823]          & 0.552 [0.508, 0.594]               & 0.257                 & 0.077$^{b}$               & $3.9\times10^{-16}$  & 4106.1 \\
Gemini 3.5 Flash-Lite (cloud) & 0.907 [0.878, 0.934]          & 0.694 [0.650, 0.736]               & 0.279                 & 0.053$^{b}$               & $1.1\times10^{-4}$   & 1212.2$^{c}$ \\
\bottomrule
\end{tabular}

\vspace{2pt}
\begin{minipage}{\textwidth}\footnotesize
$^{a}$ Never abstains; maximum attainable full accuracy is 378/500 = 0.756.
$^{b}$ Computed from the model's self-reported confidence.
$^{c}$ Includes network round-trip time.
$^{d}$ Always commits to an intent, so this equals $1-$ full accuracy.
\end{minipage}
\end{table*}
```

### (ii) Markdown (for Word)

| Method | Forced acc [95% CI] | Full acc [95% CI] | Wrong-when-resolved | AURC | McNemar p | Latency (ms) |
|---|---|---|---|---|---|---|
| Intent Engine (ours) | 0.976 [0.960, 0.989] | 0.838 [0.804, 0.870] | 0.135 | 0.003 | — | 0.096 |
| MFI | 0.481 [0.429, 0.529] | 0.364ᵃ [0.322, 0.406] | 0.636ᵈ | 0.499 | 2.4×10⁻⁵⁰ | **0.006** |
| Per-User Majority | 0.627 [0.579, 0.672] | 0.474ᵃ [0.428, 0.516] | 0.526ᵈ | 0.162 | 1.7×10⁻³⁴ | 0.008 |
| Logistic regression | **0.981** [0.968, 0.995] | **0.844** [0.812, 0.874] | 0.129 | 0.013 | 0.804 | 0.179 |
| Random forest | 0.976 [0.960, 0.989] | 0.814 [0.780, 0.846] | **0.128** | **0.002** | 1.000 | 7.867 |
| Qwen3 14B (local) | 0.780 [0.738, 0.823] | 0.552 [0.508, 0.594] | 0.257 | 0.077ᵇ | 3.9×10⁻¹⁶ | 4106.1 |
| Gemini 3.5 Flash-Lite (cloud) | 0.907 [0.878, 0.934] | 0.694 [0.650, 0.736] | 0.279 | 0.053ᵇ | 1.1×10⁻⁴ | 1212.2ᶜ |

*Caption:* Comparison with baselines on the 500-case benchmark. Forced: top-ranked intent on the 378 intent-gold cases. Full: final output on all 500 cases. Latency is the median per decision. Best value per column in bold; McNemar p is against the Intent Engine and is not ranked.

ᵃ Never abstains; maximum attainable full accuracy is 378/500 = 0.756.
ᵇ Computed from the model's self-reported confidence.
ᶜ Includes network round-trip time.
ᵈ Always commits to an intent, so this equals 1 − full accuracy.

## A4. Results

On this benchmark, the engine outperformed MFI, Per-User Majority and both LLMs (Table~\ref{tab:baselines}; McNemar p ≤ 1.1×10⁻⁴). Its forced-choice accuracy of 0.976 compares with 0.481 for MFI, 0.627 for Per-User Majority, 0.780 for Qwen3 14B and 0.907 for Gemini 3.5 Flash-Lite. The supervised classifiers matched its forced-choice accuracy (logistic regression 0.981, p = 0.804; random forest 0.976, p = 1.000), but only when trained on labelled data from the same generator.

In the learning curve (Table~\ref{tab:lc}), both classifiers remained below the engine with 200 labelled cases (forced-choice 0.964 and 0.940; full 0.783 and 0.762) and reached it only with approximately 400. When features encoding dataset-generation artifacts were removed, their full accuracy fell to 0.818 and 0.784, below the engine's 0.838. When an entire scenario type was withheld from training (Table~\ref{tab:loco}), neither classifier identified any genuinely ambiguous case (0/60), whereas the engine flagged 25 of 60.

The LLMs failed in opposite directions. Gemini over-committed: it resolved 92.6% of cases, and 27.9% of its committed answers were wrong. Qwen over-hedged, answering AMBIGUOUS in 38.6% of cases while 25.7% of its committed answers were wrong. With calibrated abstention, the engine resolved 78.8% of cases with a wrong-when-resolved rate of 13.5%. It achieved this without labelled data, offline, at a median latency of approximately 0.1 ms per decision (0.096 ms), compared with 4106.1 ms for Qwen3 14B and 1212.2 ms for Gemini 3.5 Flash-Lite.

## A5. Supplementary Tables

### Learning curve

**LaTeX**

```latex
\begin{table}[t]
\centering
\caption{Classifier accuracy versus number of labelled training cases (mean $\pm$ s.d. over 5 subsamples per outer fold; fixed hyperparameters). The Intent Engine uses no labelled data.}
\label{tab:lc}
\begin{tabular}{rcccc}
\toprule
 & \multicolumn{2}{c}{Logistic regression} & \multicolumn{2}{c}{Random forest} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
$N$ & Forced & Full & Forced & Full \\
\midrule
25  & 0.580 $\pm$ 0.009 & 0.389 $\pm$ 0.024 & 0.566 $\pm$ 0.021 & 0.354 $\pm$ 0.029 \\
50  & 0.890 $\pm$ 0.005 & 0.658 $\pm$ 0.008 & 0.840 $\pm$ 0.015 & 0.610 $\pm$ 0.020 \\
100 & 0.942 $\pm$ 0.011 & 0.732 $\pm$ 0.005 & 0.903 $\pm$ 0.012 & 0.697 $\pm$ 0.028 \\
200 & 0.964 $\pm$ 0.007 & 0.783 $\pm$ 0.005 & 0.940 $\pm$ 0.007 & 0.762 $\pm$ 0.011 \\
400 & 0.984 & 0.848 & 0.979 & 0.818 \\
\midrule
Intent Engine & 0.976 & 0.838 & 0.976 & 0.838 \\
\bottomrule
\end{tabular}
\end{table}
```

**Markdown**

| N | LR forced | LR full | RF forced | RF full |
|---|---|---|---|---|
| 25 | 0.580 ± 0.009 | 0.389 ± 0.024 | 0.566 ± 0.021 | 0.354 ± 0.029 |
| 50 | 0.890 ± 0.005 | 0.658 ± 0.008 | 0.840 ± 0.015 | 0.610 ± 0.020 |
| 100 | 0.942 ± 0.011 | 0.732 ± 0.005 | 0.903 ± 0.012 | 0.697 ± 0.028 |
| 200 | 0.964 ± 0.007 | 0.783 ± 0.005 | 0.940 ± 0.007 | 0.762 ± 0.011 |
| 400 | 0.984 | 0.848 | 0.979 | 0.818 |
| Intent Engine | 0.976 | 0.838 | 0.976 | 0.838 |

*Caption:* Classifier accuracy versus number of labelled training cases (mean ± s.d. over 5 subsamples per outer fold; fixed hyperparameters). The Intent Engine uses no labelled data.

### Leave-one-category-out

**LaTeX**

```latex
\begin{table*}[t]
\centering
\caption{Leave-one-category-out evaluation: classifiers trained on all other scenario categories and tested on the held-out one. The Intent Engine requires no training. --- indicates no intent-gold cases in the category.}
\label{tab:loco}
\begin{tabular}{lrcccccc}
\toprule
 & & \multicolumn{2}{c}{Intent Engine} & \multicolumn{2}{c}{Logistic regression} & \multicolumn{2}{c}{Random forest} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Held-out category & $n$ & Forced & Full & Forced & Full & Forced & Full \\
\midrule
Clear context            & 120 & 1.000 & 0.975 & 0.850 & 0.625 & 0.792 & 0.275 \\
Temporal-dependent       & 80  & 1.000 & 0.850 & 0.953 & 0.700 & 0.891 & 0.500 \\
Personalization-dependent& 80  & 1.000 & 1.000 & 1.000 & 0.938 & 1.000 & 0.900 \\
Combined temporal/personal & 70 & 1.000 & 1.000 & 1.000 & 0.971 & 1.000 & 0.986 \\
Genuinely ambiguous      & 60  & ---   & 0.417 & ---   & 0.000 & ---   & 0.000 \\
Conflict                 & 40  & 1.000 & 0.675 & 0.923 & 0.475 & 0.885 & 0.500 \\
Cold start               & 20  & 1.000 & 0.550 & 0.875 & 0.500 & 0.750 & 0.400 \\
Literal fallback         & 20  & ---   & 1.000 & ---   & 1.000 & ---   & 0.450 \\
Adversarial              & 10  & 0.100 & 0.100 & 1.000 & 1.000 & 1.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{table*}
```

**Markdown**

| Held-out category | n | Engine forced | Engine full | LR forced | LR full | RF forced | RF full |
|---|---|---|---|---|---|---|---|
| Clear context | 120 | 1.000 | 0.975 | 0.850 | 0.625 | 0.792 | 0.275 |
| Temporal-dependent | 80 | 1.000 | 0.850 | 0.953 | 0.700 | 0.891 | 0.500 |
| Personalization-dependent | 80 | 1.000 | 1.000 | 1.000 | 0.938 | 1.000 | 0.900 |
| Combined temporal/personal | 70 | 1.000 | 1.000 | 1.000 | 0.971 | 1.000 | 0.986 |
| Genuinely ambiguous | 60 | — | 0.417 | — | 0.000 | — | 0.000 |
| Conflict | 40 | 1.000 | 0.675 | 0.923 | 0.475 | 0.885 | 0.500 |
| Cold start | 20 | 1.000 | 0.550 | 0.875 | 0.500 | 0.750 | 0.400 |
| Literal fallback | 20 | — | 1.000 | — | 1.000 | — | 0.450 |
| Adversarial | 10 | 0.100 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 |

*Caption:* Leave-one-category-out evaluation: classifiers trained on all other scenario categories and tested on the held-out one. The Intent Engine requires no training. — indicates no intent-gold cases in the category.

## A6. Limitations

Several limitations apply. On ten adversarial cases containing a misleading context word, the engine was correct only once, whereas both classifiers and both LLMs were correct on all ten and MFI and Per-User Majority on five; these failures are attributable to the ontology's hand-written trigger lists, which treat the misleading word as evidence for a competing intent. The scenarios, gold labels and engine rules were written by the same author, which may favour the engine's mechanism, and the synthetic generator left artifacts, such as filler words never occurring in ambiguous cases, that supervised models can exploit. The LLM prompt reported elapsed times but did not state the engine's two-minute relevance window, and Gemini 3.5 Flash-Lite is a lightweight model, so the LLM results do not represent an upper bound for LLMs. LLM confidence was self-reported. Evaluation on independently annotated data from real users is required to confirm these findings.

---

# PART B: REFERENCE DETAILS FOR CO-AUTHORS

## B1. What the reviewer asked and what we did

- **Request:** replace the "literal" baseline, which always outputs the recognized word (accuracy 0.078), with four real baselines: a base prior, a per-user majority vote, a trained classifier and an LLM. All must be compared with the engine on the same test cases.
- **Delivered:** six baselines in `baselines/`, covering all four requested types; the classifier and the LLM each come in two variants. Nothing was changed in the engine, the ontology or `evaluation/`.
- **Engine reference:** the engine is the production `full_reliability_weighted` configuration, evaluated exactly as `evaluation/final_variants.py` does. Every run checks that it reproduces the published numbers (419/500 gated, 369/378 top-1), with 0 per-case mismatches.
- **Extra experiments on the classifiers:** artifact ablation, learning curve and leave-one-category-out.

## B2. Test set and inputs

- **Cases:** 500, generated by `evaluation.final_dataset.generate_final_dataset()` with seed 20240521. This is the same set as `run_final_evaluation.py` uses.
- **Gold labels:** 378 intent-gold, 83 AMBIGUOUS and 39 LITERAL.
- **Other properties:** 31 gold intents, 12 target words, 35 simulated users, 9 scenario categories and 466 unique prompts.
- **Inputs every method receives** (`CaseInput` in `baselines/common.py`):
  - `recognized_word`;
  - the earlier words, with `sequence_offsets_seconds`;
  - `personal_seed`, the user's (word, intent, count, days_ago) history;
  - the candidates from `IntentOntology.lookup(word)`.
- **Forbidden fields** (used only for scoring and per-category reporting):

  | Field | Why it is excluded |
  |---|---|
  | `ground_truth`, `expected_status` | They are the answers. |
  | `notes` | Contains the gold intent id verbatim. |
  | `sample_id` | Its prefix encodes the scenario category. |
  | `scenario_category` | Determines the label, e.g. genuinely_ambiguous is always AMBIGUOUS. |
  | `user_id` | Its prefix encodes the simulated personality (U1xx medical-leaning, U2xx task-leaning). |

- **Isolation:** history is seeded into fresh in-memory stores per case. Nothing carries over between cases, and `data/*.db` is never read or written.

## B3. Exact rules per method

**MFI**
- Ranks candidates by `base_probability`, breaking ties by higher `priority_weight`, then by earlier position in `data/intent_ontology.json`.
- **HELP:** `medical_assistance` and `task_assistance` tie at 0.25. `medical_assistance` wins on `priority_weight` (1.2 vs 1.0).
- **GO:** `destination_request` and `permission_to_leave` tie at 0.35 with equal `priority_weight` (1.0). `destination_request` wins because it comes first in the file.
- Confidence: `base_probability`. Never abstains.
- The engine itself does not read these prior fields.

**Per-User Majority**
- Counts `personal_seed` entries for the target word only, weighted by count. Intents that aren't candidates for the word are ignored.
- **Equal counts:** resolved by the MFI ordering.
- **No history for the word:** falls back to the full MFI ranking with `used_fallback = True`. This happened in 293/500 cases, including 194/378 intent-gold cases.
- Confidence: the winner's share of the count, or `base_probability` for fallback cases. Never abstains.

**Classifiers (LR, RF)**
- **Features:**
  - the target word as a one-hot feature;
  - recency-weighted earlier-word presence, the sum of exp(−offset/τ);
  - the same values keyed by (target word, earlier word);
  - the number of earlier words, the offset of the most recent earlier word, and a has-earlier-words flag;
  - per candidate in the user's history: the count, count × exp(−days/30), and its share of the total;
  - a has-history flag.

  The vocabulary is built from the training fold only.
- **Labels:** 33 classes.
- **Masking:** outputs are restricted to the candidates plus AMBIGUOUS and LITERAL, then renormalized.
  - Gated answer: argmax over the masked classes.
  - Forced answer: argmax over the candidates.
  - Confidence: the forced candidate's renormalized probability.
- **Models:**
  - LR: `class_weight="balanced"`, standardized features, lbfgs; C ∈ {0.1, 1, 10}.
  - RF: `class_weight="balanced"`, `random_state=20240521`; n_estimators ∈ {200, 500}.
  - Both: τ ∈ {30, 60, 120, 300, 600} s.
- **Protocol:** outer `StratifiedKFold(5, shuffle=True, random_state=20240521)`; inner 3-fold selection by macro-F1 of the gated answer.
- **Chosen per outer fold (0–4):**
  - LR: τ = 60, 60, 120, 120, 120; C = 0.1, 1, 0.1, 0.1, 1.
  - RF: τ = 120, 60, 120, 60, 60; trees = 500, 200, 500, 500, 200.
- **Total fit time including tuning:** LR 12.0 s, RF 36.9 s.
- **Learning curve:** used fixed hyperparameters (LR τ = 120, C = 0.1; RF τ = 60, 500 trees).

**LLMs**
- **Prompt:** identical for both (`baselines/llm_prompt.txt`): the target word, the earlier words with "N seconds ago", the user's history, and the candidates as id | label | message.
  - Ontology trigger lists, priors and weights are not included.
  - AMBIGUOUS and LITERAL are defined neutrally.
- **Output:** JSON `{answer, best_intent, confidence}`.
  - `answer` is the gated answer.
  - `best_intent` is the forced answer.
  - `confidence` is self-reported.
- **Validation:** an invalid output is retried once; if still invalid, it is recorded as LITERAL / first candidate / confidence 0. Neither model produced an invalid output (0/500 each).
- **Settings:**
  - **Qwen3 14B:** `qwen3:14b`, Ollama 0.40.17; qwen3, 14.8B parameters, Q4_K_M, context 40960. Temperature 0, seed 20240521, `num_predict` 60, `think: false`, per-case JSON schema as `format`. An untimed warm-up loads the model.
  - **Gemini 3.5 Flash-Lite:** `gemini-3.5-flash-lite`, google-genai 3.6.1. Temperature 0, seed 20240521, `max_output_tokens` 60, `thinking_level: MINIMAL`, `response_json_schema` per case. SDK retries disabled, 60 s timeout, throttle below 15 requests/min.
  - Both ran on 2026-11-21.
- **Gemini quota:** the calls were split across two Google Cloud projects because of the free-tier limit, with an identical model and settings.
  - Project 1: 495 calls (450 ok, 44 HTTP 503, 1 hung request).
  - Project 2: 19 calls (16 ok, 3 HTTP 503).

## B4. Full tables

**Decision rates (all 500 cases)**

| Method | Resolved | Ambiguous | Literal | Wrong when resolved |
|---|---|---|---|---|
| Intent Engine | 0.788 | 0.134 | 0.078 | 0.135 (53/394) |
| Classifier (LR) | 0.806 | 0.090 | 0.104 | 0.129 (52/403) |
| Classifier (RF) | 0.768 | 0.124 | 0.108 | 0.128 (49/384) |
| LLM-local (Qwen3 14B) | 0.608 | 0.386 | 0.006 | 0.257 (78/304) |
| LLM-cloud (Gemini 3.5 Flash-Lite) | 0.926 | 0.056 | 0.018 | 0.279 (129/463) |
| MFI / Per-User Majority | 1.000 | 0 | 0 | 0.636 / 0.526 (= 1 − full accuracy) |

**Additional per-method values**

| Method | Macro-F1 (forced) | Latency mean / median (ms) | McNemar b / c |
|---|---|---|---|
| Intent Engine | 0.987 | 0.223 / 0.096 | — |
| MFI | 0.245 | 0.006 / 0.006 | 192 / 5 |
| Per-User Majority | 0.407 | 0.011 / 0.008 | 137 / 5 |
| Classifier (LR) | 0.971 | 0.181 / 0.179 | 7 / 9 |
| Classifier (RF) | 0.969 | 6.223 / 7.867 | 9 / 9 |
| LLM-local (Qwen3 14B) | 0.603 | 4185.4 / 4106.1 | 83 / 9 |
| LLM-cloud (Gemini 3.5 Flash-Lite) | 0.803 | 2642.5 / 1212.2 | 35 / 9 |

*b* = engine right and the method wrong; *c* = the reverse.

What is timed:

| Method | Timed | Not timed |
|---|---|---|
| Engine | lookup + Layer 3/5 evaluation + resolve + gate | engine construction, history seeding |
| Classifiers | single-case featurization + `predict_proba` + masking | training and tuning |
| LLMs | per-case wall time (Gemini includes network) | rate-limit pauses, model load |

**Forced-choice accuracy per category (intent-gold cases)**

| Category (n) | Engine | MFI | Per-User Majority | LR | RF | Qwen3 14B | Gemini 3.5 FL |
|---|---|---|---|---|---|---|---|
| clear_context (120) | 1.000 | 0.275 | 0.275 | 0.992 | 0.992 | 0.575 | 0.833 |
| temporal_dependent (64) | 1.000 | 0.438 | 0.438 | 0.953 | 0.938 | 0.703 | 0.875 |
| personalization_dependent (80) | 1.000 | 0.725 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| combined_temporal_personal (70) | 1.000 | 0.643 | 1.000 | 1.000 | 1.000 | 1.000 | 0.986 |
| conflict (26) | 1.000 | 0.385 | 0.500 | 0.923 | 0.885 | 0.500 | 0.769 |
| cold_start (8) | 1.000 | 0.375 | 1.000 | 0.875 | 0.875 | 1.000 | 1.000 |
| adversarial (10) | 0.100 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |

**Artifact ablation** (removes sequence length, has-earlier-words, the most-recent offset and the 7 filler-word features). Evidence that these are artifacts:

- 0/83 AMBIGUOUS-gold cases contain a filler word, against 220/417 other cases.
- 60/83 AMBIGUOUS-gold cases have exactly 4 earlier words.

| Method | Forced [95% CI] | Full [95% CI] | Resolved / Ambiguous / Literal | Wrong when resolved |
|---|---|---|---|---|
| Intent Engine | 0.976 [0.960, 0.989] | 0.838 [0.804, 0.870] | 0.788 / 0.134 / 0.078 | 0.135 |
| LR, all features | 0.981 [0.968, 0.995] | 0.844 [0.812, 0.874] | 0.806 / 0.090 / 0.104 | 0.129 |
| LR, no artifacts | 0.992 [0.981, 1.000] | 0.818 [0.784, 0.852] | 0.822 / 0.062 / 0.116 | 0.151 |
| RF, all features | 0.976 [0.960, 0.989] | 0.814 [0.780, 0.846] | 0.768 / 0.124 / 0.108 | 0.128 |
| RF, no artifacts | 0.989 [0.979, 0.997] | 0.784 [0.748, 0.820] | 0.778 / 0.062 / 0.160 | 0.149 |

**Outside the two-minute window:** in the 16 temporal_dependent cases with LITERAL gold, all the context is more than 120 s old.
- The engine answers LITERAL on 16/16.
- Gemini resolves all 16.
- Qwen resolves 6 and answers AMBIGUOUS on 10.

## B5. Reproduction

**Environment:**
- `.venv/` with Python 3.14.7, scikit-learn 1.7.2 and google-genai 3.6.1. google-genai is not in `requirements.txt`; install it with `pip install google-genai`.
- Ollama 0.40.17 with `qwen3:14b`.

Run from the repo root.

```bash
# Everything, with both LLMs replayed from cache (no Gemini key needed; qwen3:14b must be listed by Ollama)
.venv/bin/python -m baselines.run_baselines --llm local

# Same, with Gemini in live mode (needs GEMINI_API_KEY and GEMINI_MODEL=gemini-3.5-flash-lite; no calls if the cache is complete)
.venv/bin/python -m baselines.run_baselines --llm all

# Extra experiments (classifiers only)
.venv/bin/python -m baselines.artifact_ablation
.venv/bin/python -m baselines.learning_curve
.venv/bin/python -m baselines.leave_category_out
```

**Recomputed versus cached:**
- **Recomputed every run (deterministic):** the engine, MFI, Per-User Majority, LR and RF. The classifiers are retrained, which takes about 1 minute.
- **Replayed from cache:** both LLMs.
- **Current Gemini rows:** scored from the completed cache against the stored engine rows. The full `--llm local` regeneration has not been re-run since then.
- **Warning:** `--llm none` overwrites `metrics.json` and `summary.csv` without the LLM rows.

**Files in `baselines/results/`:**

| File | Contents |
|---|---|
| `predictions.csv` | 3,500 rows (7 methods × 500 cases): answers, confidence, correctness, latency |
| `metrics.json` | All metrics, classifier fold logs, and `llm_runs` (model ids, settings, run date, call and cache counts) |
| `summary.csv` | One row per method |
| `risk_coverage.csv` | Risk–coverage points |
| `artifact_ablation.json`, `learning_curve.csv`, `leave_category_out.csv` | The extra experiments |
| `llm_cache_llm_local.jsonl`, `llm_cache_llm_cloud.jsonl` | Raw LLM responses |
| `gemini_call_log_project1.jsonl`, `gemini_call_log.jsonl` | Gemini call logs (no keys) |
