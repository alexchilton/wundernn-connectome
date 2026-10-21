# Prior art: Wunder Fund ML competitions (RNN Challenge, Predictorium, Alpha Connectome)

Research date: 2026-10-20.
Method: WebSearch + WebFetch only, no code run.
Caveat stated up front: wundernn.io is a JS-rendered SPA.
WebFetch only returned page titles for it ("Wunder Fund RNN challenge"), never body content.
Everything about wundernn.io page content below therefore comes either from a GitHub repo that quotes/mirrors the docs, from the Habr announcement post, or from WebSearch's own AI-generated summary of search snippets (flagged explicitly as SEARCH-SUMMARY, not fetched primary text).
Numbers that conflict across sources are presented as conflicts, not resolved by guessing.

## 1. There are three distinct Wunder Fund competitions, not one

Evidence points to a clear lineage, each one a superset of the last.

1. **"Wunder Fund RNN Challenge"** (the original/first one) — simplest task, R² metric.
2. **"Predictorium" (internally "Challenge 2")** — single-instrument LOB task, Weighted-Pearson (WP) metric, same feature-naming scheme (p/v/dp/dv) as Alpha Connectome.
3. **"Alpha Connectome" (internally "Challenge 3")** — the task in this brief: two instruments, 112 features, 20,000-row sequences, WP metric.

Primary confirmation that Predictorium and Alpha Connectome share metric and submission mechanics comes from two independent GitHub repos that quote each competition's own README verbatim (`KNambiarDJsc/Wunder_predictorium-challenge`, `mingda19/wunderfund`, `dozken/wunder`) — see sections 2 and 4.

Sources: [github.com/dozken/wunder](https://github.com/dozken/wunder) (explicitly names "Challenge 2: Predictorium" and "Challenge 3: Alpha Connectome (WNN33)" as subfolders of one repo), [github.com/KNambiarDJsc/Wunder_predictorium-challenge](https://github.com/KNambiarDJsc/Wunder_predictorium-challenge/blob/main/README.md), [github.com/mingda19/wunderfund](https://github.com/mingda19/wunderfund).

## 2. Task format comparison

### 2a. Wunder Fund RNN Challenge (the original, ~autumn 2025)

- Per the Habr announcement: N=32 anonymized features, 517 training sequences, each sequence 1,000 states, states chronological, sequences independent.
- Metric: R² (coefficient of determination), averaged over all N output features — confirmed independently by three repos (`1rvinn/wundernn`, `ManishjangirIITG/wunder_fund`, `aicheye/wundernn`) that all describe an R²-scored, single-vector-per-step prediction task with no WP/weighting concept.
- No stated warm-up split in the Habr post itself; `aicheye/wundernn`'s README (as summarized) describes a 100-step warm-up before scoring starts, which may describe this challenge or may already describe Predictorium — the two could not be disambiguated from the fetched text.
- Dates: start 15 September, submissions close 1 December, then 1–15 December is a rescoring/winner-determination window on a fresh private dataset. Prize pool $13,600 USDT, top 8 get cash + fast-track interview at Wunder Fund.
- `Meteoman05/Wunderfund-RNN-challange` repo description states this ran "autumn of 2025," which is the only explicit year found for this specific challenge (the Habr post itself gives month/day but WebFetch did not surface its publication year).

Sources: [habr.com/.../954662](https://habr.com/ru/companies/wunderfund/articles/954662/) (primary announcement, Russian), [kpfu.ru mirror article](https://kpfu.ru/computing-technology/startoval-wunder-rnn-challenge-mdashsorevnovanie.html), [github.com/Meteoman05/Wunderfund-RNN-challange](https://github.com/Meteoman05/Wunderfund-RNN-challange), [github.com/1rvinn/wundernn](https://github.com/1rvinn/wundernn), [github.com/aicheye/wundernn](https://github.com/aicheye/wundernn), [github.com/ManishjangirIITG/wunder_fund](https://github.com/ManishjangirIITG/wunder_fund).

**Conflict to flag:** one WebSearch AI-summary (not a fetched page) described this same challenge's input as "six bid and six ask price levels, their twelve volumes, four trade prices, four trade volumes" (i.e. an LOB-specific 32-channel scheme) with a baseline of 0.617052 WP — that is actually the Predictorium/Connectome vocabulary (WP metric, p/v/dp/dv framing) bleeding into the summary of a different challenge. Treat that specific combination (32 LOB channels + WP + 0.617052) as unreliable; the R²-metric, generic-"N anonymized features" version from the primary Habr post and three independent repos is the better-supported description of the original RNN Challenge.

### 2b. Predictorium (Challenge 2, ~Dec 2025 – Mar 2026)

Directly confirmed by two repos quoting its own docs:

- 28 features per step: `p0-p11` (bid/ask LOB price levels), `v0-v11` (matching volumes), `dp0-3`/`dv0-3` (trade prices/volumes) — i.e. exactly the single-instrument version of the feature scheme Alpha Connectome doubles up to 112 (two 52-feature instrument blocks + 8 `a` columns).
- Two targets `t0`, `t1` (future price-movement indicators) — same target naming as Alpha Connectome.
- 1,000-row sequences, first 99 rows warm-up (`need_prediction` false), rest scored.
- Dataset: 10,721 training sequences, 1,444 validation sequences.
- Metric: Weighted Pearson Correlation Coefficient, predictions clipped to [-6, 6], weighted by |target| — same family of metric as Alpha Connectome's (which instead clips to [-2, 2] per the brief), computed per target then averaged.
- Baseline: "vanilla GRU," shipped as ONNX for inference speed — architecturally identical framing to the brief's stateful-GRU-in-ONNX baseline.
- Submission: Docker container based on `python:3.11-slim-bookworm`.
- Dates: one source (Chan-Weng-Kin-Damien repo) states "January–March 2026" for this competition; a different WebSearch AI-summary states "December 31, 2025 – March 1, 2026" with an $13,600 USDT / top-8 prize structure — the two are consistent with each other (Dec 31 start, Mar 1 cutoff, "Jan-Mar" as the bulk of it) and not treated as a real conflict.
- Field size: at least 5,062 entrants (from one participant's repo stating "finished 417th of 5,062").

Sources: [github.com/KNambiarDJsc/Wunder_predictorium-challenge](https://github.com/KNambiarDJsc/Wunder_predictorium-challenge/blob/main/README.md), [github.com/MylesThomas/wunder-predictorium](https://github.com/MylesThomas/wunder-predictorium/blob/main/docs/QUICK_START.md), [github.com/Chan-Weng-Kin-Damien/LOB-prediction](https://github.com/Chan-Weng-Kin-Damien/LOB-prediction), [github.com/AsadKhalil/wundernn-predictorium](https://github.com/AsadKhalil/wundernn-predictorium), [github.com/dozken/wunder](https://github.com/dozken/wunder).

### 2c. Alpha Connectome (Challenge 3) — the brief's own task, independently confirmed

Two separate repos (not affiliated with each other, as far as can be told) quote near-identical Alpha Connectome README text, which matches the brief closely:

- 112 float32 features: two 52-feature groups for instruments `i0`/`i1` (book + trade), plus 8 extra columns `a0`-`a7`.
- Targets `t0`, `t1` = future price-movement indicators of instrument `i0`.
- 20,000 rows/sequence, 99-row warm-up, predictions scored on steps 99-19,999.
- Training: 10,607 sequences / 212.14M rows. Validation: 1,873 sequences / 37.46M rows.
- Metric: Weighted Pearson, weights = clip(|target|, ..) clipped to [-2.0, 2.0] (one repo's wording), computed per target then averaged across both targets and all eligible rows.
- Submission: stateful `solution.py` exposing `PredictionModel()` + `predict(data_point)`, returns 2 finite predictions per scored row or `None` during warm-up, zipped ≤20MB.
- Hardware: 1 vCPU, 16GB RAM, CPU-only, 60-minute wall clock for the full test set, Linux + Python 3.10/3.11, no internet.
- **Baseline score conflict:** `mingda19/wunderfund`'s README (as fetched) states the vanilla-GRU ONNX baseline scores **0.589595 WP**, and describes it as running "statelessly on a single CPU thread" — this is lower, and a different statefulness claim, than the brief's own baseline figure of **0.617 WP** for a **stateful** GRU (hidden 128). Do not treat 0.589595 as this brief's baseline; it may be an earlier/different baseline snapshot of the same competition page, fetched at a different time, or a misreading by the summarizing step inside WebFetch. Flagging rather than resolving.

Sources: [github.com/mingda19/wunderfund](https://github.com/mingda19/wunderfund), [github.com/dozken/wunder](https://github.com/dozken/wunder) (challenge-3 subfolder).

## 3. Winning / notable solutions on the EARLIER competitions

### 3a. RNN Challenge — KlishevDA, 11th place (has a written technical report in-repo)

- Architecture "GRUForecaster": 1-layer unidirectional GRU, 300 hidden units, over a 100-step context window, **plus** a parallel MLP branch on just the last timestep (dim→300→300, ReLU); the two 300-dim outputs are concatenated to 600-dim; 4 independent MLP heads map 600→output, predictions averaged across heads.
- Trained as windowed 100-step-context/single-step-target regression (not full-sequence stateful BPTT), but run statefully at inference by keeping the GRU hidden state across steps.
- Loss: plain MSE across all 4 heads.
- Optimizer: AdamW, lr 1e-3 with a hand-set schedule (1e-3 epochs 1-3, 1e-4 epochs 4-5, 2e-5 epochs 6-9), weight_decay 3e-4, gradient clip 1.0, no dropout.
- Normalization: per-model z-score standardizer, (x-mean)/(std+eps).
- Ensembling: 3 independently trained models from 3 different head/tail/middle data splits, predictions averaged.
- Reported incremental gains (their own numbers, in R²): custom LR schedule +0.005, hyperparameter tuning (hidden size/context length/batch size) +0.004, multi-head averaging +0.003, 3-model ensemble +0.003, auxiliary window-statistics losses +0.0005-0.001 (unstable).
- What they tried and discarded: deeper GRUs, dropout, LSTM, TCN, attention, data augmentation.
- Resources: ~9 min/model train time on a single Colab GPU (~27 min for the 3-model ensemble), ~7MB/model.

Source: [github.com/KlishevDA/WunderFund-RNN-Challenge-11th-Place-Solution](https://github.com/KlishevDA/WunderFund-RNN-Challenge-11th-Place-Solution).

### 3b. RNN Challenge — HuggingFace `msrishav/wunder-rnn-gru-ensemble`

- 5-member ensemble of causal GRUs, heterogeneous sizes: 3 members at 256 hidden units, 2 at 384, all 2-layer, unidirectional, predicting at every timestep while carrying hidden state across the whole sequence (full-sequence stateful, unlike 3a's windowed approach).
- Training: full-sequence BPTT, loss masked to scored steps only (warm-up excluded from loss).
- Normalization: input "pre-whitening" to approx. N(0,1) without heavy tails — the model card explicitly notes the data is mean-reverting, and naive "predict current state as next state" gives R²≈-0.37, which is why whitening/pre-processing mattered.
- Inference: single-core, 4.17 ms/prediction, O(1) work per step — relevant directly to the brief's own "1 CPU, stateful, deterministic" constraint.
- Scores: CV R²=0.3922, matching the public-leaderboard top of 0.3920 at the time, with the eventual finals winner at 0.3964 — i.e. the ceiling on this earlier, simpler (R²-scored) challenge sits just under 0.40, with the best public entries clustering at 0.39-0.40.
- Training hardware: single RTX 3050, ~1.8s/epoch.

Source: [huggingface.co/msrishav/wunder-rnn-gru-ensemble](https://huggingface.co/msrishav/wunder-rnn-gru-ensemble).

### 3c. RNN Challenge — other entries (lower signal)

- `1rvinn/wundernn`: finished 22nd, R²=0.3894 — consistent with the ~0.39-0.40 ceiling above. Code described by its own author as "very messy"; the author points to a separate (unfetchable, auth-gated) Google Drive folder for their real write-up.
- `aicheye/wundernn`: a from-scratch Transformer attempt (sinusoidal+learnable positional encoding, multi-head self-attention, residual+layernorm), mean R²=0.396 with strong per-feature variance (best feature R²=0.536, worst 0.259); no leaderboard placement stated; this is in the same ~0.39-0.40 band as the GRU ensembles, i.e. no evidence that attention beat GRU here.

Sources: [github.com/1rvinn/wundernn](https://github.com/1rvinn/wundernn), [github.com/aicheye/wundernn](https://github.com/aicheye/wundernn).

### 3d. Predictorium — Chan-Weng-Kin-Damien, 417th of 5,062

- Blend of 3 models: the organizer-provided GRU baseline, a DeepLOB-style attention model, and a causal TCN (dilations 1/2/4/8/16/32, kernel size 3, ~62k parameters).
- Feature engineering: added one step of lookback ("delta features") on top of the 32 raw channels.
- Training trick: trained the TCN on full sequences rather than short windows, which raised the number of supervised targets per sequence from ~8 (windowed) to 901 (full scored region) — a direct, measured point in favor of full-sequence training over windowed sampling for this task family.
- Engineering discipline: TCN receptive field deliberately matched to the 128-step context length, with a hard failure guard if it didn't; blend weights fit via sequence-grouped cross-validation with shrinkage.
- Inference: 3 TCN seeds averaged via a shared history buffer, served through ONNX Runtime with graph optimizations on, CPUExecutionProvider.
- Score: mean correlation 0.2663 on a sealed holdout (544 sequences, 490,144 scored ticks), with a large, reported target asymmetry — t0=+0.4108 vs t1=+0.1217 — i.e. one of the two targets is intrinsically far easier to predict than the other, a finding likely to transfer to Alpha Connectome's own t0/t1 pair given the shared naming and lineage.
- The repo mentions a longer internal `RESULTS.md` covering rejected approaches, not independently fetched here.

Source: [github.com/Chan-Weng-Kin-Damien/LOB-prediction](https://github.com/Chan-Weng-Kin-Damien/LOB-prediction).

### 3e. Alpha Connectome itself — dozken/wunder, challenge-3 (in-progress public attempt, not a final winner)

This is the single most load-bearing find: someone is already working the exact competition in the brief, in the open, with a public score.

- Architecture: GRU 192 hidden units × 2 stacked layers, with a 64-128-wide input projection and first-difference input preprocessing.
- Normalization: notes that Wunder's features arrive "already rank-transformed and clipped (mass at ±2.32)"; after their own standardization values reach ±58, so they additionally clip standardized inputs to ±10. No per-sequence (online) renormalization at inference, specifically to preserve level information the global WP metric needs.
- Loss: sequence-level Weighted-Pearson loss computed via running *detached* sufficient statistics carried across 1000-step truncated-BPTT chunks (gradients only flow through the current chunk; the repo reports that naive per-chunk Pearson loss, without carrying the running stats, degraded scores 0.575→0.553).
- Handling the brief's "hidden scoring mask": trained an auxiliary small GRU to predict which rows are likely to be in the scoring mask (held-out AUC 0.94), then applied loss only to rows it predicts are scored — unmasked-row-only training scored 0.58 vs 0.30 training everywhere, i.e. mask-prediction is a first-order lever on this task, not a minor tweak.
- Training recipe: AdamW + cosine LR schedule, 3-5 epochs over train + non-held-out validation sequences, batch of 64 full sequences with state carried across batches, EMA/SWA weight averaging after training.
- Inference speed: shipped ONNX model (`seq_5ep`) runs at 21.3 µs/row (measured via their own `export_slim.py`), against a budget of ~96 µs/row (37.46M rows / 60 min) — about 4.5x headroom in local measurement, but they note the same exported graph measured 38→52→62 minutes across four separate days on (presumably) the real scoring host, forcing them to budget ~3x more conservatively than their local Docker numbers suggested. This is a concrete, directly-relevant warning about scoring-host latency variance for anyone targeting this same 60-minute/39.4M-row budget.
- Score: public leaderboard 0.6569 (rank #11, submission id `0IP4NDZ1`); best held-out (192 sequences) 0.6864; local-to-public offset consistently ≈ -0.037. Held-out score decays by scoring quarter within a sequence — Q1 0.703, Q2 0.685, Q3 0.654, Q4 0.638 — i.e. the stateful GRU's hidden state degrades over the ~20,000-step horizon despite being trained with 1000-step truncated-BPTT chunks, which directly supports the brief's framing of "sequences of 20,000 steps" as the hard part.

Source: [github.com/dozken/wunder](https://github.com/dozken/wunder), challenge-3 subfolder.

### 3f. Other Alpha-Connectome-named repos found (low/no technical content)

- `rahulb0802/alphac` — titled "Wunder Fund RNN Challenge," returned HTTP 404 on fetch (repo may be private, renamed, or deleted since indexing).
- `Jenish1706/alpha-connectome` — scaffolding only (configs/scripts/src/submission/tests dirs, a `benchmark_latency.py`), no architecture, no score; tells participants to pull datasets/baseline from wundernn.io.
- `NIKITA420240/wunder-connectome` — surfaced in search with a PR titled "Feat/model workflows," not independently fetched in this pass (not prioritized over the two repos above that had real numbers); worth a follow-up read if more Connectome-specific technique detail is needed.

## 4. Leaderboard signal on Alpha Connectome itself (mostly unverified — read carefully)

- The only numbers traced to an actual fetched page (not a search-engine summary) are from §3e: 0.6569 public / 0.6864 held-out for one named participant's own model.
- Everything else below is **SEARCH-SUMMARY**, i.e. WebSearch's own AI-generated gloss over snippet text, never independently confirmed by fetching wundernn.io (which would not render for WebFetch):
  - top entry "545" at 0.832 WP — unverified, and notably far above every other number seen (including this brief's own stated 0.617 baseline and the 0.6569-0.6864 range found in a real repo), so treat as suspect until confirmed from the live site.
  - a cluster around 0.656-0.659 ("WanderingMagician," "Private_sharing_2," "kak-dela").
  - a lower cluster around 0.620-0.622.
- No Discord leaks, Telegram leaderboard screenshots, or public notebooks for Alpha Connectome specifically were found in this pass beyond the GitHub repos above.

## 5. Russian-language / community sources found

- Habr (primary, company blog): the RNN Challenge announcement post, [habr.com/ru/companies/wunderfund/articles/954662/](https://habr.com/ru/companies/wunderfund/articles/954662/) — no comments section content was returned by WebFetch, so participant discussion there (if any) was not captured.
- Wunder Fund also publishes general deep-learning Habr content (unrelated to this competition) under the same company blog, e.g. articles on LSTM, generative models, architecture zoos, transformers — background on the firm's ML culture but not competition-specific.
- A mirror/syndication of the Habr announcement exists on a university site: [kpfu.ru/computing-technology/startoval-wunder-rnn-challenge-mdashsorevnovanie.html](https://kpfu.ru/computing-technology/startoval-wunder-rnn-challenge-mdashsorevnovanie.html).
- A LiveJournal syndication mirror of the same Habr post also exists: [habr-all.livejournal.com/19171532.html](https://habr-all.livejournal.com/19171532.html).
- No ODS.ai/Mattermost thread, no Telegram channel post, and no Discord server specifically discussing Wunder Fund's challenges were located; searches for these only returned generic ODS.ai/Telegram community pages with no Wunder Fund content, so treat "no evidence found" as the honest answer here, not "confirmed absent."
- A single informal social post (Threads) mentions the RNN Challenge dates/prize pool and links to wundernn.io, consistent with (not adding beyond) the Habr figures.

## 6. Net takeaways for the Alpha Connectome brief, inferred (not stated by any one source)

- INFERENCE: the task has visibly been hardened release-over-release by the same organizer — R² (RNN Challenge) → WP + single instrument (Predictorium) → WP + two instruments + hidden scoring mask (Alpha Connectome) — so solution techniques validated on Predictorium (full-sequence training, TCN/GRU/attention blends, mask-aware loss) are the most directly transferable prior art, more so than the earlier R²-scored RNN Challenge.
- INFERENCE: across all three competitions, nobody found beat a GRU-family stateful recurrent model outright; the best-documented techniques are about *how* to train/mask/export the GRU (full-sequence truncated BPTT with carried statistics, mask-prediction sub-model, ONNX export headroom), not about swapping in a fancier architecture — attention and TCN blends matched or trailed GRU-only baselines in the fetched evidence, they did not clearly beat them.
- INFERENCE: the dozken repo's observed held-out-score decay by scoring-quarter (0.703→0.638) is the strongest direct evidence that state-degradation over a 20,000-step sequence is a real, measured risk for this brief's task, not a theoretical concern.
- HYPOTHESIS (unverified): the brief's own baseline (stateful GRU hidden-128, 0.617 WP) and the 0.589595 WP figure found in `mingda19/wunderfund`'s README may be two different snapshots of the same evolving baseline; this was not resolved and should not be treated as settled.

## dozken/wunder challenge-3: read directly on 2026-10-20 (README.md and RESUME.md, repo pushed 2026-09-18)

Everything below is that author's own measurement, quoted from their notes; none of it is reproduced by us yet.
Status 2026-09-16: public 0.6569, rank #11; #1 was 0.6601 and the prize line (#8) 0.6578 at that date.
The metric changed on 2026-09-15 from per-sequence WP to global WP; their rankings inverted, so anything older than that in their log is under the old metric.
Their provided-baseline numbers: held-out 192 seqs 0.6272 (global), full valid 0.5896 (old metric); our starter pack states 0.617052, which may be the rescored figure.
Local held-out to public offset: about -0.03 (0.6864 -> 0.6569).
Mask: about 10% of rows scored, in sticky runs (P(scored | previous scored) = 0.77); a3, a2, a4 carry most of the row-level mask signal.
A GRU mask predictor trained on valid (AUC 0.94) labels train.parquet; the loss is weighted by the predicted scored probability ("soft mask").
Oracle on valid-only training: loss on all rows 0.569 vs loss on is_scored rows 0.605.
Sequence-level Pearson loss (running detached stats over the prefix) beat per-chunk Pearson; then a global pooled Pearson loss.
State reset every 2000 rows in training AND inference (reset2k) gave their best single model, held-out global 0.6864, under the global metric.
Longer TBPTT (chunk 4000) won under the old metric and lost under the global one (level drift across the sequence).
Do not recentre predictions per sequence (0.6823 -> 0.5756); do not rescale (x1.5 costs 0.0038, x2 costs 0.0176, because of the +-2 clip).
First-difference inputs, input projection 64-128, AdamW + cosine, 3-5 epochs, train + non-held-out valid.
Width: GRU 384x2 only about +0.003 over 192x2; LSTM worse than GRU.
Ensemble of two losses: +0.0056 held-out, but two 192x2 graphs timed out on the platform.
Latency: platform about 3x slower than local; the same GRU 192x2 graph took 38, 52, 62 and 64 min on different days; hard kill at 70 min.
Their rule of thumb: platform minutes ~ 1.9 x local-Docker us/row; treat > 30 local us/row as unsafe.
Features are already rank-transformed and clipped (mass at +-2.32); standardised inputs need a +-10 clip.

Note: the competition rules forbid sharing code or models publicly; this repo does so.
We use it for ideas only, and write and train our own code; its checkpoints and submission zips are not used.

---

# Part 2: Alpha Connectome community repos (survey 2)

Scope: read-only survey via `gh api`, no code run, no data downloaded.
Seven repos claiming to target the Wunder Fund "Alpha Connectome" competition (Sep 11 - Nov 15 2026, 20,000-row sequences, 112 features, predict t0/t1, global weighted Pearson, 1 vCPU / 60 min / 39.4M rows).
Then four repos from earlier, related Wunder Fund competitions, checked only for transferable technique and noted task differences.
dozken/wunder already known, not re-surveyed per instruction.

Convention below: a line that reports what a repo's file says is marked **[STATED]**.
A line that is my own reading of the evidence is marked **[INFERENCE]**.

## 1. mnemonik-dev/wunder — richest repo, most directly useful

By far the most developed of the seven; a multi-week, well-instrumented research log.

**[STATED]** Source: `https://github.com/mnemonik-dev/wunder` README.md — solution engine is "Neutrino" (Rust), an `evoforge` genetic algorithm searching feature hyper-parameters, champion replayed by a NumPy `solution.py`.
**[STATED]** Source: README.md — best public score **0.6508**, local (honest half) 0.67841; leader at the time was 0.6646 public.
**[STATED]** Source: `docs/SESSION-REPORT.md` — closing report dated 2026-09-25, covering work from 2026-09-22 to 2026-09-25; "the public score did not improve during this work" after the 0.6508 submission.
**[STATED]** Source: `docs/SESSION-REPORT.md` §2 — component WP scores on the full validation set (1,873 sequences, 3,528,273 scored rows): organisers' GRU baseline 0.61705; linear read-out (ridge on 336 streaming features) 0.63314; MLP 512x128 at 13.3M rows 0.66743; MLP at 35M rows 0.66885; their own GRU 128x2 trained on all 211M rows 0.67024; gradient-boosted trees (400 trees) 0.65511; diagonal state-space model (1024 state) 0.64546.
**[STATED]** Source: `docs/SESSION-REPORT.md` §3 — architecture is a Rust streaming causal feature transform (raw features + residual against fast/mid/slow EMA + lagged difference, 336 features at full spec) feeding three read-outs (ridge, MLP, GRU), blended per target.
**[STATED]** Source: `docs/NEUTRINO_SOLUTION.md` — feature transform gene schema: `ema_fast` (2-60), `ema_mid` (5-300), `ema_slow` (30-2000), `raw_prices`, `use_i1`, `use_dmid`, `use_dslow`, `use_diff` + `diff_lag` (1-50), `vol_norm` + `vol_span`, `use_imbalance`, `log10_lambda` (ridge strength), `weight_power` (sample-weight exponent on `|y|`).
**[STATED]** Source: `docs/NEUTRINO_SOLUTION.md` — champion spec found by the GA: fast EMA span 28, 42-row lagged difference, both instruments, raw prices visible, ridge λ=3.0e-4, weight power 0.75; mid/slow EMA, volatility normalisation and imbalance switched off.
**[STATED]** Source: `docs/SESSION-REPORT.md` §4 "Auditing its decisions" — the GA's holdout (60 sequences) drifted ±0.034 from full-validation truth while the effects it was selecting between were worth only ±0.003; it converged within 20 of 68 trials and spent the rest fitting noise. Two of its block choices (mid/slow EMA off) were later shown wrong (+0.002 measured at scale) once screened at 1,000 sequences instead of 60.
**[INFERENCE]** The generalizable lesson here is that a noisy holdout used for feature search can select the wrong config even when the search mechanism itself is sound; screening candidate blocks directly at a scale where the holdout noise is smaller than the effect being measured is cheap insurance.
**[STATED]** Source: `docs/SESSION-REPORT.md` §5 (iteration 4) — their own GRU (stateful, trained from scratch on raw 112 columns, truncated BPTT over 512-row windows) scored 0.670 alone, +0.053 over the organisers' GRU with an identical 128x2 architecture, attributed entirely to more training data (211M rows/epoch, streamed directly from the Parquet row groups, no export/disk) and a metric-aligned sample weighting.
**[STATED]** Source: `docs/SESSION-REPORT.md` §5 (iteration 5) and §8.1 — substituting their better GRU (0.670) for the organisers' weaker one (0.617) in the blend made the blend *worse* (0.67742 → 0.67647); **five independent model families (linear, MLP, GRU, GBM, a diagonal state-space model) all converged to 0.93-0.96 pairwise residual correlation** while individually spanning 0.6171-0.6702 WP, i.e. blend value comes from error decorrelation, not individual accuracy, and this project found no architecture that decorrelates.
**[INFERENCE]** This is the single most load-bearing finding in the whole survey for us: it is evidence (from an independent team, cross-checked by their own multiple architectures) that just building "yet another sequence model" on these features is very unlikely to move an ensemble, because every family they tried learns essentially the same function. Any model we build should be checked for residual-correlation against whatever else we have *before* investing more tuning time in it.
**[STATED]** Source: `docs/SESSION-REPORT.md` §8.5 / `docs/SPEC-NEXT-MODEL.md` §9 — a diagonal multi-timescale state-space layer (state 1024, spans log-spaced 2 to 20,000 rows, O(N) recurrence costing 11.2 µs/row vs the GRU's 21.3 µs at state 256) was purpose-built to test whether "short memory" was the shared blind spot; it trained to 0.6455 alone but still landed at 0.9532-0.9556 residual correlation with the MLP — **ruled out long memory as the blind spot**, killed at a pre-declared 40-minute gate rather than the planned 6.5-hour full run.
**[STATED]** Source: `docs/SESSION-REPORT.md` §8.6 — of all loss/weighting changes tried (weight-power 1.0 vs 0.75, joint-target head predicting `(t0-t1)/2`/`(t0+t1)/2`, a correlation loss, importance weighting by `p(scored|x)`), only `weight_power 1.00` (the metric's own exponent) helped, by +0.0007 — sub-threshold, everything else was negative.
**[STATED]** Source: `docs/SESSION-REPORT.md` §8.4 and §10 — data forensics: "columns are per-column affine transformed, including sign flips" — e.g. `i0_p10` correlates -0.84 to -0.93 with sibling bid prices, volumes go negative, best-ask reads below best-bid; classical microstructure features (microprice, book imbalance, spread) are **"structurally unrecoverable"** on this data and the imbalance feature scored worst of all families tested (-0.004).
**[STATED]** Source: `docs/SESSION-REPORT.md` §10 — `corr(t0, t1) = -0.738`, negative in all 40 sequences measured (range -0.937 to -0.427); price columns share a dominant PCA factor (PC1 = 70%), volume columns do not (PC1 = 25%).
**[STATED]** Source: `docs/SESSION-REPORT.md` §10 — **the scoring mask is strongly structured**: a classifier predicts `is_scored` from the 112 raw features at AUC 0.867, scored rows carry 19-26% larger target magnitudes than unscored rows, but correcting for this via importance weighting did not help (-0.0004) because the feature→target relationship is sign-stable on scored vs. unscored rows — reweighting only shrinks effective sample size.
**[STATED]** Source: `docs/RUNBOOK.md` / `docs/MLP_3WAY_BLEND.md` — **runtime is the binding constraint, not accuracy**: scoring host measured 1.6-2.1x slower than their dev machine; a config projected at 44.3 min timed out (scored zero), one at 33.5 min scored; practical ceiling stated as **~50 µs/row, ~33 min projected** against the 60-min/4200s limit.
**[STATED]** Source: `docs/RUNBOOK.md` — profiling found half the measured inference budget (49.4→43.5 µs/row after fix) was going to overhead, not arithmetic: fancy-index operations copying arrays to themselves, a fresh `np.concatenate` allocation every row, and unused EMA updates running regardless of whether the champion spec used them — all removed with predictions unchanged to six decimals.
**[STATED]** Source: `docs/RUNBOOK.md` §5b — **local-to-public gap**: three submissions, local "eval half" 0.66800/0.67468/0.67841 → public 0.6423/0.6482/0.6508; the shift is stable at -0.024 to -0.027 (≈70-73% of a local gain survives to public). They set a rule: **a change needs ≥+0.003 on the honest (non-tuned) half of validation before it is distinguishable from noise**, because a quoted "full" blend number (which includes the half the blend weights were fitted on) is optimistic — one +0.00077-on-full change was only +0.00017 on the honest half and scored *below* the model it replaced on the public board.
**[INFERENCE]** This 0.003-on-honest-half threshold and the explicit "quote eval half, never full" discipline are directly reusable as our own go/no-go gate before spending a submission.

## 2. MarkBrezina/WunderFundRNN — the data-forensics / target-identity repo

**[STATED]** Source: `https://github.com/MarkBrezina/WunderFundRNN` README.md (author's own words, quoted verbatim) — "At the end I understood, that t0 and t1 was correlated but inversely, the targets were also correlated with price... The targets are buying edge and sell edge, so these are 'of course' related to each other. Buy now sell later, short now buy back later are inversely correlated... Those two are inversely correlated, with a certain gap, relating to execution, slippage, fee and more."
**[INFERENCE]** This is an unverified hypothesis about target semantics (buy-edge / sell-edge), not a proven decomposition — the author states it was reached after microprice/VWAP/"optimal inventory" hypotheses "fell flat." It is corroborated independently by mnemonik-dev/wunder's measured `corr(t0,t1) = -0.738` (negative in all 40 sequences), which is consistent with — though does not itself prove — an inverse buy/sell-edge story.
**[STATED]** Source: `submission/lobfeat.py` docstring (kit v2, the version actually shipped) — explicit data forensics: "every input is already quantile-normalised (columns sit in about ±2.5, the extra columns bottom out at -5.19934 = sklearn QuantileTransformer's floor, and 'no trade' is a single atom near 0.025)"; and "the price-like columns are not monotone prices: max(bid columns) >= min(ask columns) on 100% of rows, and each column sits in a narrow band around its own fixed value."
**[STATED]** Source: `submission/lobfeat.py` docstring — "So nothing here assumes book structure (best quotes, mid, spread, order flow); v1 did, and on this data those features were meaningless. The GRU does the temporal modelling" — i.e. the **huge 361-slot engineered feature table in README.md (event-clock histories, encoded order-book sort statistics, causal expanding-context stats, etc.) is the author's own superseded v1 approach**, explicitly abandoned in favour of a far simpler v2: just the 112 raw inputs, their one-row differences, and a log time feature (225 total), centred/scaled/`asinh`-compressed.
**[INFERENCE]** This matches mnemonik-dev/wunder's independent finding that classical microstructure features are "structurally unrecoverable" on this (affine-transformed, sign-flipped) feature encoding — two unrelated teams reached the same conclusion from different angles (ablation at scale vs. direct column forensics). Worth treating as reasonably solid: elaborate order-book-shape features are very unlikely to pay off here.
**[STATED]** Source: `submission/MODEL.md` — shipped model: stateful 1-layer GRU, hidden 128, trained 12 epochs, `dev_wp=0.4538`, **`valid_wp=0.653240825`** (t0 0.65363, t1 0.65285), `out_scale=0.8546`.
**[STATED]** Source: `submission/solution.py` / `netstep.py` — inference is a hand-written NumPy (optionally `numba`-JIT'd, falling back automatically if numba is missing) replay of a PyTorch GRU's exact forward pass (gate order r,z,n matching PyTorch's convention), ~10 µs/row feature update + ~20-28 µs/row GRU step at hidden 128.
**[INFERENCE]** The README's open questions section (book-crossing features, orderbook-direction/regime detection, rolling volatility/correlation) reads as earlier, now-superseded v1 brainstorming predating the v2 forensics — should not be read as the project's current direction.

## 3. mingda19/wunderfund — causal dilated TCN with WaveNet-style streaming inference

**[STATED]** Source: `https://github.com/mingda19/wunderfund` `solution/MODEL.md` — architecture: 7-layer causal dilated Conv1d, kernel=3, dilations (1,2,4,8,16,32,64) → receptive field 255 steps, 48 channels, ~64K params, head → (t0, t1).
**[STATED]** Source: `solution/MODEL.md` — inference reimplements training's forward pass as **true incremental (WaveNet-style) streaming**: each conv layer keeps a ring buffer of just its own required past inputs (2×dilation+1) and does an O(1) update per row instead of replaying the full 255-step window every row; verified against the batched PyTorch forward pass to ~1e-6 precision before being adopted. A naive (non-incremental) version was measured at **1141 µs/row, ~12x over budget**.
**[INFERENCE]** This WaveNet-style ring-buffer trick is directly transferable: any causal-conv architecture we try needs this incremental-inference rewrite to be viable under the 1-vCPU/60-min budget, and "verify numpy-streamed output against the batched training forward pass to ~1e-6" is the right acceptance test before trusting it.
**[STATED]** Source: `solution/MODEL.md` "Weight storage format" — they deliberately did **not** export the TCN to ONNX (unlike the baseline GRU), because the fast inference path depends on 7 separate per-layer ring buffers rather than one hidden-state vector, and the incremental logic lives outside the model's `forward()`; ONNX would only capture the slow, full-window recompute unless `forward()` were restructured to take/return the buffers explicitly. They shipped plain-numpy weights (`.npz`) instead, needing only `numpy` at inference (confirmed against the scoring container's `requirements.txt`).
**[STATED]** Source: `solution/MODEL.md` "Fine-tuning decision log" — weighted MSE (`|clip(target,-2,2)|`-weighted) vs plain MSE: **WP 0.546 → 0.609 (+0.063)**, called "the single biggest lever in every experiment."
**[STATED]** Source: `solution/MODEL.md` — PLS factor-compression input channels (112→132 channels): WP 0.6087→0.6100 (+0.0013, called noise-level) but **+51% latency** (63.6→96.0 µs/row), pushing projected full-test time to 59.9 of the 60-min budget — rejected on cost/benefit.
**[STATED]** Source: `solution/MODEL.md` — LightGBM standalone reached WP 0.379→0.444 with more data/weighting, but an optimally grid-searched TCN+LightGBM blend scored WP 0.6088 vs. pure-TCN's 0.6087 — "a 0.0001 difference, i.e. noise," with the optimal blend weight effectively discarding LightGBM. Stated reason: "LightGBM's domain+EMA features are a hand-engineered subset/summary of the exact same raw signal the TCN already consumes directly... no independent information for LightGBM to contribute."
**[INFERENCE]** This is a second, independent instance of the same non-decorrelation finding as mnemonik-dev/wunder (§1 above) — a different architecture (TCN) and a different would-be ensemble partner (LightGBM on hand features) hit the same wall. Strengthens the case that marginal architecture variety alone won't help an ensemble on this data; a decorrelating partner needs a genuinely different information source, not just a different model class over the same raw features.
**[STATED]** Source: `experiment_1_training/model_training.md` §1 "New loss function: wp_aligned_loss" — identifies a real mismatch: the (then-)existing weighted-MSE loss pools `w=|clip(target,-2,2)|`-weighted squared error over *every row in the whole batch* (sequences, steps, both targets combined into one division), so a sequence with intrinsically larger target scale silently dominates the gradient; but the **actual WP metric normalises per (sequence, target) pair first, averages the two targets per sequence, then averages equally across sequences** — every sequence counts equally regardless of its own scale. `wp_aligned_loss` mirrors the metric's own two-stage averaging; a hand-checked toy 2-sequence example with a 10x scale difference gave 0.3175 (new) vs 0.7038 (old pooled) on identical inputs, confirming a real behavioural difference.
**[INFERENCE]** This loss/metric-alignment gap (naive pooled weighted-MSE vs. the metric's actual per-sequence-then-equal-average structure) is a concrete, well-reasoned, and — as far as this survey found — otherwise-unaddressed lever worth checking against our own training loss.
**[STATED]** Source: `solution/MODEL.md` "Known gaps" — `a5`/`a7` (two of the eight auxiliary `a0..a7` columns) correlate at 0.65/0.74 with the **per-sequence magnitude of the target's quantization step** (`eda/notes.md` §9, not itself fetched in this survey), described as "a real, unused lever" at the time of writing.
**[INFERENCE]** This is a specific, checkable claim (not verified by us) about `a5`/`a7` encoding per-sequence target scale; worth a cheap correlation check against our own training data before relying on it.
**[STATED]** Source: `solution/MODEL.md` "Validation results" — best shipped TCN (600 of 10,607 available train sequences): 0.6087 WP on 50 held-out sequences, 0.6050 on the full `ScorerStepByStep` 150-sequence sample, vs. organiser baseline 0.617052 — **"we are currently ~0.008-0.012 below this."** Measured latency 63.5 µs/row → ~39.6 min projected (comfortably inside the 60-min budget).
**[STATED]** Source: `full_training/runs/tcn_full.log` and `experiment_1_training/runs/tcn_experiment1{,_7}.log` (raw training logs, read directly) — later full-dataset runs (all 10,607 sequences, 15 epochs) reached only **best val mean corr 0.4291 / 0.4451 / 0.3968** across three configurations (7-layer, 8-layer+wp_aligned_loss, and a third variant).
**[INFERENCE]** These full-data-run numbers use a **different metric** (raw unweighted Pearson correlation printed by the training loop, not the competition's clipped weighted-Pearson WP) than the 0.6087 WP quoted for the 600-sequence model in `solution/MODEL.md`, so they are not directly comparable on their face; taken at face value they look worse, but that may be a metric-definition artifact rather than a real regression. No log or doc in the repo shows these full-data checkpoints ever being converted to the numpy inference format or re-scored with the actual WP metric — **the repo's only WP-validated, shipped result remains the 600-sequence model at 0.6087 WP, which is below the organiser baseline.**

## 4. Jenish1706/alpha-connectome — clean scaffold, no trained model

**[STATED]** Source: `https://github.com/Jenish1706/alpha-connectome` `submission/solution.py` — the shipped `PredictionModel` is explicitly a placeholder: an EMA of the first two raw features, clipped to [-2,2], with the comment "Replace with an exported model once trained."
**[STATED]** `src/models/` and `src/training/` directories contain only `.gitkeep` — no model or training code exists in this repo.
**[INFERENCE]** Nothing here is a competitive entry; **mark as scaffold-only**. Two small pieces are independently useful as reference, though: `src/utils/metric.py` is a clean, from-scratch reimplementation of the exact competition metric (clip ±2, weight `|clip(y)|`, degenerate-input → 0.0 rather than NaN, averaged over t0/t1) with unit tests (`tests/test_metric.py`) — worth a side-by-side check against our own metric implementation; and `src/data/streamer.py` is a reasonable `IterableDataset` design for TBPTT training directly off Parquet row groups (one row group = one sequence, chunked into fixed-size `SequenceChunk`s for truncated-BPTT, hidden-state detach/reset left to the training loop).

## 5. MattGoddd/wundernn_sep26, Marcocxm171/wunder_fund_challange, Alanduan21/Duan_Yihe_wnn_connectome — starter-pack copies, nothing original

**[STATED]** All three repos contain only the unmodified organiser starter-pack layout (`README.md`, `docs/*.md`, `baseline/`, `utils.py`, `requirements.txt`) — no `solution/`, no training scripts, no experiment docs.
**[STATED]** Verified by git blob SHA: `baseline/solution.py` in all three repos (and in mingda19/wunderfund) is byte-identical — same git SHA `684bdd4f4c681d5bbb6493024f9470f7cba7f885` — confirming these are the organiser's unmodified baseline, not a rewritten variant.
**[STATED]** `Marcocxm171/wunder_fund_challange/intro.ipynb` is a **0-byte empty file** (confirmed via the GitHub contents API: `"size":0`).
**[INFERENCE]** These three repos add nothing beyond what we already have from the starter pack itself; no further reading needed.

---

## Earlier Wunder Fund competitions (technique survey only; task differences noted)

These four are **not** the Alpha Connectome competition. Two distinct earlier tasks appear across them:

- **"Wunder Challenge" / "WunderFund RNN Challenge"** (1rvinn, KlishevDA, and apparently MehulBandhu): predict the **full next market-state vector** (N anonymised features, not just two targets), scored by plain **R²** averaged across features, 1000-step sequences with a 100-step warm-up, feature count/semantics undocumented in what was fetched.
- **"LOB Predictorium"** (kc5587): predicts **two** future price-movement targets from **32** LOB/trade features (not 112), 1000-step sequences (not 20,000), weighted Pearson with weight `|target|` but predictions/targets **clipped to [-6, 6]** (not [-2, 2]) — same metric family as Connectome but a different, smaller feature/clip/sequence-length regime. Author-reported competition result 0.3098 WP, 49th of ~5,000 (top 1%).

**[STATED]** Source: `https://github.com/kc5587/wunderfund-lob-predictorium` README.md — submitted runtime is a **five-member ensemble (2 GRU + 2 LSTM + 1 gated causal TCN)** with **target-specific blend weights**, plus **a residual model trained on ensemble-disagreement features to correct the base predictions**, all exported to ONNX, one inference thread, state reset on `seq_ix` change.
**[INFERENCE]** The residual/disagreement-correction model (a second-stage model consuming the base ensemble's own predictions/spread, not just raw features) is a technique not seen in any of the seven Connectome repos and is directly transferable as an ensembling idea beyond simple weighted-average blending.
**[STATED]** Source: `docs/RESEARCH.md` — research scripts cover: GRU + highway head + augmentation + **weighted-correlation loss** + **auxiliary future-feature forecasting task** (training-time-only auxiliary targets, not available at inference); LSTM with variance-normalised augmentation; gated dilated-causal-TCN ("GLU TCN") with "flattened rolling-state export."
**[INFERENCE]** The auxiliary future-feature forecasting task (multi-task training signal, single-task inference) is a reusable idea worth considering for Connectome, independent of whether it was adopted in kc5587's final v96.1 (`docs/RESEARCH.md` is explicit that inclusion of this script "does not establish that its later checkpoints were submitted").
**[STATED]** Source: `docs/RESEARCH.md` — "Some historical research scripts cap target weights when fitting or validating. Those experimental objectives are not interchangeable with the uncapped competition metric." — i.e. this author also hit and flagged a loss/metric weight-capping mismatch, independently echoing mingda19's `wp_aligned_loss` finding (§3 above) that naive weighted losses can silently diverge from what the competition metric actually computes.
**[STATED]** Source: `KlishevDA/WunderFund-RNN-Challenge-11th-Place-Solution` README.md — **11th place**, R² baseline 0.372 → final submission. Architecture: GRU backbone (hidden 300, windowed input, context length 100) fused with a parallel MLP on the last timestep only, 4 parallel MLP output heads averaged, **3 independently trained models on 3 deliberately different train/val splits** (head/tail/middle 5% holdouts) averaged at inference; stateful inference despite windowed training.
**[STATED]** Source: KlishevDA README.md, "Key Boosters" — measured, ranked gains: custom LR schedule (simplified ReduceLROnPlateau-like) +0.005 R²; hyperparameter sizing (hidden/context/batch, found a ceiling past which further increase did not help) +0.004; multi-head output averaging +0.003; multi-model (seed+split) ensembling +0.003; windowed statistics / autoencoder / per-feature loss weighting each ~+0.0005-0.001 but "unstable", abandoned in favour of more heads/models.
**[STATED]** Source: KlishevDA README.md, "What Didn't Work" — deeper GRU (2-3 layers): no gain; dropout: no gain; TCN and attention variants: "not successful"; data augmentation: decreased performance; LSTM: inconclusive, author ran out of time to tune it, suspects windowed-vs-stateful training choice matters more than cell type.
**[INFERENCE]** KlishevDA's "windowed training, stateful inference" split (train on fixed 100-step windows for speed/stability, deploy the same weights in a recurrent streaming loop) is a reusable training-efficiency pattern distinct from mnemonik-dev/wunder's and mingda19's full-sequence/TBPTT approaches — worth considering if full-sequence training time becomes a bottleneck for us.
**[STATED]** Source: `MehulBandhu/Wunderfund-RNN-competition` README.md and `train.py` — Teacher-Student knowledge distillation: an ensemble of **Transformer teachers trained at multiple window sizes** (24 up to 260, organised in named "families"), their predictions aggregated into soft targets, **distilled into compact Student models**, exported to ONNX with "online residual bias correction" applied post-hoc. Training environment stated as Colab Pro+ (A100 80GB) over ~8 hours — a GPU-side training budget, decoupled from the 1-vCPU CPU-only inference budget.
**[INFERENCE]** The multi-window-size teacher ensemble → single compact ONNX student distillation pattern is the clearest "how to get ensemble-like latency-constrained inference without paying ensemble-like inference latency" technique in the whole survey, directly relevant to our 1-vCPU/60-min constraint; it is more sophisticated than anything attempted in the seven Connectome repos.
**[STATED]** Source: `1rvinn/wundernn` README.md — "would not recommend to go through my approaches through the `competition_package` directory; it's very messy" (author's own words); **#22 leaderboard place, final R² 0.3894**. Repo tree shows 60+ dated/named experiment variants (ema, lstm, attention with a dozen cosine/head/dropout/fold permutations, cnngru, transformer, k-fold ensembles up to 10 folds, generative-pretrained-transformer-named variants, etc.), each kept as a separate zip and unpacked directory with its own checkpoint(s).
**[INFERENCE]** This repo is not useful for technique extraction (no written rationale ties a given variant name to a measured result beyond a few `summary.txt` files not read in this pass) but is a clear negative example: unstructured one-off-variant sprawl with no comparative ledger makes it nearly impossible, even for the author, to know what worked and why — the opposite of mnemonik-dev/wunder's and mingda19's documented decision logs. Worth noting as "what not to do" rather than mining further.

---

## Summary: ranked, actionable findings for us

1. **Five independent model families (linear ridge, MLP, GRU, GBM, diagonal state-space) converge to 0.93-0.96 pairwise residual correlation on this exact feature set, across two unrelated teams (mnemonik-dev/wunder and mingda19/wunderfund, the latter also showing a TCN+LightGBM blend adds nothing).** Before investing tuning time in a new architecture, check its residual correlation against whatever model(s) we already have — a new architecture alone is unlikely to decorrelate. Source: `mnemonik-dev/wunder/docs/SESSION-REPORT.md` §5/§8.1, `mnemonik-dev/wunder/docs/SPEC-NEXT-MODEL.md` §9, `mingda19/wunderfund/solution/MODEL.md`.
2. **Classical microstructure features (microprice, spread, book imbalance, bid/ask-level interpretation) are not recoverable on this data** — columns are per-column affine-transformed including sign flips, confirmed independently by direct column forensics (MarkBrezina) and by feature ablation at scale (mnemonik-dev/wunder, imbalance feature scored -0.004). Do not spend time on book-structure feature engineering. Source: `MarkBrezina/WunderFundRNN/submission/lobfeat.py`, `mnemonik-dev/wunder/docs/SESSION-REPORT.md` §8.4/§10.
3. **A naive `|clip(y)|`-weighted MSE loss, pooled across the whole batch, does not mirror the actual WP metric**, which normalises per (sequence, target) pair first and then averages equally across sequences — a sequence with larger target scale silently dominates a pooled loss. `wp_aligned_loss` fixes this and was hand-verified to change behaviour on a toy example (0.3175 vs 0.7038). An independent, differently-scoped competition (kc5587) flagged the same class of mismatch from a different angle. Check our own training loss against this. Source: `mingda19/wunderfund/experiment_1_training/model_training.md` §1, `kc5587/wunderfund-lob-predictorium/docs/RESEARCH.md`.
4. **Runtime, not accuracy, is the practical ceiling**: mnemonik-dev/wunder measured the scoring host at 1.6-2.1x slower than their dev machine and lost a submission (scored zero) to a 44.3-min projection against the 60-min/4200s limit; their stated safe ceiling is ~50 µs/row / ~33 min projected, with half their original inference budget wasted on avoidable per-row allocation/fancy-indexing overhead before profiling. Profile `solution.py` early and budget in µs/row, not just WP. Source: `mnemonik-dev/wunder/docs/RUNBOOK.md`.
5. **A causal dilated TCN needs an explicit WaveNet-style incremental (ring-buffer, O(1)/row) reimplementation to be viable** — the naive full-window-recompute version measured 1141 µs/row (~12x over budget) before this rewrite; verify the incremental version against the batched training forward pass to ~1e-6 before trusting it, exactly as mingda19 did. Also explains why they shipped plain numpy weights instead of ONNX for this architecture. Source: `mingda19/wunderfund/solution/MODEL.md`.
6. **Set an explicit, pre-declared go/no-go threshold before trusting any local gain**: mnemonik-dev/wunder's rule is "quote the honest/untuned half of validation, never the blend-fitted full number, and require ≥+0.003 improvement there before believing a change" — a +0.00077-on-full change was only +0.00017 (noise) on the honest half and scored below the model it replaced on the public board. The local→public shift was stable at -0.024 to -0.027 across three submissions. Adopt this gate before spending one of our 5 daily submissions. Source: `mnemonik-dev/wunder/docs/RUNBOOK.md` §5b, `docs/SESSION-REPORT.md` §9.
7. **Multi-window-size teacher ensemble distilled into a single compact ONNX student** (MehulBandhu, earlier competition) is the most sophisticated latency-vs-ensemble-quality technique found anywhere in this survey; worth considering if we want ensemble-level accuracy without paying N-model inference cost. A simpler, directly-portable version of the same idea: kc5587's **residual/disagreement-correction second-stage model**, trained on the base ensemble's own prediction spread, which is a different information source than raw features and therefore a more promising decorrelation candidate than "yet another architecture on the same raw inputs" (see #1). Source: `MehulBandhu/Wunderfund-RNN-competition` README.md/`train.py`, `kc5587/wunderfund-lob-predictorium` README.md.
8. **t0/t1 are very likely an inverse buy-edge/sell-edge pair** (`corr(t0,t1) = -0.738` measured by mnemonik-dev/wunder across 40 sequences; MarkBrezina's unverified but directionally consistent "buying edge and sell edge" hypothesis) and **the scoring mask itself is predictable from the raw features at AUC 0.867** though correcting for it does not help accuracy (mnemonik-dev/wunder measured this twice, importance weighting cost -0.0004). Worth knowing when designing a joint-target head or a mask-aware feature, but note mnemonik-dev/wunder already tried and rejected a joint `(t0-t1)/2, (t0+t1)/2` head (-0.0011) and mask-importance weighting, so don't re-run those specific variants without a new angle. Source: `mnemonik-dev/wunder/docs/SESSION-REPORT.md` §10, `MarkBrezina/WunderFundRNN/README.md`.

## Our check of this report (2026-10-20, 23:10)

Finding 3 (pooled |y|-weighted MSE does not mirror a per-sequence-averaged metric) describes the OLD metric; the current METRIC.md in our starter pack is global (one pooled correlation per target), which our loss matches; board: weighted MSE 0.63279 vs plain MSE 0.61346.
Finding 8 (mask weighting -0.0004 for mnemonik-dev) contradicts our held-out result: soft-mask 0.67231 vs |y|-weighted 0.66368 (+0.0086, one seed each); the board score of sub_wsoft.zip will settle it.
Finding 1 (residual correlation 0.93-0.96 across model families) is consistent with our two-model ensemble gain of +0.0003.
