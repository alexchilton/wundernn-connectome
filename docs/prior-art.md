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
