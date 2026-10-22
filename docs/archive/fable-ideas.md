# Fable's ideas for Alpha Connectome

Round 2 (2026-10-21) supersedes round 1 where they disagree: round-1 idea 4 (cold start) is measured dead, and distillation is unnecessary because two members fit under the time cap as one fused 256-wide GRU.

# Ideas 2: what is left after the overnight nulls (Fable, 2026-10-21 morning)

Nothing was trained; no run was longer than 60 s; the Colab session was not touched.
Every number marked *measured* comes from this Mac (M4), repo `/Users/alexchilton/PycharmProjects/wundernn-connectome`, data `out/data/valid_*.npy` (held-out = valid[0:192]), predictions `out/full1_holdout_pred.npy`, checkpoints `out/full1_best.pt`, `out/final_s{1,2}/best.pt`, `out/full_{reset2k,h192,l3,dps,all}/best.pt`, run 2026-10-21 between 07:20 and 08:40 with the repo `.venv`.
Scripts `part_a.py` to `part_n.py` and their raw outputs `part_*.out` are in the session scratchpad `/private/tmp/claude-501/-Users-alexchilton-PycharmProjects-iwb-agent1/b33ff776-f0ca-4139-a365-8ede742e7ae3/scratchpad/` (session-specific; the raw numbers are repeated in the appendix).
Anything not marked *measured* is a prior or a **HYPOTHESIS** and is labelled so.

## Headline

The honest stack of what is measured or queued is about +0.010 held-out: 2-member fused ensemble +0.0035, longer schedule with EMA weights +0.002 to +0.004 (prior), a loss-diverse second member +0.002 (prior), valid[0:192] in the final training set +0.001 (prior).
That lands at about 0.693 held-out, 0.661 public, against a prize line of 0.6702 public.
I did not find the missing +0.01, and I measured nine candidate levers dead on 192 held-out sequences (section 2).
What I did find is an error map that tells where the missing score lives and gives every queued arm a sharper read-out (section 1).

Three corrections to `docs/fable-ideas.md`:
cold start (idea 4) is dead, the oracle warm state is worth +0.0003 on the full metric;
"Int8: the time is ORT overhead, not arithmetic" is wrong above 128 units, a trivial graph costs 3.9 µs, GRU 128×2 13.9, 256×2 21.7, 384×2 47.2 (*measured*, tight loop), so weight streaming dominates from 256 up;
the distillation ceiling is now irrelevant because two members fit under the latency cap without distillation (section 3, idea 1).

## 1. The error map: the metric is lost in long quiet epochs

Define an epoch as a run of rows in which none of the 22 `i0_p*` columns changes.
Two facts (*measured*, 192 held-out sequences):

- The target is constant inside an epoch: P(t0 or t1 changes | no `i0_p` change) = 0.00001; P(change | some `i0_p` change) = 0.59; 29.5% of rows start a new epoch.
- WP of full1 by epoch age (rows since the last `i0_p` change), scored rows only:

| age | share of scored rows | share of metric weight | WP full1 | mean \|y\| |
|---|---|---|---|---|
| 0 (change row) | 0.249 | 0.227 | 0.803 | 0.62 |
| 1 | 0.093 | 0.081 | 0.783 | 0.59 |
| 2-3 | 0.097 | 0.091 | 0.768 | 0.64 |
| 4-7 | 0.090 | 0.091 | 0.734 | 0.69 |
| 8-15 | 0.090 | 0.092 | 0.681 | 0.69 |
| 16-31 | 0.096 | 0.099 | 0.642 | 0.70 |
| 32-63 | 0.091 | 0.097 | 0.584 | 0.72 |
| 64-127 | 0.078 | 0.086 | 0.526 | 0.75 |
| 128+ | 0.116 | 0.135 | 0.416 | 0.78 |

So 19% of scored rows and 22% of the weight sit in epochs older than 64 rows, scored at 0.42 to 0.53, while the change rows score 0.80.
An oracle that moves predictions 10% toward the truth gains +0.025 if applied to the age ≥ 64 rows and +0.023 if applied to the age < 8 rows, which carry more than twice the weight (*measured*), so per unit of weight the old-epoch region is 2.5× as sensitive.

What the old-epoch region is not:

- Not model drift inside the epoch: holding the epoch's first prediction scores 0.602 overall and 0.19 on age ≥ 64 rows against the model's 0.462; a causal within-epoch mean scores 0.649 (*measured*).
  The model's later predictions are better than its earlier ones, so information about the fixed target keeps arriving (volumes, trades, `i1`) and the model uses it.
- Not a population artefact only: age-0 rows of epochs that turn out to last 64+ rows score 0.33 (644 rows), so long epochs are intrinsically harder from their first row, and the model climbs from 0.33 to 0.46 as the epoch ages.
- Not a stable bias: the weighted bias on age ≥ 128 rows is −0.37 on even sequences and +0.22 on odd ones; a flat +0.05 offset on age ≥ 64 rows loses 0.0001 and +0.15 loses 0.0013; an out-of-fold per-age-bucket offset loses 0.010 (*measured*).
  The band is dominated by a few sequences with very long epochs, and its WP is noisy: 0.455 to 0.471 across seeds 0, 1, 2 against 0.801 to 0.804 on change rows.
- Not fixed by anything we trained: every overnight member sits at 0.455 to 0.471 on age ≥ 64 and 0.795 to 0.804 on age 0 (*measured*, appendix part_m); the 4-member mean reaches 0.467 and 0.807.
- Not linearly recoverable from anchor features: adding (x − x at the epoch start) and log(age) to a ridge head on the GRU's top state, fitted on 600 training-pool sequences and tested on held-out, moves 0.6772 to 0.6739; on raw features alone, x + anchor deltas + log age scores 0.613 against x + one-row diff 0.617 and x + diff + EMA100 0.631 (*measured*, part_k).

What to do with the map: evaluate every queued arm by age band, not only by the total.
The bands are independent read-outs with different noise: seed sd about 0.002 on age 0 and about 0.008 on age ≥ 64.
A chunk-length or memory change that works must show on age ≥ 64 first, because that is where the information needed is older than a 500-row chunk.
The age split costs one script (part_m.py pattern: `chg = diff(x[:, :, i0_p]) != 0`, `age` = rows since `chg`).

## 2. Measured dead this morning (all 192 held-out sequences, full1 unless stated)

| lever | test | result |
|---|---|---|
| cold start / warm state | replay rows 0-98 (or 0-499, 0-1999) before predicting; oracle init from the sequence's own end state; mean-state h0 | full 0.68306 / 0.68273 / 0.68304; oracle 0.68337 (+0.0003); rows [99,300) 0.621 → 0.629 oracle |
| position-in-sequence correction | OOF per-position offset or scale, 9 bins, even/odd sequences | −0.0059 / −0.0083; per-sequence quarter bias sd 0.40, share negative 0.47 to 0.50 |
| level rescale | p + (α−1)·causal expanding mean of own p, α 0.8 to 1.5; non-causal α 1.1 to 1.3 | flat or negative: 0.68310 at α=1, 0.68272 at 1.1, 0.68166 at 1.2; non-causal 0.68285 at 1.1 |
| causal second stage on own output | OOF ridge on p, EMA10/100/1000(p), expanding mean | 0.68310 → 0.67506 |
| slow hidden-state features in the head | ridge on h2 (+EMA100/1000/5000 of h2, + cumulative mean, + h1, + x), fitted on valid[192:792], tested held-out | h2 alone 0.6772 (the trained head is 0.6831, the refit costs 0.006); every addition lower: +EMA100 0.670, +EMA1000 0.670, +EMA5000 0.669, +cm 0.671, +h1 0.674, +x 0.675 |
| anchor (epoch-start) deltas and log age | as above | 0.6772 → 0.6739 |
| hold / average within an epoch | see section 1 | 0.602 / 0.649; blends 0.25·hold 0.680, 0.25·cmean 0.682 |
| age-dependent offset or scale | OOF, 9 age buckets | −0.0102 / −0.0110 |
| target reparametrisation | WP of (p0+p1) vs (y0+y1) and (p0−p1) vs (y0−y1) | plain corr: t0 0.540, t1 0.554, sum 0.483, diff 0.564; residual corr(r0, r1) −0.76; the sum is not a near-deterministic spread (var 0.60, between-sequence share 0.22) |
| float16 inputs vs the platform's float32 | full1 on the parquet rows (float32) vs `valid_X.npy` (float16) | 0.68310 vs 0.68310; max input difference 0.0019, max prediction difference 0.022 |
| more ensemble members | best mean by subset size over 8 local checkpoints | 2: 0.68655 (s1+reset2k), 3: 0.68669, 4: 0.68673, 5: 0.68667, all 8: 0.68628; residual corr 0.988 among same-recipe members, 0.982 with dps/all |
| hand-written GRU step (batched MatMul, N members in one graph) | ORT latency, parity 7.8e-7 vs the GRU op | N=1 24.8 µs, N=2 34.2, N=3 42.7, N=4 50.4; slower than the GRU op (13.9) at every N |
| two or three GRU-op members in one graph (`src/ensemble.py`) | ORT latency | 2: 25.9 µs, 3: 36.5 µs (tight loop; the scorer loop adds ×1.13) |

Feature-group ablation at inference (out of distribution, only the order is trusted; *measured*): freezing `a0..a7` at their row-0 value costs 0.063, `i0_v` 0.050 (0.101 when its rows are permuted), all of `i1` 0.034, `i0_p` 0.029, `a5`+`a7` 0.025, `i1_dv` 0.010, `i1_dp` 0.007, `i1_p` 0.002, `i1_v` 0.001.
A linear probe gives i0-only 0.596, i0+i1+a 0.606, so `i1` carries about +0.01 linearly and the model uses it; no sign that `i1` is under-used.
The target's horizon is not visible as a single feature: corr(t0, Δ`i0_p3` over h rows) saturates at −0.15 for h ≥ 300 (part_c).

## 3. Ranked: what to run

Order is expected gain × P over cost.
Priors are my belief about the held-out delta on the full1 recipe.

### 1. Two members in one 256-wide GRU op (block-diagonal weights), no distillation

Mechanism: ORT's cost is the GRU op, not the session call (trivial graph 3.9 µs).
Two 128×2 members become one GRU with hidden 256 whose `weight_hh` is block-diagonal per gate, layer 2's `weight_ih` block-diagonal as well, and a head with `[W_a/2, W_b/2]` and `(b_a+b_b)/2`; the result is exactly the member mean.
Latency (*measured*, tight loop, `part_e.py`): 128×2 13.9 µs, 256×2 21.7 µs, against 25.9 µs for two GRU ops in one graph and 27.9 µs for the two-session ensemble measured earlier.
The scorer loop multiplies by about 1.13 (full1 15.7 vs 13.9), so expect about 24.5 µs locally, under the 28 µs rule; at the baseline's 2.29× platform ratio that is 37 min, at dozken's 3× for a 192-wide graph 48 min.
Gain (*measured*): s1 + reset2k mean 0.68655, +0.0035 over full1, +0.0017 over s1 alone; a loss-diverse member (idea 3) should raise this.
Smallest test: 40 lines that build the fused `nn.GRU`s from two checkpoints, parity against the two ONNX members to 1e-6 on 300 rows, `src/score_holdout.py` for the time, then one submission as a timing probe (the public score should read about 0.656).
Cost: an hour, one slot.
Disproof: scorer time above 28 µs, or the platform above 45 min.
Risk, stated once: the 256 weights are 2.7 MB per call; if the platform core has a 1 to 2 MB L2 the ratio can be worse than 2.29×; that is what the probe measures.
Prior: +0.0035 held-out (measured), P(platform fits) 0.8.

### 2. Read every queued arm by age band

Mechanism: section 1.
Smallest test: `part_m.py` pattern on each `out/<arm>/best.pt` as it lands (15 s each).
Specific predictions: `day_chunk1000` helps on age ≥ 64 or not at all; `morning_ema8` helps evenly; `day_pearson` helps on age 0 to 7 if the scale/offset freedom matters, and is to be judged as an ensemble member (idea 3), not as a single model.
Cost: nothing.
Disproof: n/a, it is a read-out.

### 3. A loss-diverse second member for the fused graph

Mechanism: ensemble gain scales with (1 − residual corr); same-recipe members sit at 0.988 (*measured*), dozken's two-loss pair gave +0.0056 against our +0.0035.
Smallest test: when `day_pearson` finishes, residual corr with full1/s1 and the 2-mean; if the corr is at or below 0.975 and the member is within 0.003 of full1, it is the second block in idea 1.
Cost: nothing beyond the queued arm.
Disproof: residual corr ≥ 0.985, or the 2-mean below 0.6866.
Prior: +0.001 to +0.003 on top of idea 1, P 0.5.

### 4. Longer schedule with EMA weights (queued)

Mechanism: full1's epoch-end evals rise 0.6778, 0.6796, 0.6818, 0.6831 (`logs/colab_full1.log`), +0.0013 in the last epoch while the LR annealed to 1e-8, so the gain is from annealing, not saturation.
Smallest test: `morning_ema8` is queued; read raw and EMA evals at matching batches.
Disproof: epoch 8 not above epoch 4 by the seed sd (0.0012) × 2.
Prior: +0.002 to +0.004, P 0.6.

### 5. Longer truncation inside the epoch structure (chunk 1000, queued; chunk 2000 next)

Mechanism: the information that predicts a long epoch's target is the order flow since the epoch started, 64 to 1,400 rows back, and gradients stop at 500 rows; the state can carry it but is only trained to write it over a 500-row horizon.
dozken's chunk-4000 loss was under level drift across 20,000 rows; with reset 2000 (null for us, so free) chunk 2000 cannot drift beyond the reset.
Smallest test: `day_chunk1000` by age band; if the age ≥ 64 band moves by +0.01 or more, run chunk 2000 with reset 2000.
Disproof: the age ≥ 64 band within ±0.008 (seed spread) of full1.
Prior: +0.000 to +0.004, P 0.35.

### 6. HYPOTHESIS: an auxiliary head for the epoch clock

Mechanism (unmeasured): predict log(rows until the next `i0_p` change) and the sign of that change as auxiliary targets, computed offline, weight 0.1 to 0.2, dropped at export.
This forces the state to hold a running estimate of "how far into a quiet period are we and which way will it break", which is exactly the quantity that separates the 0.80 band from the 0.42 band.
Against it: the linear probe found nothing in anchor deltas and log age that the state does not already hold, and the ridge test of slow state features was negative, so the state may already encode this.
Smallest test: one full-data arm, read by age band.
Disproof: age ≥ 64 band unchanged within the seed spread.
Prior: +0.000 to +0.003, P 0.3.

### 7. Final-submission data: add valid[0:192], up-weight valid (board-only)

Unchanged from `fable-ideas.md` idea 10: +0.001 prior for the 192 sequences; the up-weight is sign-unknown and only the board can score it.

## 4. Not worth running, with the number that kills each

Cold start, learned h0, warm-up replay, time-in-sequence input: oracle warm state +0.0003 (part_d).
Position or age offsets and scales: −0.006 to −0.011 out of fold (part_a, part_l).
Any post-hoc map of the model's own outputs, causal or not: −0.004 to −0.008 (part_b).
Slow EMA or cumulative-mean features of the hidden state in the head: −0.006 to −0.014 against a refit head (part_h2).
Anchor-delta inputs: −0.003 linear (part_k).
Ensembles of more than two same-recipe members: +0.0002 for the third and fourth (part_f).
Hand-written GRU graphs in ORT: 1.8× slower than the GRU op (part_e).
Three members under ORT: 36.5 µs fused, 47.2 µs as one 384-wide op (part_e).
float16 training data: identical WP on float32 inputs (part_f).
Target reparametrisation (sum/diff): the sum is predicted worse than either target (part_b).

## 5. What I still cannot tell you

Whether the test set is in the valid regime or the train regime; the held-out → public offset (−0.026 to −0.036) does not separate sampling noise from regime shift, and only board pairs can.
Whether the 0.42 band is a noise floor or a model floor; nothing I measured moved it, and its seed spread (±0.008) is four times the full metric's.
What the top-8 have; the measured family saturates at about 0.690 held-out (0.687 four-member mean + schedule), and 0.700 needs something outside the family, most likely in the old-epoch band.

## Appendix: raw outputs (M4, 2026-10-21, scratchpad part_*.out)

part_a (positions, full1, 192 seqs): WP by row bin [99,200) 0.619, [200,300) 0.623, [300,500) 0.676, [500,1000) 0.740, [1000,1500) 0.642, [1500,2000) 0.647, [2000,3000) 0.681, [3000,5000) 0.637, [5000,7500) 0.683, [7500,10000) 0.692, [10000,12500) 0.709, [12500,15000) 0.729, [15000,17500) 0.660, [17500,18500) 0.734, [18500,19000) 0.665, [19000,19500) 0.699, [19500,19750) 0.603, [19750,19900) 0.622, [19900,20000) 0.631.
Per-sequence weighted bias by quarter (t0): means −0.034, −0.027, −0.026, +0.004; sd 0.45, 0.41, 0.40, 0.39; share negative 0.49, 0.47, 0.49, 0.50.
OOF position offset 0.68310 → 0.67722; scale → 0.67485.

part_b (level and target structure): corr(y0, y1) scored −0.70; var y0 1.004, y1 1.004, sum 0.599, diff 3.417; plain corr p vs y: t0 0.540, t1 0.554, sum 0.483, diff 0.564; residual corr −0.763; residual sd 0.804, 0.808, sum 0.645, diff 1.477; between-sequence variance share t0 0.077, t1 0.084, sum 0.216, diff 0.057.
α sweep (causal expanding mean): 0.8 0.68153, 0.9 0.68274, 1.0 0.68310, 1.1 0.68272, 1.2 0.68166, 1.3 0.68001, 1.5 0.67528; non-causal 1.1 0.68285, 1.2 0.68195.
OOF own-output second stage 0.67506.

part_c (64 seqs, target semantics): corr(t0, Δi0_p3 over h): h=1 +0.103 (i0_p20), 50 −0.112, 100 −0.125, 300 −0.148, 500 −0.151, 1000 −0.149, 5000 −0.143; current-row top corr i0_p3 0.236, i0_p0 0.197; at t0-change rows corr(Δt0, Δi0_p3) 0.685 with the price feature changing in 99 to 100% of those rows; P(t0 change | i0_p0 unchanged) 0.002; mean |t0| by tail bin 0.62 to 0.74 (no truncation); t0 = 0 share 0.200 overall, 0.219 at the last row.

part_d (cold start, full1): zero init full 0.68310, [99,300) 0.6213; replay 99 0.68306 / 0.6189; replay 500 0.68273 / 0.5996; replay 2000 0.68304 / 0.6299; oracle end-state init 0.68337 / 0.6293; mean-state h0 0.68333 / 0.6164; top-layer |h| 0.412 at step 99 vs 0.459 at 5000, unit-mean corr 0.89.

part_e (ORT latency, 60,000 calls, single thread): trivial 3.9; full1 128×2 13.9; 192×2 16.5; 256×2 21.7; 384×2 47.2; ensemble.py 2 members 25.9; 3 members 36.5; hand-written batched N=1 24.8 (parity 7.8e-7), N=2 34.2, N=3 42.7, N=4 50.4.

part_f (members and ensembles): full1 0.68310, s1 0.68488, s2 0.68256, reset2k 0.68378, h192 0.68120, l3 0.68075, dps 0.67808, all 0.67829; residual corr 0.981 to 0.989; best 2-mean 0.68655 (s1, reset2k), 3 0.68669, 4 0.68673, 5 0.68667; 3 seeds 0.68625; all 8 0.68628; float32 inputs 0.68310.

part_h2 (ridge heads fitted on valid[192:792], λ=1000): h2 0.67719; h1+h2 0.67400; h2+x 0.67542; h2+EMA100 0.67009; h2+EMA1000 0.67036; h2+EMA5000 0.66941; h2+cm 0.67129; h2+3 EMAs 0.66271; everything 0.65506. (The even/odd OOF version on 192 seqs, part_h, is invalid: 96 sequences cannot fit slow features; h2 alone gave 0.659 there.)

part_i (ablation, linear probe, event clock): see section 2 and section 1.

part_j (epochs, full1): P(price change) 0.2953, P(target change) 0.1742, P(target change | no price change) 0.00001; hold 0.60249, cmean 0.64851; by age [0,1) 87,803 rows 0.8032 / hold 0.8032; [1,2) 0.7826 / 0.7764; [2,4) 0.7675 / 0.7509; [4,8) 0.7337 / 0.7011; [8,16) 0.6808 / 0.6116; [16,32) 0.6416 / 0.4902; [32,64) 0.5841 / 0.3362; [64,128) 0.5257 / 0.2406; [128,∞) 0.4164 / 0.1500; age-0 rows of epochs of length [1,2) 0.818, [2,8) 0.810, [8,32) 0.689, [32,64) 0.469, 64+ 0.331.

part_k (anchor probe): model head all 0.68310, age ≥ 64 0.4620, age < 8 0.7797; ridge h2 0.67719 / 0.4519 / 0.7759; h2+lage 0.67699; h2+delta 0.67404; h2+delta+lage 0.67385; h2+x+delta+lage 0.67187; raw x 0.60929; x+diff 0.61716; x+diff+EMA100 0.63126; x+delta+lage 0.61296; all raw 0.63021.

part_l (weight by age and oracle sensitivity): table in section 1; oracle 10% toward truth on age < 8 +0.02284, on 8 ≤ age < 64 +0.02354, on age ≥ 64 +0.02517; 20%: +0.04319, +0.04452, +0.04758; OOF age offset −0.01016, scale −0.01095.

part_m (members by age band [0,1) / [1,8) / [8,64) / [64,∞)): full1 0.8032 / 0.7596 / 0.6401 / 0.4620; s1 0.8018 / 0.7596 / 0.6414 / 0.4711; s2 0.8041 / 0.7607 / 0.6392 / 0.4567; reset2k 0.8028 / 0.7605 / 0.6412 / 0.4628; h192 0.8027 / 0.7584 / 0.6350 / 0.4619; l3 0.8009 / 0.7586 / 0.6386 / 0.4547; dps 0.7958 / 0.7526 / 0.6363 / 0.4591; all 0.7954 / 0.7521 / 0.6326 / 0.4694; mean(s1, reset2k) 0.8050 / 0.7622 / 0.6433 / 0.4698; mean4 0.8066 / 0.7632 / 0.6434 / 0.4671.

part_n (bias stability): weighted bias t0 by age band on even / odd / four 48-sequence quarters: [0,1) −0.018 / −0.020 / −0.010 −0.031 −0.031 −0.009; [64,128) −0.202 / +0.044 / −0.105 −0.005 −0.258 −0.034; [128,∞) −0.369 / +0.215 / +0.125 −0.165 −0.474 +0.209; flat offset on age ≥ 64: +0.05 0.68301, +0.10 0.68260, +0.15 0.68185, +0.20 0.68076.


---

# Round 1 (Fable, 2026-10-20 23:30)

Nothing was trained and nothing longer than 15 s was run.
Every number marked *measured* comes from this Mac (M4), repo `/Users/alexchilton/PycharmProjects/wundernn-connectome`, data `out/data/valid_*.npy`, checkpoints `out/ab_wsoft/best.pt` (held-out 0.67231, mtime 22:26), `out/ab_wmse/best.pt`, `out/ab_mse/best.pt`, `out/mask1/best.pt`, run between 23:00 and 23:25 with the repo `.venv`.
The analysis scripts live in the session scratchpad (`part1.py` to `part7.py`); their raw output is in the appendix.
Everything else is belief and is labelled as a prior.
`docs/prior-art-2.md` was read after the first draft; its findings are cross-checked against our measurements in section 3.

Headline: I found no single structural lever; the two large pools I measured (per-sequence level, +0.042 oracle; mask-confidence split, 0.53 / 0.66 / 0.78) are both already exploited by the model as far as causal information allows.
The gap closes, if it closes, by stacking five or six +0.002 to +0.005 items, each of which must be measured above a noise floor of about ±0.01 per quarter-epoch eval.

## 0. Where we stand and what the measurements say

The full-data soft-mask GRU 128×2 finished at held-out 0.68310 (coordinator, 4 epochs, Colab L4; per-quarter 0.657 / 0.688 / 0.719 / 0.675).
With the held-out to public offset of −0.03, the prize line (0.6702) is about 0.700 held-out and #1 (0.6738) about 0.704, so the remaining gap is +0.017 to +0.021 held-out.
Quarter-epoch evals of that run jump about ±0.01 (coordinator), and the 600-sequence runs swing ±0.005 to ±0.01 between epochs (`logs/ab_wsoft.log`: 0.663, 0.658, 0.669, 0.672, 0.672).
Bootstrap over held-out sequences gives sd 0.029 on 64 sequences and about 0.017 on 192 (*measured*); that is sequence-sampling noise shared by all models scored on the same set, and it is why the same model reads 0.681 on 64 sequences, 0.658 on 128 and 0.672 on 192.
A full-data epoch costs about 5 min on the L4 (coordinator), so every "smallest test" below is a full-data arm, not a 600-sequence run; 600-sequence runs are both slower on MPS and noisier.
Gate for believing any arm: same seed, same epoch, paired on the same 192 sequences, and +0.003 or more, which matches mnemonik-dev's rule on their honest half; weight-averaged checkpoints (idea 2) are the cheapest way to shrink the eval swing below that gate.

The largest pool is the per-sequence level.
Replacing the model's weighted mean prediction per sequence with the true one lifts WP from 0.6812 to 0.7231 (+0.042 on 64 sequences, *measured*).
The level carries 15% of the weighted target variance (per-sequence weighted mean sd 0.50, range −1.4 to +0.6).
The model predicts the per-sequence level with corr 0.82 to 0.87 and the per-2000-row block level with corr 0.77.
The level is a regime quantity, not momentum: block-to-block autocorrelation is 0.24, past-block mean to current block 0.26 to 0.41, first half to second half 0.49, while first-half feature summaries predict the second-half level at out-of-fold corr 0.56.
The regime is readable from `a5` and `a7`: per-sequence mean `a7` has corr +0.77 with the target's quantisation step (median non-zero |Δt0|, p50 0.47) and the step has corr −0.69 with the level; mean `a5` +0.64 and the floor share of `a7` −0.83 (*measured*, 300 sequences; confirms mingda19's claim in prior-art-2).
But the model already uses it: its predicted level tracks the step at corr −0.82 (t0) and −0.86 (t1), tighter than the true level does (−0.67, −0.75), the level residual has corr +0.03 with mean `a7` for t0, and an out-of-fold per-sequence level correction from whole-sequence `a`-feature means lowers WP (0.6582 to 0.6553 on 128 sequences).
Causal volatility, level and range features of the previous block explain corr 0.11 to 0.16 of the block-level residual and move block-level corr from 0.770 to 0.772.
So the +0.042 is mostly irreducible with the information in the features, and "regime features" are not a lever; what is left of it is in the model's own training (ideas 1, 2, 4).

The second pool is the mask.
WP on scored rows by mask-model P tercile is 0.526 / 0.662 / 0.775 (*measured*).
The |y|-weighted model without any mask weighting shows the same split (0.527 / 0.658 / 0.768), and plain MSE 0.506 / 0.645 / 0.760, so the low tercile is inherently hard and the soft mask did not sacrifice it.
Feeding P(scored) or the mask-model state into the main model has no mechanism left; drop it.
Soft mask vs |y|-weighted, paired on 64 sequences: +0.0035 (0.6812 vs 0.6777); on 192 sequences +0.0086; mnemonik-dev report −0.0004 for importance weighting by P(scored | x) on a different model; the board result of `sub_wsoft.zip` settles it, and until then I treat the soft mask as a small real gain.

Dead on arrival, measured on 64 held-out sequences with the 0.672 model:
Output scaling: WP is 0.68116 at every factor from ×0.7 to ×1.1 and 0.68105 at ×1.2; no prediction exceeds ±2 and 3.4% exceed ±1.5; prediction sd is 0.82 vs target sd 0.89, which is the shrinkage an MSE-optimal predictor should have.
Post-hoc recombination of (p0, p1) fitted on 32 sequences and applied to the other 32: cross-target linear −0.0075, cubic −0.0110, tanh −0.0083; a joint-target head is the in-graph version of the same thing and mnemonik-dev measured it at −0.0011.
Input clipping at ±10: the stored features already stop at |x| = 5.2 (the organisers' quantile floor, 0.66% of values sit on it).
Target momentum from past targets: block-to-block corr 0.24.
Regime features from `a5`/`a7` or volatility: see above.

## 1. Ranked ideas

Each idea: mechanism, smallest test, cost, what disproves it, prior.
Priors are my belief about the held-out delta on the current full-data recipe; P is my belief it is positive at all.
Order is expected gain × P divided by cost, with the coordinator's weighting toward loss design, distillation and mask or regime use applied where the measurements allow it; where they do not, I say so.

### 1. Normalise the loss by a global weight mass, not per chunk

Mechanism: `loss_fn` divides each 500-row chunk's weighted squared error by that chunk's own `w.sum()`, so a chunk with weight mass 0.5 pushes as hard as one with mass 73; the global metric counts every row by its weight, so this is the one place our loss still disagrees with METRIC.md.
Per-500-row chunk weight mass (*measured*): true mask on valid, 14.7% of chunks have zero mass, non-zero p10 2.6, p50 20.6, p90 80.8, p99 276; soft mask on train600, p10 0.5, p50 13.9, p90 73.3, so p90/p10 = 145.
This is the global-metric cousin of mingda19's `wp_aligned_loss` finding (prior-art-2 §3), which addressed the old per-sequence metric; under the current pooled metric the correct fix is the opposite direction, a constant denominator.
Smallest test: denominator `B * C * w_bar` with `w_bar` the mean row weight over the training pool computed once at start-up; full-data arm, same seed, compare epoch by epoch to the 0.68310 run.
Cost: one line, one 25-min arm.
Disproof: no epoch beats the matching epoch of the reference by 0.003.
Prior: +0.001 to +0.005, P 0.55.
Adam removes the overall scale, so the effect works only through the relative weight of chunks inside Adam's moving averages; that is why the prior is modest despite the 145× dispersion.

### 2. Weight averaging and a longer schedule

Mechanism: the full run's evals swing ±0.01 per quarter epoch and still rise; an EMA of the weights (decay 0.999 per optimiser step, about 7,700 steps per epoch) or SWA over the last epoch removes the swing, and a longer cosine tail takes the rise.
Smallest test: add `--ema-weights 0.999` to `train_gru.py` (one shadow copy of the parameters, updated after each `opt.step()`), evaluate raw and EMA weights at every eval, 4 and then 8 epochs on the best architecture from tonight.
Cost: 20 lines, 25 to 50 min per arm, zero inference cost.
Disproof: EMA weights never beat raw at the same eval, and epoch 8 is not above epoch 4 by 0.003.
Prior: +0.003 to +0.008, P 0.7; dozken used EMA/SWA on their best model, and this also lowers the noise floor for every other test.

### 3. Chunk 2000 inside the 2000 reset window

Mechanism: with `--state-reset 2000` the state is zeroed every 2000 rows anyway, so chunk 2000 gives full backpropagation inside every window at no change to what inference sees; the target has autocorrelation 0.455 at lag 100 and 0.157 at lag 500, so gradients that cross 500-row boundaries carry signal.
Smallest test: tonight's reset-2000 arm rerun with `--chunk 2000`, same seed (activations 64 × 2000 × 128 per layer, fine on the L4).
Cost: one arm.
Disproof: no gain over the chunk-500 reset-2000 arm at matching epochs.
Prior: +0.000 to +0.005, P 0.5; only if reset 2000 wins tonight.

### 4. Learned initial state, a time-in-sequence input, and warm-up replay

Mechanism: every sequence start and every reset is a cold start from zeros; Q1 scores 0.657 against 0.719 for Q3 on the full model (coordinator) and rows 99 to 300 score 0.661 against 0.70 mid-sequence on the 600-sequence model (*measured*, noisy); a learned h0 and a `log(1 + steps since reset)` input (MarkBrezina shipped a log-time feature, prior-art-2 §2) tell the model how warm its state is, and replaying the 99 warm-up rows once more before the first prediction is free.
Smallest test: replay first, because it needs only `solution.py` and the existing ONNX (score held-out 192 with and without; 99 extra calls per 20,000 rows is 0.5% of time); then h0 as a parameter exported as the init vector in `meta.json`, plus the time input as one more state tensor, one full-data arm with reset 2000.
Cost: trivial for replay, 30 lines for h0 and time, one arm.
Disproof: WP on rows 99 to 300 and on the 300 rows after each reset does not move by 0.01 on 192 sequences.
Prior: +0.000 to +0.003, P 0.45; worth more if reset 2000 wins, because then there are ten cold starts per sequence.

### 5. Distil an ensemble into one graph

Mechanism: a mean of two models trained with different losses gains +0.0028 over the better member (0.6840 vs 0.6812, member prediction corr 0.963, *measured* on 64 sequences); the earlier +0.0003 was for two near-identical models; a 128×2 student trained on the teacher mean keeps 15.4 µs.
Honest ceiling: prior-art-2 §1 reports 0.93 to 0.96 residual correlation across five model families, and our two-model gains are +0.0003 to +0.0028, so the student can recover at most about +0.003 and probably less; a bigger teacher does not raise this, because dozken measured 384×2 at only +0.003 over 192×2 and mnemonik-dev's 1024-state SSM did not decorrelate.
Smallest test: average tonight's five full-data members on held-out first; only if the mean beats the best member by 0.003 or more, label train with the mean (every row) and train a student with loss |y|·P·(p − teacher)² + 0.3·|y|·P·(p − y)².
Cost: one labelling pass over train plus one arm; the members exist anyway.
Disproof: the raw five-member mean gains under +0.003, or the student keeps under half of it.
Prior: +0.001 to +0.003, P 0.5; the coordinator's list ranks this high and the measurements rank it mid, I give the measurements.

### 6. Better mask labels: calibration and a larger, longer-trained mask model

Mechanism: the loss weight is |y| × P, so P's shape sets where capacity goes; the odds-division calibration under-predicts (corrected mean 0.069 vs true share 0.095 on 64 sequences, *measured*), and the mask model was still improving at its last epoch (AUC 0.908, 0.929, 0.933).
A non-causal labeller looked attractive because labelling is offline, but the cheap proxy says no: a centred ±2 moving average of causal P drops AUC from 0.932 to 0.922, and max(causal, future-max) to 0.921, so future rows' P carries little beyond the current P.
Smallest test: Platt scaling on valid[192:392] instead of odds division; hidden 128, 6 epochs; relabel train; one full-data arm with the same seed.
Cost: 1 h mask training on MPS, one labelling pass, one arm.
Disproof: main-model delta under 0.003 at matching epochs.
Prior: +0.000 to +0.003, P 0.45; bidirectional labeller P 0.3, last.

### 7. Pearson loss with running global statistics

Mechanism: the metric is scale- and offset-free; wMSE spends gradient matching absolute level and scale, a Pearson loss with detached running weighted means and variances across chunks and batches does not.
Evidence is mixed: dozken used it but never compared it with |y|-weighted MSE, only with per-chunk Pearson and plain MSE; mnemonik-dev tried a correlation loss and it was negative (prior-art-2 §1); the measured prediction sd 0.82 vs 0.89 is what MSE-optimal shrinkage should produce, so wMSE is not obviously wasting capacity.
Smallest test: `--loss wpearson` with weights |y|·P and running statistics as dozken describe, one arm, same seed.
Cost: 40 lines, one arm.
Disproof: no gain at matching epochs.
Prior: +0.000 to +0.004, P 0.35.

### 8. Lift the latency cap with a native GRU step

Mechanism: base 128×2 costs 15.9 µs of which the arithmetic is a few µs; a C step called through `ctypes` (flat float32 weights, two matvecs and three gates per layer) runs 128×2 in about 2 to 4 µs and leaves room for 384×2 or a three-member ensemble under 28 µs; MarkBrezina shipped a numpy/numba GRU replay and mingda19 shipped numpy weights without ONNX (prior-art-2), so non-ONNX inference is accepted, but a compiled `.so` is not demonstrated by anyone.
Honest ceiling: this changes the ceiling, not the slope, and the slope is flat (384×2 +0.003, ensembles +0.003 to +0.006), so the enabled gain is +0.004 to +0.008 at best.
Smallest test: compile a trivial `.so` on Colab (x86_64 Linux), ship it in a baseline zip that calls it once and falls back to ONNX, read the platform log; only then write the GRU step and check parity against ONNX to 1e-6.
Cost: half a day for the probe, one day for the step; one submission slot for the probe.
Disproof: the probe fails on the platform, or the step is not under 10 µs in Docker on the Mac.
Prior: +0.004 to +0.008 if it works, P(platform allows it) 0.5.

### 9. Auxiliary block-level target

Mechanism: a third head predicts the weighted mean of y over the next 2000 rows (computed offline); it forces the state to hold a level estimate and is dropped at export.
Measured against it: the level residual is uncorrelated with everything I tried (a-features +0.03, causal volatility 0.11 to 0.16), so the aux target would mostly teach the model what it already knows.
Smallest test: one arm with the aux loss at weight 0.2, compare per-sequence level corr (0.82 today) and WP.
Cost: 30 lines, one arm.
Disproof: level corr unchanged and WP delta under 0.003.
Prior: +0.000 to +0.004, P 0.3.

### 10. Final-submission data and the valid regime

Mechanism: train600 and valid are different regimes: 12 of 112 features shift with |t| > 3 between train600 and valid[192:256] against 1 of 112 between two valid slices, and mean |t0| is 0.697 in train600 vs 0.609 in valid[192:] (t = 12.8, *measured*); valid[0:192] is 1.5% more data from the valid regime; whether test is in the valid regime is unknown.
Smallest test: for the final zip only, train the frozen recipe once more with `--add-valid-from 0`; a 2× to 3× up-weight of valid sequences or a final half-epoch on valid[192:] can be tested only on the board, because held-out valid shares the valid regime and would flatter it.
Cost: one arm, one slot each.
Disproof: public delta under +0.003 (invisible for the +192 sequences; possibly visible for the up-weight).
Prior: +0.000 to +0.001 for the extra data; sign unknown and |delta| up to 0.005 for the up-weight.

## 2. Not recommended, with the measurement that kills each

Output scaling or shrinkage: flat from ×0.7 to ×1.1, no predictions beyond ±2.
Post-hoc cross-target, polynomial or tanh maps: −0.0075, −0.0110, −0.0083 out of fold; joint-target head −0.0011 (mnemonik-dev).
Feeding P(scored) or mask-model state into the main model, hard mask, or a floor on P: the low-P tercile is equally hard for the unmasked model (0.527 vs 0.526).
Regime features from `a5`/`a7` means, floor shares, or realised volatility: the model's predicted level already tracks the quantisation step at −0.82/−0.86 and an oracle-fitted correction lowers WP.
Int8: the time is ORT overhead, not arithmetic; 192×2 adds 5 µs for 2.25× the weights.
Target engineering (predict t0 and t0 + t1, or principal components): the head is linear, any linear reparametrisation is already reachable, and the post-hoc test says there is no free gain there.
Extra input clipping: max |x| is 5.2 and that is the organisers' clip.
Second-stage residual or disagreement model (kc5587): with member residual correlation 0.93 to 0.96 there is almost no disagreement to learn from on these features.
Book-structure features: columns are affine-scrambled with sign flips (prior-art-2 §1 and §2); nothing to add.
New architectures on the same inputs: five families converge to 0.93 to 0.96 residual correlation (prior-art-2 §1).

## 3. Cross-check of docs/prior-art-2.md against our measurements

Residual correlation 0.93 to 0.96 across families: consistent with our prediction corr 0.963 between the wsoft and wmse models and ensemble gains of +0.0003 to +0.0028.
Mask-importance weighting −0.0004 (mnemonik-dev): contested by our paired +0.0035 (64 sequences) and +0.0086 (192); the tercile split shows the soft mask gains uniformly (+0.004 mid, +0.007 high) rather than by sacrificing rows; the board decides.
Joint-target head −0.0011 (mnemonik-dev): agrees with our post-hoc −0.0075.
Weight exponent 1.0 beats 0.75 (mnemonik-dev): we already use the metric's |y|.
`wp_aligned_loss` (mingda19): built for the old per-sequence metric; our loss is pooled, which matches the current METRIC.md; the remaining mismatch is the per-chunk denominator (idea 1).
`a5`/`a7` encode the per-sequence quantisation step (mingda19, unverified there): verified here, mean `a7` +0.77, mean `a5` +0.64, floor share of `a7` −0.83; and the step is the level (−0.69); but the model already uses it (section 0).
Quantile floor −5.19934 (MarkBrezina): matches our `a5` floor at −5.199 on 13.0% of rows; six features sit on it.
Platform 1.6 to 2.1× slower (mnemonik-dev): ours 2.29×, keep the 28 µs rule.
Distillation (MehulBandhu, earlier competition): the pattern transfers, the payoff here is bounded by the ensemble gain (idea 5).
Auxiliary future-feature forecasting (kc5587): not measured by anyone on this task; it is a cousin of idea 9 with the same objection, P 0.3.

## 4. Order I would run them

Tonight's arms settle width, depth, reset, diff+proj+skip and EMA features.
Tomorrow, in parallel: idea 1 (loss normalisation, one line) and idea 2 (weight EMA) as full-data arms on the L4; idea 4's replay scored on held-out with the existing ONNX; idea 8's platform probe submitted.
Then idea 3 if reset 2000 won, and idea 4's h0 and time input on the reset model.
Then idea 5 (average tonight's members, distil only if the mean clears +0.003) and idea 6 (Platt plus bigger mask model).
Ideas 7, 9 and 10's up-weight only if the gap is still above 0.01 after that.

## Appendix: raw measurements (M4, 2026-10-20 23:00 to 23:25)

part1 (48 valid sequences, numpy):
A scored t0: mean −0.1503, wmean −0.1575, std 0.876, mean|y| 0.677, P(y=0) 0.141, P(|y|≥2) 0.039, weight share of |y|≥2 0.116.
A unscored t0: mean −0.2182, wmean −0.2332, std 0.790, mean|y| 0.611, P(y=0) 0.184, P(|y|≥2) 0.028, weight share 0.091.
A corr(t0,t1) scored −0.785, all rows −0.750.
B P(t0 changes vs previous row) 0.107; P(scored | changed) 0.080; P(scored | unchanged) 0.086; plateau length median 3, mean 9, p90 15, max 1,400.
B P(scored_t+1 | scored_t) 0.743; P(scored_t+1 | not) 0.024.
C top standardised mean differences scored vs not: i1_v4 −0.40, i1_p13 +0.38, i0_p3 +0.38, i1_p18 −0.37, a5 −0.37, i0_p11 −0.35, a4 −0.33, i0_dp0 +0.33.
C a5: 13.0% of rows sit at the floor −5.199; P(scored | a5 < −3) 0.136 against base 0.085.
C max|x| 5.20; 0.66% of values at |x| > 3; features at the clip: a7, a6, i1_dv3, a0, i1_dv0, a5.
C i1 price features are unchanged from the previous row 87% of the time.
D t0: total weighted variance 1.688, between-sequence share 0.153, per-sequence weighted mean sd 0.501, range −1.42 to +0.60.
E strongest corr(t0, future change of a feature): i0_p0 h=500 −0.276; strongest current-row |corr|: i0_p3 0.319.
F train600_P mean 0.0808 (valid scored share 0.0851); the 0.2302 in `logs/mask1.log` predates the calibration step.
G no drift across the valid index: corr(seq index, per-sequence t0 level) −0.004; scored share 0.0923 (first 192) vs 0.0949 (rest).

part2 (64 held-out sequences, ab_wsoft):
WP 0.68116 (t0 0.671, t1 0.691); WP on all rows ≥ 99: 0.464; WP on unscored rows: 0.424.
Scale sweep ×0.7 to ×1.1: 0.68116; ×1.2 0.68105; ×1.5 0.67937; pred sd 0.822 vs y sd 0.894 (t0); |p| > 2: 0.0000; |p| > 1.5: 0.034.
Post-hoc cross −0.0075, poly3 −0.0110, tanh −0.0083.
WP by position: [99,300) 0.661, [300,1000) 0.699, [1000,2000) 0.619, [2000,5000) 0.682, [5000,10000) 0.707, [10000,15000) 0.700, [15000,20000) 0.664.
Bootstrap sd over sequences 0.029 (64 seqs), about 0.017 for 192.
Per-sequence level corr(y, p): t0 0.819, t1 0.869; sd of y level 0.513 vs p level 0.377 (t0).
Oracle per-sequence level: 0.72308 (+0.042); within-sequence only (both demeaned): 0.66625.
Weighted mean (p − y) by quarter, t0: −0.008, −0.085, −0.198, +0.051.
WP by mask-P tercile: 0.526 / 0.662 / 0.775.
Causal mask AUC 0.932; centred ±2/±5/±10 smoothing of P: 0.922 / 0.914 / 0.902; corrected mean P 0.069 vs true share 0.095.

part3 (train600 vs valid):
12/112 features with |t| > 3 (per-sequence means, 64 vs 64), against 1/112 for valid[1000:1064] vs valid[192:256]; top: i0_p20 +5.0, i0_p0 −4.2, i0_p3 −4.1, i1_v13 +4.0, i1_v18 +4.0.
t0 mean: train600 −0.292 vs valid[192:] −0.249 (t −4.9); mean|t0|: 0.697 vs 0.609 (t +12.8); P(t0 = 0): 0.219 vs 0.219.
Scored share by quarter in valid: 0.095, 0.095, 0.097, 0.097.

part4 (level persistence, 336 valid sequences with all 10 blocks scored; models on 64):
t0: corr(level first half, second half) 0.491; block lag-1 autocorr 0.242, lag-3 0.206; sd of block levels 0.786, of sequence means 0.418, within sequence 0.666.
Past-block mean → current block corr by block: 0.26, 0.30, 0.29, 0.35, 0.35, 0.34, 0.32, 0.31, 0.41.
Ridge on first-half feature summaries → second-half level, out of fold: t0 0.561, t1 0.532; true first-half level → second-half: 0.430, 0.507.
Top summaries: sd i0_p14 0.63, sd i1_p14 0.63, mean i0_p3 0.62; corr(level, i0_p0 end − start) −0.098.
WP by P tercile: wsoft 0.526/0.662/0.775, wmse 0.527/0.658/0.768, mse 0.506/0.645/0.760.
Level corr by quarter (wsoft, t0): 0.739, 0.784, 0.749, 0.819.
Mean of wsoft+wmse 0.6840 (best member 0.6812); member corr 0.963; three-model mean 0.6841.

part5 (128 held-out sequences, 1,068 blocks):
Block-level corr(y, p) 0.770 (t0), 0.773 (t1); residual sd 0.50; out-of-fold corr of causal previous-block volatility/level/range features with the residual 0.106 (t0), 0.156 (t1); level corr after adding them 0.772, 0.777.
Per-500-row chunk weight mass, true mask: 14.7% zero, non-zero p10 2.6, p50 20.6, p90 80.8, p99 276, CV 1.46.
Soft mask on train600: p10 0.5, p50 13.9, p90 73.3, p99 164, p90/p10 145.

part6 (300 valid sequences, a-features vs target quantisation step):
Per-sequence median non-zero |Δt0|: p10 0.125, p50 0.469, p90 0.657; unique t0 values per sequence p50 14.
corr(mean a7, step) +0.77; corr(mean a5, step) +0.64; corr(floor share a7, step) −0.83; corr(mean a4, step) +0.31.
corr(step, level) −0.69; corr(step, mean|t0|) −0.16; corr(mean a7, level) −0.47; corr(mean a5, level) −0.38.
Within-sequence sd of a7 3.09 vs across-sequence sd of its mean 1.06.

part7 (128 held-out sequences, ab_wsoft):
Per-sequence level corr(y, p): t0 0.836, t1 0.874; residual sd 0.28, 0.26.
corr(residual, mean a7): t0 +0.03, t1 +0.22; floor share a7 −0.02, −0.19; mean a5 +0.03, +0.04; step +0.01, +0.01.
corr(level p, step) −0.82 (t0), −0.86 (t1) vs corr(level y, step) −0.67, −0.75.
Out-of-fold ridge of seq-mean a-features → residual: corr −0.18 (t0), +0.25 (t1); WP with that correction 0.65533 vs base 0.65816.
