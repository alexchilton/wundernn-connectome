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
