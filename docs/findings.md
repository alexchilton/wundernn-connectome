# Findings ledger — Alpha Connectome

One consolidated record of submissions, measured results, open hypotheses, and where the detail lives.
Built from: fable-ideas.md, fable-plan-2026-10-21.md, final-recipe.md, position-blend-2026-10-22.md, HANDOFF.md, tutorial.md section 8.
docs/prior-art.md was not read for this ledger; it stays as reference.

## 1. Board results

Newest first. Held-out is valid[0:191] unless stated. Platform time is given where a source states it.

| date | submission | held-out | public | platform time | source |
|---|---|---|---|---|---|
| 2026-10-22 | e8 (`out/c1000_lr1e3_e8`) + dz96p32_s1 (`out/dz96p32_s1`), fused (`fz_e8_dz96s1`) — **today's best** | 0.69043 | **0.65924** | 47m52s | HANDOFF.md |
| 2026-10-22 | e8 + dz128, fused | — | TIMEOUT | 1.29x platform ratio | HANDOFF.md |
| 2026-10-22 | dz128 single | 0.68655 | 0.65566 | — | HANDOFF.md |
| 2026-10-21 | e8 + old final_s1, fused (uploaded and originally logged as "e8 + s1") | 0.68858 | 0.65817 | — | HANDOFF.md (corrected label); fable-plan-2026-10-21.md (correction); tutorial.md §8 (same 0.68858, logged "pending" before the public score landed) |
| 2026-10-21 | seed 1 + reset2k, fused ("fs1+r2k") | 0.68655 | 0.65599 | 47m06s | tutorial.md §8; fable-plan-2026-10-21.md appendix ("fs1+r2k 0.68655") |
| 2026-10-21 | two seeds of full1, fused ("fs1+fs2") | 0.68588 | 0.65625 | 46m12s | tutorial.md §8; fable-plan-2026-10-21.md appendix ("fs1+fs2 0.68588") |
| 2026-10-21 or earlier | GRU soft mask, full training set, 4 epochs ("full1") | 0.68310 | 0.65392 | — | tutorial.md §8; HANDOFF.md |
| earlier | GRU, \|y\| x P(scored) soft mask, 600 train seqs | 0.67231 (epoch 4) | 0.63693 | — | tutorial.md §8; HANDOFF.md |
| earlier | GRU, \|y\|-weighted MSE, 600 train seqs | 0.66197 (epoch 4) | 0.63279 | — | tutorial.md §8; HANDOFF.md |
| earlier | GRU, plain MSE, 600 train seqs | 0.64906 (epoch 4) | 0.61346 | — | tutorial.md §8; HANDOFF.md |
| 2026-10-20 | provided baseline GRU 128x2 | 0.61987 (full valid 0.617052, equals organisers' figure) | 0.59353 | 26m36s | tutorial.md §8; HANDOFF.md |

Notes on the table:

The "e8 + old final_s1" row is a correction, not a new result.
fable-plan-2026-10-21.md found that the pair uploaded and logged as "e8 + s1" (`out/fz_e8_s1`) is actually `c1000_lr1e3_e8` + the OLD chunk-500/lr-2e-3/4-epoch `final_s1` (member 0.68488), per `out/fz_e8_s1/meta.json`, not the new-recipe seed-1 pair.
The true new-recipe pair, `c1000_lr1e3_e8` + `c1000_lr1e3_e8_s1`, scores 0.69168 held-out (fable-plan-2026-10-21.md) and has never been built or uploaded.

"dz128 single" (0.68655 -> 0.65566, HANDOFF.md) and "seed 1 + reset2k fused" (0.68655 -> 0.65599, tutorial.md section 8) share the identical held-out score to 5 decimals but are different models with different public scores.
This reads as a coincidence, not a data conflict: position-blend-2026-10-22.md independently lists `dz128_s0` at 0.6866 (4dp, matches 0.68655 rounded), confirming dz128 is its own checkpoint, separate from the fs1+r2k fusion.

Offset (held-out minus public) is about -0.031 for dz pairs, range -0.026 to -0.036 overall (HANDOFF.md).
Timing limit: 1.2x the baseline's platform/M4 ratio fits at 47m52s, 1.29x timed out (HANDOFF.md).

## 2. Measured, kept

Each line is a change that measurably helped, with the number and its source.

- |y|-weighting the loss beats plain MSE: 0.66197 vs 0.64906 held-out, 0.63279 vs 0.61346 public (tutorial.md section 8).
- Soft mask (|y| x P(scored)) beats |y|-weighted MSE: 0.67231 vs 0.66197 held-out (epoch 4), 0.63693 vs 0.63279 public (tutorial.md section 8); paired on 64 sequences +0.0035 (0.6812 vs 0.6777), on 192 sequences +0.0086 (fable-ideas.md round 1, part2/section 0).
- Full training set beats 600 training sequences on the same soft-mask recipe: 0.68310 vs 0.67231 held-out (tutorial.md section 8).
- Two-seed fusion beats the single full1 model: 0.68588 vs 0.68310 held-out (+0.00278), 0.65625 vs 0.65392 public (+0.00233) (tutorial.md section 8).
- Best 2-member fused ensemble beats every tested single member: s1+c2000 0.69265 held-out vs best single member s1 0.69161 (+0.00104) (fable-plan-2026-10-21.md appendix).
- s1 + reset2k fused: +0.0035 over full1 (0.68655 vs 0.6831), +0.0017 over s1 alone (fable-ideas.md round 2 headline and section 3 idea 1).
- dz96p32_s1 as the second member beats the old final_s1 as the second member, both fused with e8: 0.69043 vs 0.68858 held-out (+0.00185), 0.65924 vs 0.65817 public (+0.00107) (HANDOFF.md).
- Fusing two members into one block-diagonal GRU (hidden 256) reproduces the exact member mean and is faster than running two separate ONNX sessions: 21.7 us vs 25.9 us tight loop (fable-ideas.md round 2 section 3 idea 1, part_e).

## 3. Measured, null or worse

- Distillation, teacher mean(e8, old final_s1), --distill 0.5, plain 128 on the e8 recipe (tools/day_chain7.sh): student s0 single best 0.68821 / last 0.68646 vs e8 0.68703 / 0.68542 (+0.001, at the ceiling); fused with dz96p32_s1 it scores 0.69031 (best) / 0.68964 (last) vs e8 + dz96p32_s1 0.69043: the single-model gain does not survive pairing (logs/fuse_stu_dz96s1.log).

Each tested idea that did not help, with the number and source.

- Position-weighted blend, e8 + dz128_s0, 2000-step segments: cross-fitted gain -0.00054 (position-blend-2026-10-22.md, logs/pos_blend_e8_dz128.log).
- Position-weighted blend, 5000-step segments (quarters): cross-fitted gain +0.00041, below the pair SE of about 0.001; not pursued (position-blend-2026-10-22.md, logs/pos_blend_e8_dz128_q.log).
- Per-quarter greedy teacher selection vs global greedy, cross-fitted: -0.00090 (even->odd), -0.00078 (odd->even); not pursued (position-blend-2026-10-22.md, logs/teacher_qmix.log).
- Greedy member selection does not transfer between halves: one half picks 2 members, the other 6, and one half loses to the board pair (position-blend-2026-10-22.md).
- State reset 2000: full_reset2k 0.68378 vs full1 0.6831, below the gate, out (final-recipe.md; HANDOFF.md "tried overnight").
- Diff + proj + skip inputs: full_dps 0.67807 vs full1 0.6831, out (final-recipe.md; HANDOFF.md).
- EMA(10,100,1000) features added to diff+proj+skip: full_all 0.67829 vs full_dps 0.67807, out (final-recipe.md; HANDOFF.md, "+ EMA(10,100,1000)" 0.67829).
- Time-in-sequence feature built on diff+proj+skip: full_dpst run produced no usable number ("None") vs full_dps 0.67807, out (final-recipe.md; HANDOFF.md, "Time-feature arm dropped").
- Width/depth: 192x2 (h192) 0.68121, 128x3 (l3) 0.68075, both vs full1 0.6831, out (final-recipe.md; HANDOFF.md).
- 12-epoch run on 600 sequences: peaks 0.66944 at epoch 4, then declines (overfits) (HANDOFF.md).
- 128x3 on 600 sequences: 0.66804 vs 128x2 0.67231 (HANDOFF.md).
- `--loss-norm batch` (morning_norm): best 0.68364, null (HANDOFF.md).
- `--ema-weights 0.999`, 8 epochs (morning_ema8): best 0.68431, epoch-end evals flat 0.678-0.683 from epoch 1; longer schedule does not help (HANDOFF.md).
- Cold start / warm state (replay rows 0-98, 0-499, 0-1999; oracle init; mean-state h0): full metric 0.68306 / 0.68273 / 0.68304; oracle init only +0.0003 (0.68337) (fable-ideas.md round 2 section 2, part_d).
- Position-in-sequence correction (OOF per-position offset or scale, 9 bins, even/odd): -0.0059 / -0.0083 (fable-ideas.md round 2 section 2, part_a/part_l).
- Level rescale via causal expanding mean (alpha 0.8-1.5): flat or negative, 0.68310 at alpha=1 down to 0.68166 at alpha=1.2 (fable-ideas.md round 2 section 2, part_b).
- Causal second-stage ridge on the model's own output: 0.68310 -> 0.67506 (fable-ideas.md round 2 section 2, part_b).
- Slow hidden-state features (h2 + EMA100/1000/5000, cumulative mean, h1, x) in a refit head: h2 alone 0.6772 (refit costs 0.006 vs the trained head's 0.6831); every addition lower, down to 0.669-0.675 (fable-ideas.md round 2 section 2, part_h2).
- Anchor (epoch-start) deltas and log age added to the h2 ridge: 0.6772 -> 0.6739 (fable-ideas.md round 2 section 2, part_k).
- Hold (freeze prediction within an epoch) / causal within-epoch mean: 0.602 / 0.649 vs the model's 0.68310 overall (fable-ideas.md round 2 section 2, part_j).
- Age-dependent offset or scale (OOF, 9 age buckets): -0.0102 / -0.0110 (fable-ideas.md round 2 section 2, part_l).
- Target reparametrisation (sum/diff of t0,t1): plain corr sum 0.483 vs t0 0.540 / t1 0.554; the sum is predicted worse than either target (fable-ideas.md round 2 section 2, part_b).
- float16 vs float32 inputs: identical WP, 0.68310 vs 0.68310 (fable-ideas.md round 2 section 2, part_f).
- More than 2 ensemble members: 2-mean 0.68655, 3-mean 0.68669 (+0.0002 for the third), 4-mean 0.68673, 5-mean 0.68667, all 8 0.68628 (fable-ideas.md round 2 section 2, part_f); on the new recipe, 3-mean vs best pair -0.00023 (paired SE 0.00052, z -0.4), 4-mean vs best pair -0.00047 (SE 0.00075, z -0.6) (fable-plan-2026-10-21.md section 1 bootstrap table).
- Hand-written batched GRU step in ORT: slower than the GRU op at every N (N=1 24.8 us vs GRU-op 13.9 us; N=2 34.2, N=3 42.7, N=4 50.4) (fable-ideas.md round 2 section 2, part_e).
- Int8 quantization: `quantize_dynamic` leaves the GRU op and float32 initializers untouched (file size unchanged, 2.72 MB); interleaved timing 16.6-19.9 us vs 18.7-19.4 us fp32, no reliable gain (fable-plan-2026-10-21.md section 3).
- Distillation: the fused pair beats its best member by only +0.00007 (e8+s1) to +0.0016 (other pairs), so a student has at most +0.001-0.0016 to recover (fable-plan-2026-10-21.md section 3).
- Loss- or input-diverse second member (dzish: pearson loss, diff, proj, reset 2000): residual correlation 0.984-0.988 with the wmse members, same range as same-recipe seed pairs (0.986-0.993); as a second member it is within 0.0006 of c2000/c4000 (fable-plan-2026-10-21.md section 3).
- Longer chunks (c2000, c4000) as single-member recipe changes: inside the 0.0028 single-member seed SE vs the c1000 seeds; as second members, c2000/c4000/dzish are all equal within 0.0006 (fable-plan-2026-10-21.md section 1, section 3).
- Bigger fused widths: 320-wide 27.7 us tight loop, at the platform cap; 384-wide 35.7 us, over it (fable-plan-2026-10-21.md section 3, appendix).
- Three GRU-op members fused in one graph: 36.5 us, slower than two members' 25.9 us and not worth the third member's gain (fable-ideas.md round 2 section 2, part_e).
- OOF per-sequence level correction from whole-sequence `a`-feature means: lowers WP, 0.6582 -> 0.6553 on 128 sequences (fable-ideas.md round 1 section 0).
- Output scaling/shrinkage, x0.7 to x1.1: WP flat at 0.68116 (ab_wsoft model); x1.2 0.68105; no prediction exceeds +/-2 (fable-ideas.md round 1 section 0, part2).
- Post-hoc cross-target recombination of (p0,p1): linear -0.0075, cubic -0.0110, tanh -0.0083 (fable-ideas.md round 1 section 0, part2).

## 4. Facts about the instrument

Held-out noise (sequence bootstrap, B=1000, 192 held-out sequences, all *measured*, fable-plan-2026-10-21.md section 1):

| comparison | diff | paired SE |
|---|---|---|
| single score, any model (shared sampling noise) | - | 0.0162 to 0.0171 |
| two singles, same recipe, different seeds (s1 vs e8) | +0.00457 | 0.00280 |
| two singles (s1 vs c2000) | +0.00206 | 0.00234 |
| two pairs sharing no member (s1+c2000 vs e8+fs1) | +0.00408 | 0.00179 |
| two pairs sharing no member (e8+s1 vs e8+fs1) | +0.00310 | 0.00129 |
| two pairs sharing one member (s1+c2000 vs e8+s1) | +0.00097 | 0.00088 |
| two pairs sharing one member (s1+c2000 vs s1+c4000) | +0.00019 | 0.00075 |
| two pairs sharing one member (s1+c2000 vs s1+dzish) | +0.00059 | 0.00096 |
| pair vs its best member (e8+s1 vs s1) | +0.00007 | 0.00137 |
| 3-mean vs best pair | -0.00023 | 0.00052 |
| 4-mean vs best pair | -0.00047 | 0.00075 |

Bootstrap sd over held-out sequences (earlier measurement): sd 0.029 on 64 sequences, about 0.017 on 192 (fable-ideas.md round 1, part2).
A 0.004 difference between two unrelated pairs is z about 2.3 on 192 sequences, borderline; the public board then disagreed with it (fable-plan-2026-10-21.md section 2).
Enlarging held-out: 384 sequences gives SE x 0.71, 768 gives SE x 0.50; cost of 192 more sequences is 1.6% of the 12,288-sequence pool, prior -0.0005 to -0.001 held-out, HYPOTHESIS, plus 2x eval time per eval (fable-plan-2026-10-21.md section 1).

Selection optimism (best eval minus the mean of the last 8 evals of the same run, *measured* from logs): e8 +0.0019, s1 +0.0013, c2000 +0.0021, c4000 +0.0004, fs1 +0.0019, fs2 +0.0014, full1 +0.0011; EMA run dzish128 +0.0002 (fable-plan-2026-10-21.md section 1).
`best.pt` is the max of 20-40 evals on the same 192 sequences, so every held-out number carries about +0.001 to +0.002 the board never sees (fable-plan-2026-10-21.md section 1).

Held-out-to-public offset by recipe (*measured*): baseline -0.026, full1 (4 ep) -0.029, fs1+fs2 -0.030, fs1+r2k -0.031, e8+fs1 (8+4 ep) -0.030, s1+c2000 (8+8 ep) -0.036 (fable-plan-2026-10-21.md section 2).
HANDOFF.md states the same range more broadly: offset about -0.031 for dz pairs, range -0.026 to -0.036 overall.

Timing limits: 1.2x the baseline's platform ratio fits at 47m52s; 1.29x timed out (HANDOFF.md).
Platform cap in M4 tight-loop terms is about 26 to 28 us; 256-wide fuses at 19.4 us tight loop, 24.5 to 25.9 us scorer loop, 46 to 48 min on the platform (fable-plan-2026-10-21.md section 3).
Baseline platform run: 26m36s on the platform vs 11m38s on the M4, 2.29x slower; the 60-minute limit is about 26 local minutes, about 42 us per call; <=28 us is treated as the safe line because platform speed drifts (tutorial.md section 8).

us/call numbers (two separate measurement sessions; note the disagreement):
- fable-plan-2026-10-21.md appendix, ORT tight loop, random weights, single thread, IOBinding, best of 3x40,000 calls: 128 10.2 us (0.76 MB); 192 13.0 us (1.59 MB); 256 19.4 us (2.70 MB); 288 23.9 us (3.37 MB); 320 27.7 us (4.12 MB); 352 28.7 us (4.93 MB, runs 28.7-36.4); 384 35.7 us (5.82 MB, runs 35.7-45.7); 256 with 64-wide input 17.1 us; 288 with 64-wide input 20.1 us.
- fable-ideas.md round 2 part_e, ORT latency, 60,000 calls, single thread: trivial graph 3.9 us; full1 128x2 13.9 us; 192x2 16.5 us; 256x2 21.7 us; 384x2 47.2 us; ensemble.py 2 members 25.9 us; 3 members 36.5 us; hand-written batched N=1 24.8 us (parity 7.8e-7), N=2 34.2, N=3 42.7, N=4 50.4.
- These two sessions give different numbers for the same widths (e.g. 256-wide 19.4 us vs 256x2 21.7 us); both are labelled *measured* in their own doc; neither source reconciles the difference.
- The scorer loop multiplies the tight-loop number by about 1.13 (full1 15.7 vs 13.9 us tight loop) (fable-ideas.md round 2 section 3 idea 1).

## 5. Open hypotheses

- HYPOTHESIS: an auxiliary head predicting log(rows to next `i0_p` change) and the sign of that change forces the state to hold a running "how far into a quiet period" estimate, which is the quantity separating the 0.80 age-0 band from the 0.42 age>=64 band. Cheapest test: one full-data arm, read by age band (part_m.py pattern); disproof is the age>=64 band unchanged within the seed spread (+/-0.008) (fable-ideas.md round 2 section 3 idea 6).
- HYPOTHESIS: an auxiliary block-level target (weighted mean of y over the next 2000 rows) forces a level estimate into the state. Cheapest test: one arm at aux weight 0.2, compare per-sequence level corr (0.82 today) and WP; disproof is level corr unchanged and WP delta under 0.003 (fable-ideas.md round 1 section 1 idea 9).
- HYPOTHESIS: Pearson loss with running global (detached) statistics spends less gradient on scale/offset than weighted MSE. Cheapest test: `--loss wpearson` with weights |y|*P, one arm, same seed; disproof is no gain at matching epochs (fable-ideas.md round 1 section 1 idea 7).
- HYPOTHESIS: `--epochs 7` with `OneCycleLR final_div_factor 100` (instead of the default 1e4) gives the same model about 12% cheaper, since the last 12-15% of each run moves the eval by under 0.0003. Cheapest test: one 7-epoch arm vs the reference 8-epoch run, compare the final eval (fable-plan-2026-10-21.md section 4 item 6).
- HYPOTHESIS: a one-batch prefetch thread on the synchronous batch loader (2.2s/batch on the L4) could cut epoch time 20-40%. Cheapest test: implement the 20-line prefetch thread, time one epoch against the current log; only worth doing if more than 5 runs remain (fable-plan-2026-10-21.md section 4 item 7).
- HYPOTHESIS: the dz128/dz96 recipe's weight decay (0.01) and dropout change the held-out-to-public offset versus the wmse family. Cheapest test: pair a dz arm with s1 on the board and compare its offset to the -0.026 to -0.036 range already measured (fable-plan-2026-10-21.md section 3 item 5).
- HYPOTHESIS: up-weighting valid sequences 2-3x in the final training run changes the public score; sign unknown, held-out cannot read it because held-out valid shares the valid regime. Cheapest test: one board-only submission with the up-weight, final zip only (fable-ideas.md round 1 section 1 idea 10; fable-plan-2026-10-21.md section 4 item 4).
- HYPOTHESIS: the plain mean of all 13 checkpoints (0.69111 held-out) is a more robust distillation teacher than any 2-model or per-quarter-selected teacher, since per-quarter and per-half greedy selection do not transfer across halves. Cheapest test: train a student against the 13-checkpoint mean, compare held-out to the chain-7 student trained against the e8+final_s1 mean teacher (position-blend-2026-10-22.md; HANDOFF.md "Running/next").
- HYPOTHESIS: a bidirectional GRU teacher (chain 8, not yet written) adds genuine architectural diversity over the existing wmse/dzish family, whose residual correlation sits at 0.984-0.993 regardless of loss or chunk recipe. Cheapest test: build it, score residual correlation against existing members using the same <=0.975 bar used to judge day_pearson as an ensemble candidate (HANDOFF.md "Running/next"; fable-ideas.md round 2 section 3 idea 3).
- HYPOTHESIS (unverified, partially tested): the held-out-to-public gap is a regime effect (valid[192:] shares the training pool's regime, so more epochs overfit held-out more than a partly-out-of-regime public set), not pure sampling noise. The three-upload test designed to separate this (fable-plan-2026-10-21.md section 2) has landed results in HANDOFF.md (offset range -0.026 to -0.036, not flat), consistent with the regime story but not conclusively settled; cheapest remaining test is enlarging held-out to 384 sequences only once the board confirms the held-out ranking order (fable-plan-2026-10-21.md section 1, section 2).

## 6. Where the detail lives

- `docs/archive/fable-ideas.md` — two rounds of Fable's idea lists: round 1 (per-sequence level pool, mask tercile analysis, loss-normalisation and weight-EMA ideas, prior-art cross-check) and round 2 (the epoch-age error map, 9 levers measured dead, the block-diagonal GRU fusion idea, and the full appendix of raw part_a-part_n outputs).
- `docs/archive/fable-plan-2026-10-21.md` — the "e8+s1" mislabel correction, the held-out-noise bootstrap table, the regime-vs-noise investigation into the held-out/public gap, the ranked 3-week GPU-hour plan, and nine numbered code-level findings in `train_gru.py`/`solution.py`/`fuse.py`.
- `docs/archive/final-recipe.md` — the automatic final-recipe selection trace (reset 2000, diff+proj+skip, EMA features, time feature, width/depth, all rejected against full1) and the resulting flags (`--hidden 128 --layers 2`).
- `docs/archive/position-blend-2026-10-22.md` — per-quarter WP tables across four checkpoints, the position-weighted blend results (both null), and the teacher-ensemble/greedy-selection results used to pick the 13-checkpoint mean as the distillation teacher.
