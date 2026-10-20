# Ideas to close the gap to the prize line (Fable, 2026-10-20 23:30)

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
