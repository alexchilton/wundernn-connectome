# Fable plan 2026-10-21 (evening): held-out noise, the public gap, and the next three weeks

Nothing was trained, Colab was not touched, no process was killed, nothing was committed; this is the one file written.
Every number marked *measured* comes from the M4 on 2026-10-21 between 19:10 and 19:40, repo `/Users/alexchilton/PycharmProjects/wundernn-connectome`, checkpoints `out/<arm>/best.pt`, data `out/data/valid_*.npy` (held-out = valid[0:192], 352,804 scored rows).
Scripts and raw outputs: `preds.py`, `analyse.py`, `analyse.out`, `lat.py`, `lat.log` in the session scratchpad `/private/tmp/claude-501/-Users-alexchilton-PycharmProjects-iwb-agent1/b33ff776-f0ca-4139-a365-8ede742e7ae3/scratchpad/`.
Anything not marked *measured* is a prior or a **HYPOTHESIS** and is labelled so.

## Correction to the brief, first

The uploaded "e8 + s1" pair (`out/fz_e8_s1`, 0.68858 held-out, 0.65817 public) is `c1000_lr1e3_e8` + `final_s1`, the OLD chunk-500 / lr 2e-3 / 4-epoch seed-1 member (0.68488), per `out/fz_e8_s1/meta.json` (*measured*).
The pair of the two new-recipe seeds, `c1000_lr1e3_e8` + `c1000_lr1e3_e8_s1`, scores 0.69168 held-out (*measured*) and has never been built or uploaded.
`fz_e8s1_c2000` and `fz_e8s1_c4000` do use the new seed-1 run (`member_best` 0.69161 in both meta files).
All other pair scores in the brief reproduce to 5 decimals from the member predictions (0.69265, 0.69247, 0.69206, 0.68588, 0.68655).

## 1. Is held-out 0-191 too noisy to rank fused pairs 0.004 apart?

Sequence bootstrap (B=1000) on the 192 held-out sequences, all *measured*:

| what is compared | diff | paired SE | z |
|---|---|---|---|
| single score, any model (sampling noise shared by all models) | - | 0.0162 to 0.0171 | - |
| two singles, different seeds, same recipe: s1 vs e8 | +0.00457 | 0.00280 | 1.6 |
| two singles: s1 vs c2000 | +0.00206 | 0.00234 | 0.9 |
| two pairs sharing no member: s1+c2000 vs e8+fs1 | +0.00408 | 0.00179 | 2.3 |
| two pairs sharing no member: e8+s1 vs e8+fs1 | +0.00310 | 0.00129 | 2.4 |
| two pairs sharing one member: s1+c2000 vs e8+s1 | +0.00097 | 0.00088 | 1.1 |
| two pairs sharing one member: s1+c2000 vs s1+c4000 | +0.00019 | 0.00075 | 0.3 |
| two pairs sharing one member: s1+c2000 vs s1+dzish | +0.00059 | 0.00096 | 0.6 |
| pair vs its best member: e8+s1 vs s1 | +0.00007 | 0.00137 | 0.05 |
| 3-mean vs best pair: s1+c2000+c4000 vs s1+c2000 | -0.00023 | 0.00052 | -0.4 |
| 4-mean vs best pair | -0.00047 | 0.00075 | -0.6 |

So: a 0.004 difference between two unrelated pairs is z about 2.3 on 192 sequences, borderline, and the public board then disagreed (section 2).
Differences between pairs that share a member are resolved to about 0.0008, and every such difference measured today is below 0.001, i.e. c2000, c4000 and dzish are interchangeable as the second member (*measured*).
Differences between single members of different seeds have SE 0.0028, so no single-member recipe comparison made so far (chunk 1000 vs 2000 vs 4000, EMA, loss) was above noise; the seed spread s1 vs e8 (0.0046) is larger than any recipe effect.
The metric weight is concentrated: the top 10 of 192 sequences carry 18% of the weight (p50 2,062, max 11,168 per sequence), which is why the SE is large for 352,804 rows.
The t0/t1 split is not a noise estimate: t1 is 0.003 to 0.006 above t0 for every model (target difficulty, stable), but the two targets do give two semi-independent reads of a paired difference (s1+c2000 vs e8+fs1: t0 +0.0059, t1 +0.0022, same sign).

Selection optimism, *measured* from the logs (best eval minus the mean of the last 8 evals of the same run): e8 +0.0019, s1 +0.0013, c2000 +0.0021, c4000 +0.0004, fs1 +0.0019, fs2 +0.0014, full1 +0.0011; the EMA run dzish128 +0.0002.
`best.pt` is the max of 20 to 40 evals on the same 192 sequences that report the score, so every held-out number in the brief, including the fused pairs, carries about +0.001 to +0.002 that the board never sees.
EMA weights remove most of it (dzish128 eval sd 0.0002 vs 0.0005 to 0.0016 for raw-weight runs).

Enlarging the held-out set: 384 sequences gives SE x 0.71, 768 gives x 0.50.
Cost of 192 more: 1.6% of the 12,288-sequence pool, prior -0.0005 to -0.001 held-out (the mirror of the +0.001 prior for adding valid[0:192]), **HYPOTHESIS**, plus 2x eval time per eval.
It only pays for the pair-vs-pair question if the public gap is sampling noise; if it is a regime effect (section 2), a bigger valid-regime held-out does not help and the board is the only ranker.
Recommendation: do not enlarge yet; first spend three uploads (section 3, item 1) to find out which it is.
If the board confirms the held-out order, enlarge to 384 for all new runs (`--n-valid 384 --add-valid-from 384`) and keep the member selection free of the reporting set by using `--ema-weights 0.999` and the last checkpoint.

## 2. Why e8+fs1 beat s1+c2000 on the board while losing held-out

Held-out s1+c2000 - e8+fs1 = +0.0041 (paired SE 0.0018); public -0.0011.
If the public set is all 1,970 test sequences, its paired SE is about 0.0018 x sqrt(192/1970) = 0.0006, and the combined disagreement is 0.0052 against SE 0.0019, z 2.8; sampling noise alone has p about 0.005.
If the public set is a 30% subset, the public SE is about 0.001 and z 2.5.
So pure noise is not excluded, but it is unlikely, and there is a second signal: the held-out to public offset by recipe is baseline -0.026, full1 (4 ep) -0.029, fs1+fs2 -0.030, fs1+r2k -0.031, e8+fs1 (8+4 ep) -0.030, s1+c2000 (8+8 ep) -0.036.
**HYPOTHESIS (unverified)**: valid[192:] is in the training pool and shares its regime with the held-out valid[0:192] (12 of 112 feature means shift between train and valid, fable-ideas round 1); more epochs fit that regime better, which raises held-out more than it raises a test set that is partly or wholly in the train regime.
Alternative **HYPOTHESIS**: chunk 2000 specifically generalises worse (it is the one member that differs in gradient horizon); the s1+c4000 upload separates this from the first hypothesis.
Selection optimism does not explain it: it is +0.0013 to +0.0021 for every member in both pairs.
Residual correlation does not explain it: all members sit at 0.984 to 0.993 (*measured*), the e8-fs1 pair at 0.985 and the s1-c2000 pair at 0.988, so neither pair is more diverse.

Free test, three uploads, predictions stated before the fact:

| upload | held-out | if offset -0.030 (noise story) | if offset -0.036 (regime story) |
|---|---|---|---|
| e8 + s1 (new seeds, chunk 1000, 8+8 ep; not yet fused) | 0.69168 | 0.6617 | 0.6557 |
| s1 + c4000 (zip ready, `out/fz_e8s1_c4000`) | 0.69247 | 0.6625 | 0.6565 |
| s1 + dzish (zip ready, `out/fz_e8s1_dzish`) | 0.69206 | 0.6621 | 0.6561 |

If all three land at or below 0.658, the new-recipe family has a bigger offset and the held-out must not be used to rank it against the old members; the board decides the final pair.
If they land at 0.660 or above, the c2000 result was noise and the held-out order stands.

## 3. The next three weeks, ranked by expected public gain per GPU hour

Costs (*measured*, L4, `logs/colab_*.log`): one 128x2 member, 8 epochs chunk 1000 or 2000, 57 min; chunk 4000 6 epochs, 39 min; dzish (diff + proj 128 + pearson + reset + EMA) 5 epochs, 62 min; 192x2 4 epochs, 43 min.
The platform cap in M4 tight-loop terms is about 26 to 28 us (256-wide: 19.4 us tight loop, 24.5 to 25.9 us scorer loop, 46 to 48 min on the platform).

1. Upload the three existing pairs above: 0 GPU-h, 3 of today's 5 uploads; expected +0.000 to +0.004 public over 0.65817, and it settles section 2, which every later decision depends on.
2. Decide the ranker from the result: board-ranked (regime story) or held-out-ranked with 384 sequences and EMA members (noise story); 0 GPU-h.
3. More seeds of the best recipe, fuse the best two by the chosen ranker: 1 GPU-h per seed (chunk 4000 at 0.65 GPU-h is as good, *measured* within 0.0006 as a second member); the seed spread (0.0046 between the two seeds we have) is the largest effect measured today, so 3 to 4 more seeds give a best pair about +0.001 to +0.002 held-out; expected public +0.0005 to +0.001 after the winner's curse, prior P 0.6.
4. Final members retrained with `--add-valid-from 0` and `--ema-weights 0.999`, recipe frozen, no held-out left: 2 GPU-h; prior +0.0005 to +0.001 public, P 0.6; do this last, in the final week, on the recipe the board picked.
5. The dz128 arms already queued: 0 extra GPU-h; read them as members, not singles; the only unmeasured thing they bring is wd 0.01 and dropout, which may change the offset (**HYPOTHESIS**), so one of them should go to the board paired with s1.
6. A 192x2 single on the new recipe (chunk 1000, lr 1e-3, 8 ep): 1.4 GPU-h; dozken +0.003 for 2x width, our one measurement -0.002 on the old recipe (inside seed noise); a 192 single runs at 13.0 us tight loop (*measured*), about 30 min on the platform, and the pair-over-best-member gain is only +0.0001 to +0.0016 (*measured*), so a good 192 single could match the 128 pair with a 2x safety margin on time; prior +0.000 to +0.002, P 0.35.
7. Longer schedule or warm restart (`--init` from best.pt, 3 to 4 epochs at lr 3e-4): 0.5 GPU-h per member; the last two epochs of every 8-epoch run are flat within 0.0003 to 0.0007 at near-zero LR, so the 4 to 8 gain came with the LR drop, not from steps; prior +0.000 to +0.001, P 0.4.

Dead by measurement, do not spend GPU on them:

- A third member: +0.0002 (fable-ideas part_f), today -0.0002 (3-mean) and -0.0005 (4-mean) against the best pair, paired SE 0.0005 to 0.0008.
- 3 x 96 fused (288 wide): fits only with proj-64 inputs (20.1 us tight loop vs 23.9 us with raw 112 inputs; 256-wide 19.4; 320-wide 27.7; 384-wide 35.7 to 45.7, *measured* today, random weights, `lat.log`), and the third member is worth nothing (above), while 96 units per member is an unmeasured cost.
- Int8: `onnxruntime.quantization.quantize_dynamic` leaves the GRU op and its float32 initializers untouched (file 2.72 MB before and after, GRU weight dtype float32), interleaved timing 16.6 to 19.9 us vs 18.7 to 19.4 us fp32, no reliable gain (*measured*).
- Distillation: the pair is +0.00007 over its best member for e8+s1 and +0.0010 to +0.0016 for the other pairs, so a student has at most +0.001 to learn; the fusion already ships both members at no distillation cost.
- Loss- or input-diverse second member: dzish (pearson loss, diff, proj, reset 2000) has residual correlation 0.984 to 0.988 with the wmse members, the same as seed pairs (0.986 to 0.993); as a second member it is within 0.0006 of c2000 and c4000.
- Longer chunks: c2000 0.68955 and c4000 0.68889 vs c1000 seeds 0.68703 and 0.69161, inside the 0.0028 single-member SE; as second members all three are equal within 0.0006.
- Bigger fused widths: 320 wide is 27.7 us tight loop, at the platform cap; 384 is over it.

What this plan does not contain: a measured lever that closes the gap to the prize line (0.6702 public, fable-ideas round 2); the family is at about 0.692 held-out and 0.657 to 0.658 public, and items 3, 4 and 6 together are worth about +0.002 to +0.004 public on priors.
The single biggest unknown is the offset, which costs nothing to measure and is item 1.

## 4. Code: what looks wrong or leaves a cheap gain

1. `train_gru.py` line 94, the `--state-reset` guard: `A and B or C` parses as `(A and B) or C`, so `--chunk 4000 --state-reset 2000` passes the assert, but the training loop resets only at chunk starts (every 4,000 rows) while `evaluate()` and `solution.py` reset every 2,000; a silent train/inference mismatch. No run so far hit it (the c4000 runs have reset 0). Fix: require `chunk <= state_reset and state_reset % chunk == 0`.
2. Optimiser steps per epoch scale with 1/chunk: 3,840 at chunk 1000, 1,920 at 2000, 960 at 4000 (192 batches x 20,000/chunk); each chunk-length comparison is also a step-count and effective-batch comparison, and `day_chain4.sh` compensates with 6 epochs, not 4x. Not a bug, but say it when reading the chunk arms.
3. `best.pt` is selected on the reporting set (section 1): +0.0004 to +0.0021 optimism per run. Cheapest honest alternative: `--ema-weights 0.999` and take the last checkpoint (dzish128 shows +0.0002); or select on valid[0:96] and report on [96:192] at the price of SE x 1.4.
4. In EMA runs `evaluate()` scores only the EMA weights, so EMA vs raw is never measured in the same run (`morning_ema8` was read as an EMA null, but it is confounded with lr 2e-3 x 8 epochs and chunk 500). One extra eval of `model` beside `ema_model.module` costs 20 s per eval and settles it.
5. No guard that `--add-valid-from >= --n-valid`; `--n-valid 384` with the current `--add-valid-from 192` would train on valid[192:384] and report on it. One assert.
6. `OneCycleLR` default `final_div_factor 1e4` sends the LR to 4e-9; the last 12 to 15% of each run moves the eval by less than 0.0003 (*measured*, every 8-epoch log), so `--epochs 7` with `final_div_factor 100` gives the same model 12% cheaper (**HYPOTHESIS**, unmeasured).
7. The batch loader is synchronous: 64 x 20,000 x 112 float16 (287 MB) is read from the mmap and run through `make_feats` on the CPU before each GPU step; the epoch is 2.2 s per batch on the L4 for a model whose GPU work is small. A one-batch prefetch thread is 20 lines and could cut epoch time by 20 to 40% (**HYPOTHESIS**); worth it only if more than 5 runs remain.
8. `pearson_loss` clamps `p` to [-2, 2] before the correlation, so rows with |p| > 2 get gradient only from the mse term; harmless today (no prediction exceeds 2, fable-ideas round 1).
9. `fuse.py` and `solution.py` are consistent with training on resets and the `prev_x` state; parity 0 to 1e-6 on every fused model (`logs/fuse_*.log`); nothing to fix.

## Appendix: raw numbers (M4, 2026-10-21, scratchpad `analyse.out` and `lat.log`)

Member WP, held-out 192 (t0 / t1): e8 0.68703 (0.68352 / 0.69055); s1 0.69161 (0.69051 / 0.69271); c2000 0.68955 (0.68665 / 0.69244); c4000 0.68889 (0.68688 / 0.69089); dzish 0.68706 (0.68451 / 0.68961); fs1 0.68488; fs2 0.68256; r2k 0.68378; full1 0.68310.
Pairs: e8+s1 0.69168; s1+c2000 0.69265; s1+c4000 0.69247; s1+dzish 0.69206; e8+c2000 0.69011; c2000+c4000 0.69033; e8+dzish 0.68981; e8+fs1 0.68858; s1+fs1 0.69078; c4000+fs1 0.68896; fs1+fs2 0.68588; fs1+r2k 0.68655; s1+c2000+c4000 0.69242; e8+s1+c2000+c4000 0.69219; 5-mean 0.69233.
Split halves (even / odd sequences): e8+s1 0.68479 / 0.69919; s1+c2000 0.68657 / 0.69916; s1+c4000 0.68631 / 0.69899; s1+dzish 0.68553 / 0.69915; fs1+fs2 0.67849 / 0.69420; fs1+r2k 0.67920 / 0.69472 (the pair order holds on both halves).
Quarters of 48 sequences, s1+c2000: 0.73336, 0.66132, 0.66369, 0.70799.
Residual correlation matrix (weighted, scored rows, mean of t0/t1): e8-s1 0.986, e8-c2000 0.990, e8-c4000 0.988, e8-dzish 0.984, s1-c2000 0.988, s1-c4000 0.988, s1-dzish 0.985, c2000-c4000 0.993, c4000-dzish 0.988, old members (fs1, fs2, r2k) 0.985 to 0.989 with everything.
ORT tight loop, 2-layer GRU step, random weights, single thread, IOBinding, best of 3 x 40,000 calls: 128 10.2 us (0.76 MB weights); 192 13.0 (1.59 MB); 256 19.4 (2.70 MB); 288 23.9 (3.37 MB); 320 27.7 (4.12 MB); 352 28.7 (4.93 MB, runs 28.7 to 36.4); 384 35.7 (5.82 MB, runs 35.7 to 45.7); 256 with 64-wide input 17.1; 288 with 64-wide input 20.1.
Epoch times (L4): c1000 e8 427 to 550 s; c2000 430 s; c4000 380 to 420 s; dzish128 740 s; h192 625 to 657 s.
