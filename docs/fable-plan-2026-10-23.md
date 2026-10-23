# Fable plan 2026-10-23: the partner scan is noise, fix the instrument, then test the regime on the board

Nothing was trained, nothing committed, nothing deleted, no upload made.
Three things were computed on the M4 today (2026-10-23 07:50 to 08:05, repo `.venv`, data `out/data/valid_*.npy`, predictions `out/preds/*.npy`); scripts and raw output are in `/tmp/wnn_board6.py`, `/tmp/wnn_board6.log`, `/tmp/wnn_extra_pairs.py`, `/tmp/wnn_extra_pairs.log`, `/tmp/wnn_pred_ckpt.py`, and the new held-out predictions of `out/c1000_lr1e3_e8/latest.pt` are at `/tmp/e8_latest_holdout.npy` (not copied into the repo).
Everything marked *measured* is from those runs or from a cited file; everything else is labelled HYPOTHESIS.

One documentation error to fix first, because it would produce a silent null:
`docs/archive/fable-ideas.md` idea 10 and `docs/archive/fable-plan-2026-10-21.md` item 4 say to train the final members with `--add-valid-from 0` to include all of valid.
`src/train_gru.py` line 42 documents 0 as "off" and line 97 (`if a.add_valid_from:`) reads it that way, so `--add-valid-from 0` trains on train only.
Including valid[0:192] needs a code change (for example accept `--add-valid-from -1` or a separate flag).

## 0. What the six board pairs say (measured today)

Six pairs share `dz96p32_s1`; only the partner differs.
Recomputed from `out/preds` on valid[0:192], sequence bootstrap B=2000, paired against the e8 pair:

| partner | held-out | public | offset | held-out diff vs e8 pair | paired SE | z | partner single | residual corr with dz |
|---|---|---|---|---|---|---|---|---|
| e8 best.pt | 0.69043 | 0.65924 | -0.0312 | 0 | - | - | 0.68704 | 0.9816 |
| student s1 latest.pt | 0.69061 | 0.65863 | -0.0320 | +0.00017 | 0.00084 | +0.2 | 0.68857 | 0.9854 |
| c2000 best.pt | 0.69118 | 0.65835 | -0.0328 | +0.00075 | 0.00091 | +0.8 | 0.68955 | 0.9844 |
| student s1 best.pt | 0.69109 | 0.65754 | -0.0336 | +0.00065 | 0.00109 | +0.6 | 0.68958 | 0.9853 |
| e8s1 best.pt | 0.69272 | 0.65752 | -0.0352 | +0.00229 | 0.00143 | +1.6 | 0.69161 | 0.9821 |
| c4000_s1 best.pt | 0.69066 | 0.65707 | -0.0336 | +0.00022 | 0.00131 | +0.2 | 0.68875 | 0.9858 |

- The observed held-out range of the six is 0.00229; the mean range of six draws under sequence resampling is 0.00294.
  The spread the five release candidates were ranked by is the size of the noise.
- Under resampling, Spearman(held-out, public) has mean -0.29 and sd 0.44 (28% of draws are positive).
  The -0.60 point estimate is not a stable anti-correlation; it is one draw from a wide distribution centred near zero.
- The OLS slope of offset on held-out over the six points is -1.42.
  Pure winner's curse (true quality equal, held-out noise picks the order) predicts exactly -1.
  Six points cannot separate -1.42 from -1, so the data do not require a regime story.
  "Offset grows with held-out" is what selection noise looks like.
- Mean paired SE for pairs sharing dz on 192 sequences is 0.00112.
  The same quantity on the public sample is 0.00035 if the sample is all 1,970 test sequences, 0.00049 for 985, 0.00078 for 394.
  The public range of the six is 0.00217, so the public differences are probably partly real, and held-out cannot see them at any affordable size.
- The one-parameter calibration the six points allow is the offset: mean -0.0331, sd 0.0014.
  Predicted public = held-out - 0.033 +/- 0.0014 is wider than the whole candidate range.
  Nothing with more parameters can be fitted to six points with 0.0011 noise each.

Other pairs scored today from existing predictions: e8 + dz96p32_s0 0.68840 (dz96p32_s0 single 0.68329 vs dz96p32_s1 0.68739, a 0.0041 seed gap between two EMA runs of one recipe); dz_s0 + dz_s1 0.68678; e8 + dz_s0 + dz_s1 0.68945; e8 latest.pt single 0.68548, e8 latest.pt + dz96p32_s1 0.68984 (-0.00059 vs the e8 best.pt pair, paired SE 0.00043, z -1.4).

Consequence: the partner question is closed.
No partner pair differs from the e8 pair by more than 1.6 SE on held-out, the public differences among them are below what any local instrument can resolve, and the private board is a separate dataset with its own draw of the same size.
Do not upload another partner.

## 1. The plan, ranked

Ranking is by what each item changes about a decision, divided by its cost.
Items 1 to 3 cost no GPU.

### 1. Close the partner scan and retire best.pt selection (0 GPU-h)

Mechanism: `best.pt` is the max of 20 to 40 evaluations on the same 192 sequences that report the score; the pick is biased upward by the noise it was selected on, and that bias does not travel to the board.
Evidence: selection optimism measured +0.0004 to +0.0021 per run (`docs/findings.md` section 4); the only within-run board A/B we have, student s1 latest.pt vs best.pt, went +0.0011 on public while held-out said -0.0005; section 0 above.
Smallest test: none needed for the first half (a read-out); the second half is item 2.
Cost: 0.
Disproof: a partner pair beating the e8 pair by 3 paired SE (about 0.004) on held-out; none exists among 20 fused pairs built so far.

### 2. Upload e8 latest.pt + dz96p32_s1: the best.pt-vs-last test on the board (0 GPU-h, 1 upload)

Mechanism: same run, same recipe, same partner, only the checkpoint choice differs, so the public difference measures what held-out selection is worth on the board.
Evidence: held-out 0.68984 vs 0.69043 (-0.0006, z -1.4, *measured* today); student last beat student best by +0.0011 public (`submissions/release_candidates.md`); e8 best.pt optimism +0.0019 (`docs/findings.md` section 4).
Smallest test: `python src/fuse.py out/fz_e8last_dz96s1 out/c1000_lr1e3_e8/latest.pt out/dz96p32_s1/latest.pt`, time it (expect 25 to 27 us/call like `logs/release_candidates.log`), upload.
Pre-registered reading: public >= 0.6588 means selection on held-out is worth nothing on the board and every later member ships as EMA or last checkpoint; public <= 0.6580 means best.pt carried real signal and the student result was noise.
Between those, one more within-run pair (c2000 latest vs best) decides.
Cost: 2 minutes, 1 upload.
Disproof: public <= 0.6580.

### 3. Write the fold-instrument change, 15 lines in train_gru.py (0 GPU-h; used by items 5, 7, 8, 9)

Mechanism: a recipe comparison needs the same recipe evaluated on more sequences than 192, and the sequences must not be in any fold's training pool.
`--valid-slice a:b` holds out valid[a:b] and adds valid[:a] and valid[b:] to the pool; two runs with 0:936 and 936:1873 give predictions for all 1,873 valid sequences from models that never saw them.
Evidence: single-model paired SE is 0.0028 on 192 sequences and scales with 1/sqrt(n) (`docs/findings.md` section 4); on 1,873 it is about 0.0009, enough to see 0.002 to 0.003 effects at 2 to 3 SE, which 192 sequences never can.
Smallest test: the change plus `--max-batches 1` smoke on the M4.
Cost: 15 lines, no GPU.
Disproof: n/a (tooling).
Also add the two guards from `docs/archive/fable-plan-2026-10-21.md` section 4 items 1 and 5 while in the file (state-reset multiple of chunk; held-out never in the pool).

### 4. Train-regime held-out: hold 192 train sequences out of one e8-recipe run and keep every epoch-end checkpoint (1 GPU-h)

Mechanism: valid[192:] is 13.7% of the training pool (1,681 of 12,288 sequences) and is the same regime as the reporting set; if more epochs fit that regime, a train-regime held-out peaks earlier than valid[0:192] does.
Evidence that the regimes differ: 12 of 112 feature means shift with |t| > 3 between train600 and valid[192:256] against 1 of 112 between two valid slices; mean |t0| 0.697 in train600 vs 0.609 in valid[192:], t = 12.8 (`docs/archive/fable-ideas.md` idea 10, *measured*).
Whether the test set is train-like or valid-like is unknown (`docs/archive/fable-plan-2026-10-21.md` section 2), and the six board points cannot tell (section 0).
Smallest test: `--train-holdout 192` (exclude train[:192] from the pool; about 10 lines next to item 3), copy latest.pt to ep{k}.pt at every epoch end, run the e8 recipe once with `--ema-weights 0.999`, then score the 8 checkpoints on valid[0:192] with the true mask and on train[:192] with P(scored) as weight (train has no `is_scored`; P is what the loss already uses) and with plain |y| weights.
Cost: 1 GPU-h on the L4 (57 min per 8-epoch run, `logs/colab_c1000_lr1e3_e8.log`), 20 minutes of M4 scoring.
Pre-registered reading: both curves peak at the same epoch and rank the 8 checkpoints the same way (Spearman >= 0.8) means the gap to the board is sampling noise and mask, and the regime story dies locally; the train curve peaking 2 or more epochs earlier, or declining while valid rises, means held-out overfits the valid regime and the proxy becomes the mean of the two read-outs.
Disproof of the regime story: Spearman >= 0.8 and the same peak epoch.

### 5. Board test of the regime: one partner trained without valid[192:] (1 GPU-h, 1 upload)

Mechanism: if the public test is train-like, removing the valid-regime 13.7% of the pool lowers held-out and does not lower public; if it is valid-like or the gap is noise, public falls by about 0.7 of the held-out fall (the 0.70 to 0.73 transfer ratio measured by mnemonik-dev, `docs/prior-art.md` line 217, and our own 0.65625 - 0.65392 vs 0.68588 - 0.68310 = 0.84).
This is the one data lever that changes a large share of the training set, so it is the one most likely to move public by more than 0.002 in either direction; its sign is the point of the test.
Evidence: section 0 (held-out cannot rank; only the board can), `docs/archive/fable-ideas.md` idea 10 (sign unknown, |delta| up to 0.005 HYPOTHESIS).
Smallest test: e8 recipe, seed 0, `--ema-weights 0.999`, no `--add-valid-from`, ship latest.pt fused with dz96p32_s1.
Pre-registered predictions: held-out drops 0.002 to 0.004 (HYPOTHESIS: 13.7% less data and less valid-regime data); public >= 0.6590 says train-like, public <= 0.6570 says valid-like or noise.
If the public sample is small (SE 0.0008 at 394 sequences) the two predictions are 2.5 SE apart and the test is weak; say so when reading it.
Cost: 1 GPU-h, 1 upload.
Disproof (of "test is train-like"): public <= 0.6570.
Follow-up only if train-like: a second partner with valid down-weighted to 0.5 (the `rw` multiplier in train_gru.py line 237 is where a per-source weight goes), and the final members (item 6) trained on the pool the board picked.
Follow-up only if valid-like: the mirror arm, valid up-weighted 3x, same cost.

### 6. Final members: two runs on the pool items 4 and 5 picked, EMA weights, last checkpoint, no held-out selection (2 GPU-h, 1 to 2 uploads)

Mechanism: ship what the board said generalises, without the +0.001 to +0.002 of held-out selection, and with valid[0:192] added only if item 5 said valid-like (code change: see the `--add-valid-from 0` note at the top).
Evidence: items 2, 4, 5; `docs/archive/fable-plan-2026-10-21.md` section 3 item 4 (prior +0.0005 to +0.001, P 0.6, HYPOTHESIS).
Smallest test: the two runs, fused, timed, uploaded; a second seed of each as a backup candidate if GPU remains (2 more GPU-h).
Cost: 2 to 4 GPU-h, 1 to 2 uploads.
Disproof: public below the e8 + dz96p32_s1 pair by more than 0.001; then the e8 pair stays the final pick.

### 7. Capacity, measured on the fold instrument: 192x2 plain, new recipe (2 GPU-h with 2 folds)

Mechanism: a 192 single runs at 13.0 us tight loop (`docs/findings.md` section 4), about 30 minutes on the platform, so it could replace the pair with a 2x time margin; a 192 + 96 fused graph (288 wide) does not fit, since the 224-wide pair is already at 47m52s and 1.29x timed out.
To replace the pair, a 192 single must beat a 128 single by the pair's +0.0035 (`docs/findings.md` section 2).
Evidence: the only width measurement we have is on the old recipe at 4 epochs (h192 0.68121 vs full1 0.68310, inside the 0.0028 seed SE); dozken reports +0.003 for 2x width (`docs/tutorial.md`, not reproduced).
A size arm has never been run on the current recipe; "not capacity-bound" has no run behind it.
Smallest test: 192x2, chunk 1000, lr 1e-3, 8 epochs, EMA, 2 folds vs the 128x2 reference on the same 2 folds (the reference is 2 more GPU-h the first time; it is reused by items 8 and 9).
Cost: 2 GPU-h plus the 2 GPU-h reference, 0 uploads unless it wins.
Disproof: 192 single below 128 single + 0.003 on 1,873 pooled sequences (SE about 0.0009).

### 8. HYPOTHESIS: bidirectional or wide teacher, student on the e8 recipe (3 GPU-h with the instrument)

What is different from the measured distillation null: the teacher in `tools/day_chain7.sh` was mean(e8, final_s1), +0.001 to +0.0016 above the student's own ceiling, so the student had nothing to learn (`docs/findings.md` section 3).
A non-causal teacher is a different mechanism: it replaces y with E[y | past, future], a lower-variance target for a loss whose weights are |y| and whose targets are constant inside epochs 99.999% of the time (`docs/archive/fable-ideas.md` section 1).
Against it: the student cannot represent what it cannot see, so the gain is label denoising only, and the ceiling of that is unmeasured.
Smallest test: bidirectional GRU 128x2 trained 4 epochs (not exported; about 1.5 GPU-h), its predictions on all pool sequences to Drive (212M rows x 2 x float16 = 0.85 GB), a student via `--teacher-preds` (not written; `--teacher` runs the teacher online and cannot run a non-causal one), `--distill 0.5`, judged on the fold instrument.
Cost: 3 GPU-h and the `--teacher-preds` option (about 30 lines).
Disproof: student single below the e8-recipe single + 0.002 on the fold instrument.

### 9. HYPOTHESIS: auxiliary epoch-clock head, read by age band (1 GPU-h)

Unchanged from `docs/archive/fable-ideas.md` idea 6: predict log(rows to the next `i0_p` change) and its sign as auxiliary targets at weight 0.1 to 0.2, dropped at export; the age >= 64 band carries 22% of the weight at WP 0.42 to 0.53 and no trained change has moved it (seed spread 0.008).
Smallest test: one arm, read by age band with the part_m pattern; the band must move by at least 0.016 (2 seed sd) to count.
Cost: 1 GPU-h, 30 lines.
Disproof: age >= 64 band within 0.016 of e8.
Ranked last because the linear probes found nothing the state does not already hold (part_k), P about 0.3.

### Not recommended

- Another partner scan, another seed of the partner, or a third member: section 0 and `docs/findings.md` section 3 (3-mean -0.0002, 4-mean -0.0005).
- Enlarging held-out to 384 for ranking existing checkpoints: the existing checkpoints trained on valid[192:], so no larger valid held-out exists for them; the instrument only helps new runs (item 3).
- Regime-weighted or subset held-out calibrated on the six board points: six points with 0.0011 noise each cannot validate any reweighting; the only test of a proxy is prospective (two uploads where the proxy and the old held-out disagree most), and no candidate pair disagrees by more than noise.
- Leaderboard probing (encoding a test-set statistic in the predictions): `wnn_connectome_starterpack/docs/rules.md` "No messing with the infra" and disqualification; do not.
- 288-wide fused graphs: 1.29x timed out, the 224-wide pair is at 47m52s of 60.

## 2. The three questions, answered

### (1) Fixing the validation proxy

Held-out valid[0:192] is not broken; it is too small by a factor of about 10 for the question it was asked.
Paired SE for two pairs sharing a member is 0.0011, for two singles 0.0028; the candidates differ by 0.0002 to 0.0023.
No split, subset or weighting of 192 sequences changes that, and the six board points cannot calibrate one (section 0).
The cheapest experiment that tells us whether a proxy works is prospective and costs uploads, not GPU: pre-register the proxy's prediction for a pair and upload it.
Item 2 is that experiment for the proxy "last or EMA checkpoint instead of best.pt", with the prediction written down above; items 4 and 5 are that experiment for "held-out plus a train-regime read-out", one local (1 GPU-h) and one on the board (1 GPU-h, 1 upload).
For recipe questions, the fold instrument (item 3) gives SE about 0.0009 on 1,873 sequences at 2 GPU-h per arm, and the final model is a full-data retrain of the winning recipe.
Partner ranking at the 0.001 level has no proxy and needs none: the private board draws the same noise again.

### (2) What besides the partner can move public by more than 0.002

Measured ceiling of the ensemble family: residual correlation 0.982 to 0.986 between every partner and dz (section 0), pair over best member +0.0001 to +0.0016 (`docs/findings.md` section 3).
Inside that family the only unmeasured lever with a mechanism that touches more than a few percent of the gradient is the training pool itself (item 5: 13.7% of the pool is valid-regime; direction unknown; board-only).
Training length and regularisation: the offset-grows-with-held-out pattern is explained by selection noise with slope -1 (section 0), so there is no measured evidence that longer training or less regularisation costs public; item 4 is the local test of that claim, and epoch count follows from it.
Loss: `--loss pearson` with `--mse-weight 0.1` is already the dz member; wpearson on the plain member is an open HYPOTHESIS (`docs/findings.md` section 5) and is a fold-instrument arm if GPU remains.
Inputs and features: diff, proj, skip, EMA, time, slow-state and anchor features are measured nulls (`docs/findings.md` section 3); nothing new to try there.
Capacity: untested on the current recipe (item 7); the time cap allows a 192 single, not a wider pair.
Outside the family: items 8 and 9, both HYPOTHESIS, both with named disproofs.
Honest expectation: the measured family sits at 0.690 held-out and 0.659 public; the prize line was 0.6702 public on 2026-10-21 (`docs/archive/fable-ideas.md` line 15, not re-checked today).
Items 5 to 9 are each worth 0 to +0.003 on priors; none is worth +0.011.

### (3) The 23 days and one upload per day

Every upload is a candidate for the final re-score set (`rules.md`: the final pick is from previously submitted and scored solutions) or an experiment that changes a decision; nothing else is uploaded.
Dates are approximate; GPU-h are L4 hours from measured run times.

| day | upload | GPU | decides |
|---|---|---|---|
| Oct 24 | item 2: e8 latest + dz96p32_s1 | 0 | checkpoint policy |
| Oct 24-25 | - | items 3, 4 (1 GPU-h), 5 arm (1 GPU-h) | code + regime arms running |
| Oct 26 | item 5: train-only partner + dz96p32_s1 | - | test regime, final pool |
| Oct 27 | item 5 follow-up if a direction is clear (down- or up-weighted valid) | 1 | pool weight |
| Oct 27-31 | - | 2-fold reference (2) + item 7 (2) + item 8 or 9 (1 to 3) | recipe |
| Nov 1-5 | one upload per recipe winner, full-data retrain, EMA, last checkpoint | 1 to 2 each | candidates |
| Nov 6-10 | item 6: final members on the chosen pool, two seeds | 4 | final candidates |
| Nov 11-13 | backup pair (second seed) and any timing re-check | 0 | margin |
| Nov 14-15 | nothing new; choose the re-score set | 0 | final |

Total: about 12 to 16 GPU-h, 8 to 10 uploads out of 23 available.
Final selection rule, written now so it is not fitted later: among uploads built from EMA or last checkpoints, take the highest public score; add the e8 + dz96p32_s1 pair (0.65924) as the second pick; a third pick, if allowed, is the best-public candidate from the other pool choice (train-only or valid-weighted), so the private set hedges the regime question.
Check the number of allowed picks when the organisers announce it; the rules say "some".

## 3. Raw outputs

`/tmp/wnn_board6.log` (M4, 2026-10-23 07:55):

```
pair held-out WP (recomputed) | public | offset
  e8        0.69043 | 0.65924 | -0.03119
  stu_last  0.69061 | 0.65863 | -0.03198
  c2000     0.69118 | 0.65835 | -0.03283
  stu_best  0.69109 | 0.65754 | -0.03355
  e8s1      0.69272 | 0.65752 | -0.03520
  c4000     0.69066 | 0.65707 | -0.03359
paired bootstrap SE vs e8 pair (B=2000): stu_last +0.00017 SE 0.00084; c2000 +0.00075 SE 0.00091; stu_best +0.00065 SE 0.00109; e8s1 +0.00229 SE 0.00143; c4000 +0.00022 SE 0.00131
range of the six under resampling: mean 0.00294, observed 0.00229
Spearman(bootstrap draw, point-estimate order) mean 0.59, share < 0.5: 0.32
Spearman(bootstrap draw, public) mean -0.29, sd 0.44, share > 0: 0.28; point estimate -0.60
halves even/odd: e8 0.68229/0.69944; stu_last 0.68280/0.69918; c2000 0.68377/0.69927; stu_best 0.68414/0.69858; e8s1 0.68652/0.69933; c4000 0.68408/0.69774
residual corr with dz96p32_s1: e8 0.9816; stu_last 0.9854; c2000 0.9844; stu_best 0.9853; e8s1 0.9821; c4000 0.9858
mean paired SE on 192 seqs 0.00112; implied on 394: 0.00078, 985: 0.00049, 1970: 0.00035; observed public range 0.00217
```

`/tmp/wnn_extra_pairs.log`:

```
dz96p32_s0 single 0.68329  dz96p32_s1 single 0.68739
e8 + dz96p32_s0 0.68840
dz96p32_s0 + dz96p32_s1 0.68678
e8 + dz_s0 + dz_s1 0.68945
e8 latest.pt single 0.68548; + dz96p32_s1 0.68988 (bootstrap script: 0.68984, diff -0.00059, SE 0.00043, z -1.38); + dz_s0 0.68781
OLS slope offset ~ held-out -1.42; slope public ~ held-out -0.42
```

Platform timing reference: 47m52s for 39.4M test rows is 72.9 us per row, so the whole 1,970-sequence test set is run; the public score is on an unknown sample of it (`rules.md`), and the private board is a separate hidden dataset.
