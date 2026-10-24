# Release candidates (2026-10-22), upload in this order, one per day

All are a plain partner fused with dz96p32_s1 (out/dz96p32_s1/latest.pt), the dz member of the board pair e8 + dz96p32_s1
(held-out 0.69043, public 0.65924, 47m52s). Timing on the M4 matches that pair (0.93-1.10x, logs/release_candidates.log);
parity of every fused graph with its member mean is 0.0.
Predicted public = held-out + offset; the offset range is a HYPOTHESIS from four past uploads (docs/findings.md).

| zip | plain partner | held-out | predicted public |
|---|---|---|---|
| release_cand_1.zip | distill_a05_s1/best.pt (student, e8 recipe) | 0.69108 | ~0.6601 |
| release_cand_2.zip | c2000_lr1e3_e8/best.pt | 0.69118 | 0.6552-0.6601 |
| release_cand_3.zip | c1000_lr1e3_e8_s1/best.pt | 0.69272 | 0.6567-0.6587 |
| release_cand_4.zip | distill_a05_s1/latest.pt (student, last epoch) | 0.69060 | ~0.6596 |
| release_cand_5.zip | c4000_lr1e3_s1/best.pt | 0.69065 | 0.6547-0.6597 |

## release_cand_6.zip (2026-10-23, upload 2026-10-24) — Fable plan item 2

e8 `out/c1000_lr1e3_e8/latest.pt` (last epoch, not best.pt) + dz96p32_s1 latest, fused (`out/fz_e8last_dz96s1`).
Held-out 0.68984 (Fable, /tmp/e8_latest_holdout.npy), -0.0006 vs the board pair e8 best + dz96p32_s1 (0.69043, public 0.65924); paired SE about 0.0004.
Parity 0.0 on 2 x 2500 held-out rows; solution.py identical to the board pair's.
Timing on the M4, interleaved with the board pair (logs/release_cand_6.log): 26.9-29.8 vs 25.7-28.9 us/call, same graph, noise.
Pre-registered (docs/fable-plan-2026-10-23.md item 2): public >= 0.6588 says held-out best.pt selection is worthless on the board; public <= 0.6580 says it is not.

## release_cand_7.zip (2026-10-23) — Fable plan item 5 (board test of the regime)

e8 recipe + EMA 0.999, seed 0, trained WITHOUT valid[192:] (`out/e8noval_s0/latest.pt`, last epoch; pool 10,607 train sequences) + dz96p32_s1 latest, fused (`out/fz_e8noval_dz96s1`).
Member held-out: 0.68727 at the last epoch (best 0.68752 at ep5) vs 0.68896 for e8 trained with valid[192:] (-0.0017, inside Fable's predicted -0.002 to -0.004).
Parity 0.0; solution.py identical to the board pair's; timing 26.1-29.3 vs 25.6-26.7 us/call (logs/release_cand_7.log), same graph.
Pre-registered (docs/fable-plan-2026-10-23.md item 5): public >= 0.6590 says the test set is train-like; public <= 0.6570 says valid-like or noise.
Weak test if the public sample is small (SE ~0.0008 at 394 sequences): the two predictions are 2.5 SE apart.
Upload after release_cand_6.

## Board readings (uploaded 2026-10-24 09:23, Alex)

- release_cand_6.zip: public **0.65921** (platform 50m7s, 634 MB). Pre-registered >= 0.6588: held-out best.pt selection is worthless on the board (last epoch ties the board pair 0.65924, -0.00003).
- release_cand_7.zip: public **0.65815** (51m7s, 629 MB). Between the thresholds (0.6570 < 0.65815 < 0.6590): inconclusive. Board cost of dropping valid[192:] -0.0011 vs held-out -0.0017. Keep valid[192:] in training.
- Best public unchanged: 0.65924 (e8 best + dz96p32_s1).

## release_cand_8 (built 2026-10-24 18:50, not uploaded)

- release_cand_8.zip = out/fz_quiet_dz96s1: src/fuse.py out/quiet_s0/best.pt + out/dz96p32_s1/latest.pt (quiet_s0 = e8 recipe + --aux 2 --aux-kind quiet, branch arm-four; fuse.py slices the aux head to [:2]). solution.py byte-identical to the board pair's.
- Held-out (src/score_holdout.py, 192 seqs, solution.py path): **0.69088** vs board pair fz_e8_dz96s1 0.69043 (+0.00045); 26.7 us/call on the M4 (board pair class: 47m52s on the platform).
- Pre-registered reading: public >= 0.6602 = the quiet aux head transfers (+0.001 over 0.65924); <= 0.6587 = worse; between = inconclusive (pair noise ~0.001; held-out gain +0.00045 is inside it).
- Not chosen: fz_s13_dz96s1 (student13_s0 + dz96p32_s1) held-out 0.69070, 25.5 us/call, although student13_s0 alone is 0.68967 vs quiet_s0 0.68834.

## Board reading (uploaded 2026-10-24 18:42, Alex)

- release_cand_8.zip: public **0.65936** (platform 48m21s, 629 MB). Pre-registered band 0.6587-0.6602: **inconclusive**. New best public by +0.00012 over 0.65924 (held-out predicted +0.00045; offset 0.69088 -> 0.65936 = -0.0315, in the usual -0.026 to -0.036 range).
