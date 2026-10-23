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
