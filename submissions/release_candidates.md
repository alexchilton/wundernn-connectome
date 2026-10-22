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
