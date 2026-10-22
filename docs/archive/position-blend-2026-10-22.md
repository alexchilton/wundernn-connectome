# Position-weighted blend and per-quarter scores (2026-10-22)

Held-out valid[0:192], M4, `src/quarters.py` and `scratch/pos_blend.py`.

## Per-quarter WP (log: logs/quarters_2026-10-22.log)

| checkpoint | full | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| c1000_lr1e3_e8 | 0.6870 | 0.6614 | 0.6926 | 0.7243 | 0.6760 |
| final_s1 | 0.6849 | 0.6593 | 0.6925 | 0.7156 | 0.6788 |
| dz192_s1 | 0.6863 | 0.6565 | 0.6944 | 0.7185 | 0.6819 |
| dz128_s0 | 0.6866 | 0.6575 | 0.6943 | 0.7165 | 0.6836 |

Same shape for every model, so the shape is the data, not state drift.
No late-sequence decay (dozken's notes report one; our --state-reset 2000 models do not have it).

## Position-weighted blend e8 + dz128_s0 (weights fitted on even sequences, scored on odd, and back)

- 2000-step segments: cross-fitted gain -0.00054 (logs/pos_blend_e8_dz128.log).
- 5000-step segments (quarters): cross-fitted gain +0.00041 (logs/pos_blend_e8_dz128_q.log).
- Both are below the pair standard error (about 0.001). Not pursued.

## Teacher ensembles (logs/teacher_mix.log, logs/teacher_qmix.log)

- Plain mean of all 13 checkpoints: 0.69111 held-out; board pair e8 + final_s1: 0.68858.
- Greedy member selection does not transfer between halves (2 vs 6 members; one half loses to the board pair).
- Per-quarter greedy selection vs global greedy, cross-fitted: -0.00090 (even->odd) and -0.00078 (odd->even). Not pursued.
- Robust teacher for the next distillation run: the plain mean of all 13.
