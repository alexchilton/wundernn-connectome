# Handoff (final) — 2026-10-25

## State
Archived. main at github.com/alexchilton/wundernn-connectome (private) holds all code, docs and submission zips.
Checkpoints, teacher predictions and logs are in the GitHub release `archive-2026-10-25` (see README.md "Restore").
The local checkout was deleted after the push and the release upload were verified.
Nothing is running: the last Colab chain (12d, session wnn13) finished 2026-10-24 20:38 and stopped its session.

## Best on the board
release_cand_8: quiet_s0/best.pt + dz96p32_s1/latest.pt fused, held-out 0.69088, public **0.65936** (2026-10-24).
Previous best: fz_e8_dz96s1 (e8 best + dz96p32_s1), held-out 0.69043, public 0.65924.

## Last experiments (findings.md section 5 hypotheses, all measured 2026-10-24)
- quiet aux target: seed 0 +0.0013 (0.68834 vs 0.68703), seed 1 -0.0029 (best 0.68868 vs 0.69161). Null over two seeds.
- level aux target: 0.68759 (+0.0006). Null.
- 13-checkpoint distillation student: best single member 0.68967 (+0.0026), but its pair with dz96p32_s1 is 0.69070 < RC8 0.69088.
- bidirectional teacher: leaks the target (held-out 0.99641). Dead.

## If the work resumes
The open items in findings.md section 5 are speed-only (7-epoch schedule, loader prefetch) or one blind board test
(up-weighting valid sequences 2-3x in the final run). None has a measured upside above the board's noise (about 0.001).
One untested lever: a final retrain that includes the 192 held-out sequences (check first whether the release members already do).

## Docs
docs/findings.md is the ledger; docs/tutorial.md explains the task and the solution; docs/prior-art.md is the external survey.
