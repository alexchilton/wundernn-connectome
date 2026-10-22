# Handoff (state, overwritten each session) — 2026-10-22 12:15

## Branch and commit
main, pushed to github.com/alexchilton/wundernn-connectome (private).

## Best on the board (2026-10-22)
fz_e8_dz96s1: fused pair e8 (out/c1000_lr1e3_e8/best.pt) + dz96p32_s1 (out/dz96p32_s1/latest.pt, --dz --proj 32 --hidden 96,
tools/day_chain6.sh). Held-out 0.69043, public **0.65924**, platform 47m52s, M4 1.11-1.23x the plain pair (logs/timing_e8_dz96s1.log).
Zip: submissions/sub_fused_e8_dz96s1.zip. Built with: python src/fuse.py out/fz_e8_dz96s1 out/c1000_lr1e3_e8/best.pt out/dz96p32_s1/latest.pt

## Board pairs (held-out -> public)
baseline 0.61987 -> 0.59353; plain MSE 0.64906 -> 0.61346; |y|-weighted MSE 0.66197 -> 0.63279; soft mask 600 seqs 0.67231 -> 0.63693; full1 0.68310 -> 0.65392.
dz128 single 0.68655 -> 0.65566; e8 + old final_s1 0.68858 -> 0.65817; e8 + dz96p32_s1 0.69043 -> 0.65924; e8 + dz128 TIMEOUT (1.29x).
Offset about -0.031 for dz pairs (range -0.026 to -0.036 overall). Timing: 1.2x fits (47m52s), 1.29x timed out.

## Running / next (2026-10-22)
- Chain 7 (tools/day_chain7.sh, wnn7): distillation students, teacher mean(e8, final_s1), --distill 0.5, plain 128, e8 recipe.
- Next teacher: plain mean of all 13 checkpoints, held-out 0.69111 (logs/teacher_mix.log); per-quarter teachers lose (docs/archive/position-blend-2026-10-22.md).
- Then chain 8: bidirectional GRU teacher, predictions to Drive, students via a --teacher-preds option (not written yet).

## Not to repeat
Every measured null, with its number, is in docs/findings.md ("Measured, null or worse").
Distillation ceiling (docs/archive/fable-plan-2026-10-21.md section 3): a student recovers at most the teacher's gain over its best
member (+0.001 to +0.0016 for 2-model teachers); the 13-checkpoint mean raises that to about +0.004 over e8.

## Docs
docs/findings.md is the ledger (board results, kept, nulls, instrument facts, open hypotheses).
docs/tutorial.md explains the task and the solution; docs/prior-art.md is the external survey; docs/archive/ holds superseded reviews.
