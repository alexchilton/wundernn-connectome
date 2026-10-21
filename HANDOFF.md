# Handoff (state, overwritten each session) — 2026-10-21 05:20

## Branch and commit
main, pushed to github.com/alexchilton/wundernn-connectome (private).

## Best on the board
full1: GRU 128x2, loss |y| x P(scored) (soft mask), 10,607 train + valid[192:], 4 epochs, chunk 500, batch 64, seed 0, Colab L4.
Held-out valid[0:192] 0.68310, public 0.65392, 15.7 us/call, platform 26m12s. Zip: submissions/sub_full1.zip.
Command: python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 --loss wmse --soft-mask --add-valid-from 192 --hidden 128 --layers 2 --batch 64 --chunk 500 --epochs 4 --eval-every 48 --seed 0.

## Board pairs (held-out -> public)
baseline 0.61987 -> 0.59353; plain MSE 0.64906 -> 0.61346; |y|-weighted MSE 0.66197 -> 0.63279; soft mask 600 seqs 0.67231 -> 0.63693; full1 0.68310 -> 0.65392.
Offset about -0.03 (range -0.026 to -0.036). Prize line (#8) 0.67017 public = ~0.700 held-out.

## Tried overnight, full data, one change each vs full1 (best held-out, seed 0)
192x2 0.68121; 128x3 0.68075; state reset 2000 0.68378; diff+proj128+skip 0.67807; + EMA(10,100,1000) 0.67829.
None passed the +0.005 gate (tools/pick_final.py, docs/final-recipe.md). Time-feature arm dropped (built on diff+proj+skip).
600-seq Mac runs: 12 epochs peaks 0.66944 at epoch 4 then declines (overfits); 128x3 0.66804 vs 128x2 0.67231.
Ensemble of two 600-seq models: +0.0003 at 27.9 us/call.

## Running now (log: logs/overnight_final.log, chain tools/overnight_final.sh, waits on remote EXEC EXIT markers)
Colab session wnn (L4): final_s1, final_s2 (full1 recipe, seeds 1 and 2) -> seed noise; then morning_norm (--loss-norm batch) and morning_ema8 (--ema-weights 0.999, 8 epochs), seed 0, paired with full1; then the session is stopped.
Checkpoints land in out/<arm>/best.pt; export with src/export_ckpt.py, score with src/score_holdout.py.

## Next, and what would disprove it
1. Seed spread of final_s1/s2 vs full1 sets the real noise floor; a change must beat it.
2. morning_norm / morning_ema8: keep if +0.003 over full1 at matching epochs (and above the seed spread).
3. Global Pearson loss with running stats (dozken's biggest remaining difference): disproved if it does not beat full1 by the seed spread.
4. Ensemble mean of the strongest different checkpoints (tonight's arms): if < +0.003 over the best single, skip distillation.
5. Cheap arms not yet run: chunk 1000, lr 1e-3 cosine.

## Not to repeat
Output scaling (flat), regime/a5-a7 corrections, book features, LSTM, wider/deeper, new inputs as built (diff/proj/skip/EMA), reset 2000 (no gain for us), inference-only resets.
