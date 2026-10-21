# Handoff (state, overwritten each session) — 2026-10-21 08:30

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

## Morning arms (seed 0, vs full1 0.68310; seeds 1/2 = 0.68488 / 0.68256, spread about +-0.0012)
morning_norm (--loss-norm batch): best 0.68364, null.
morning_ema8 (--ema-weights 0.999, 8 epochs): best 0.68431, epoch ends flat 0.678-0.683 from epoch 1: null, longer does not help.

## Running now (Colab session wnn, chain tools/day_chain.sh, log logs/overnight_final.log, per-arm logs/colab_<arm>.log)
day_pearson, day_chunk1000, day_lr1e3, day_dozken (control: dozken recipe rebuilt with our code), then the session stops (~11:00).
Colab allows two sessions; the other one runs casmi26 until ~10:10.

## Next, and what would disprove it
1. Fuse the best two members into ONE 256-wide GRU with block-diagonal weights (exact member mean; docs/fable-ideas.md round 2).
   Measured M4 latency 21.7 us/call (x1.13 scorer loop = ~24.5, under 28); 2-member mean seed1 + reset2k = 0.68655 held-out (+0.0035).
   First upload doubles as a platform timing probe (2.7 MB weights). Disproved if public does not beat 0.65392.
2. day_pearson judged as an ensemble member: residual corr vs full1 <= 0.975 earns a slot.
3. Read every arm by epoch age (rows since last i0_p change): age >= 64 scores 0.42-0.53 and nothing has moved it.

## Not to repeat
Output scaling (flat), regime/a5-a7 corrections, book features, LSTM, wider/deeper, new inputs as built (diff/proj/skip/EMA), reset 2000 (no gain for us), inference-only resets, batch loss-norm, EMA weights + 8 epochs, cold-start fixes (Fable round 2), >2 ensemble members.
