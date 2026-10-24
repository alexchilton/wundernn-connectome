I am competing in a machine-learning competition and I am stuck on a plateau. I want new ideas, ranked, with a reason each could work. Do not repeat the things listed as already tried unless you explain why my test of it was wrong.

## The task
- Market data from a trading firm. Sequences of 20,000 rows (time steps). Each row has 112 anonymised, pre-scaled float features: for two instruments i0 and i1, 22 price-like and 22 volume-like book features (11 levels per side), 4 price changes and 4 volume changes each, plus 8 unexplained extra columns a0..a7.
- Predict two targets t0 and t1 per row (two indicators of the future price move of i0).
- Strictly online: the model sees one row at a time, in order, and must answer before the next row. State is reset between sequences. Rows 0-98 are warm-up; predictions are needed from row 99 to 19,999.
- Metric: weighted Pearson correlation, computed globally over all scored rows of the whole test set, per target, then averaged. Weights = |clipped target|; targets and predictions are clipped to [-2, 2]. Only some rows are scored (an is_scored mask): it is given for validation and hidden for test.
- Data: train 10,607 sequences (no is_scored mask), validation 1,873 (with mask), hidden test 1,970.
- Inference limits: 1 CPU core, ONNX Runtime with one thread, no GPU, 60 minutes for the whole test set (39.4 M rows, so about 90 microseconds per row on their machine; their machine is about 2.3-2.7x slower than my laptop), submission ZIP at most 20 MB.

## Where I am
- Provided baseline: GRU 128x2, 0.617 on validation, 0.5935 public.
- My best public: 0.65924 = two GRU models fused into one wider GRU graph (exact mean of the two members), each trained on the full train set plus validation sequences 192+, with a loss weighted by |y| times a learned P(scored) soft mask. Held-out score (validation sequences 0-191) 0.69043. Held-out minus public is a stable offset of about -0.030.
- Eight further pair ensembles (different seeds, chunk lengths, checkpoints, distilled students) all land at 0.6571-0.6592 public. The differences are noise. A model trained without the validation data scored 0.65815 (-0.0011).
- Two fused members fit the time budget; three do not.

## Measured as null or worse (held-out, vs about 0.683 for a single model)
- Wider or deeper GRU (192x2, 128x3); more epochs (peaks at epoch 4-6, then overfits); EMA of weights; batch loss normalisation.
- Extra inputs: differences, projections, skip connections, EMAs of features at 10/100/1000 steps, a time-in-sequence feature.
- State tricks: periodic state reset, warm-up replay, learned or mean initial state.
- Post-processing: per-position or per-age offsets and scales, causal level rescaling, a causal ridge second stage on the model output, refit heads on slow hidden-state features, freezing predictions within regimes.
- A causal local-window transformer (d 128, 4 blocks, window 256): equal to the GRU alone (0.6876 vs 0.6870); its errors correlate 0.991 with the GRU's.
- Distillation from the pair into one model: equal to the best single model.
- Predicting t0+t1 / t0-t1 instead of t0, t1: worse.
- Choosing checkpoints by held-out score: worth nothing on the public board.

## What I want from you
1. The 5 ideas most likely to add at least +0.005 public, ranked. For each: the mechanism in one sentence (why the metric would rise), the smallest experiment that would test it, and what result would disprove it.
2. Anything in the metric I may be under-exploiting: the global (not per-sequence) Pearson, the |target| weights, the clipping, the hidden scored mask.
3. Ways to fit more model into the time limit on one CPU thread (architecture, quantisation, ONNX graph tricks), with expected speed-ups.
4. Which of my "null" results you think were tested wrongly, and why.
Be concrete. Say when you are guessing.
