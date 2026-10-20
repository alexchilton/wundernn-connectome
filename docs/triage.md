# Alpha Connectome: triage before compute (2026-10-20)

## Baseline

Stateful GRU (hidden 128, two hidden tensors), ONNX, from the starter pack.
Organisers state 0.617052 WP on full valid.
Our local run on full valid: pending (logs/baseline_valid.log).
Speed on the M4 Pro, measured on 3 train sequences: 17.5 us per predict() call, 11.5 min for the 39.4M test rows.
Limit: 60 min on one server CPU, about 91 us per call; server CPU speed relative to the M4 is not measured.

## What the scorer rewards (read from METRIC.md and utils.py)

Weights are |clip(y)|, so rows with large targets dominate and rows with y = 0 count for nothing.
One correlation per target, pooled over all sequences: a per-sequence offset in the prediction is penalised.
Predictions are clipped to [-2, 2] by the scorer, so predictions outside that range lose information.
Under |y| weighting the best predictor is E[|y| y | x] / E[|y| | x], not E[y | x].
A |y|-weighted MSE loss targets that quantity directly.

## Data facts (21 train sequences, 420,000 rows, measured 2026-10-20)

t0: mean -0.234, std 0.946, 22.8% exactly zero, 4.4% outside [-2, 2].
t1: mean -0.305, std 0.986, 21.1% exactly zero, 5.0% outside [-2, 2].
corr(t0, t1) = -0.801.
t0 autocorrelation within a sequence: 0.982 at lag 1, 0.805 at lag 20, 0.455 at lag 100, 0.157 at lag 500.
Best single linear feature: i0_p3 at 0.221 for t0, i0_v0 at -0.105 for t1.
Features look standardised already (most means within +-2, stds 0.04 to 2.9).

## Three plausible bottlenecks

1. Loss does not match the metric.
   Evidence: the baseline has no training code, so its loss is unknown; |y| weighting changes the optimal predictor.
   Smallest test: train the same GRU twice on the same sequences, plain MSE vs |y|-weighted MSE, score both on valid.
   Cost: two small GRU runs, about 1 GPU hour each on Colab.
   Disproof: weighted MSE does not beat plain MSE on valid WP by more than seed noise.

2. Model capacity at fixed speed.
   Evidence: hidden 128 uses 17.5 us of a ~91 us budget; most of that time is per-call Python overhead, not the matmul.
   Smallest test: hidden 128 vs 256 vs 384, same loss, same data; time each in solution.py.
   Cost: three runs plus timing.
   Disproof: valid WP flat from 128 to 384.

3. Context length (state memory).
   Evidence: targets stay correlated for hundreds of steps; a GRU trained on short chunks may not use long context.
   Smallest test: train on chunk length 256 vs 2048 with truncated BPTT carrying state.
   Cost: two runs.
   Disproof: no WP difference between chunk lengths.

Recommended first: 1, because it costs nothing at inference time and the metric argument is direct.

## Not yet measured

Server CPU speed.
Whether valid and test come from the same period or instrument mix.
How the visible valid mask differs from the hidden test mask (the FAQ warns against fitting the visible mask).

## Board calibration (2026-10-20)

Baseline uploaded unchanged: public 0.59353, platform time 26m36s, memory 624 MB.
Offset: held-out valid[0:192] 0.61987 -> public 0.59353 = -0.026; full valid 0.61705 -> -0.024.
Platform speed: 26m36s vs 698 s locally on the M4 = 2.29x slower.
Budget: 60 min / 2.29 = 26 local min = about 42 us per call on the M4; treat <= 28 us as safe (dozken saw the same graph drift 38 -> 64 min).
Our |y|-weighted MSE GRU 128x2 (600 train seqs, epoch 4): held-out 0.66197 -> public 0.63279 (offset -0.029), platform 26m22s.
Held-out gain over baseline +0.042 became +0.039 public: the held-out set predicts the board.
