# Alpha Connectome (wundernn.io) — GRU sequence model, archived

Competition: predict the next-step neural target from a stream of connectome features, scored by weighted Pearson (WP), with a latency limit per call.
Ends 2026-11-15.
This repo is archived as of 2026-10-25: the work is finished, the local checkout was deleted, and the large artifacts live in the GitHub release `archive-2026-10-25`.

## Result

Best public score: **0.65936**, `submissions/release_cand_8.zip`.
It is two GRUs fused into one ONNX graph (`src/fuse.py`):
`quiet_s0/best.pt` (e8 recipe with an auxiliary "quiet period" target) and `dz96p32_s1/latest.pt` (diff features, projection 32, hidden 96).
Held-out WP on 192 validation sequences: 0.69088. Latency on the M4: 26.7 us/call (limit about 28 us/call on the platform).

Held-out runs about 0.031 above public (range 0.026 to 0.036 over all uploads).
The full table of uploads, kept changes and measured nulls is `docs/findings.md`.

## Where things are

| what | where |
|---|---|
| ledger: every board result, kept change and null, with numbers | `docs/findings.md` |
| task and solution explained | `docs/tutorial.md` |
| external survey | `docs/prior-art.md` |
| last state and next steps | `HANDOFF.md` |
| trainer, model, fusion, teacher dump | `src/train_gru.py`, `src/model.py`, `src/fuse.py`, `src/dump_teacher.py` |
| Colab chains (staging, Drive mirror, watchdog) | `tools/` |
| every uploaded zip and the release candidates | `submissions/` (`release_candidates.md` explains each) |
| checkpoints (`out/*/best.pt`, `latest.pt`, `*.json`) | release asset `out_checkpoints.tar.gz` |
| teacher predictions (`out/preds`) | release asset `out_preds.tar.gz` |
| run logs (`logs/`, gitignored) | release asset `logs.tar.gz` |
| prepared training data (`out/data`, 12 GB) | not archived: rebuild from the starter pack, or the Drive copy `MyDrive/wnn/data` |

## Restore

    git clone https://github.com/alexchilton/wundernn-connectome && cd wundernn-connectome
    gh release download archive-2026-10-25
    tar -xzf out_checkpoints.tar.gz && tar -xzf out_preds.tar.gz && tar -xzf logs.tar.gz
    python -m venv .venv && .venv/bin/pip install -r requirements.txt   # versions the work ran on (Python 3.11)

Rebuild RC8: `python src/fuse.py out/fz_quiet_dz96s1 out/quiet_s0/best.pt out/dz96p32_s1/latest.pt`.
