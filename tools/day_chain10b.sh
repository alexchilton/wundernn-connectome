#!/bin/bash
# Day chain 10: Fable plan 2026-10-23 items 4 and 5. e8th_s0 = e8 recipe + EMA + --train-holdout 192 (train-regime read-out from ep1-8.pt); e8noval_s0 = e8 recipe + EMA without valid[192:] in the pool (board test of the regime).
# Data: rclone copy gdrive:wnn/data. Checkpoints: the job pushes out/<arm> to gdrive:wnn/out/<arm>
# every 5 min (rclone, background thread), and this script mirrors best/latest to the Mac every 5 min.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab; S=${1:-wnn11}
L=logs/day_chain10b.log
ARMS="e8noval_s0"
remote_done() { timeout 90 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
pull() { for a in "$@"; do mkdir -p out/$a; for f in best latest ep1 ep2 ep3 ep4 ep5 ep6 ep7 ep8; do timeout 300 $C download -s $S /content/wnn/out/$a/$f.pt out/$a/.$f.tmp < /dev/null > /dev/null 2>&1 && mv out/$a/.$f.tmp out/$a/$f.pt; done; done; }
FAILED=0
wait_for() { for a in "$@"; do until remote_done $a; do pull $a; sleep 300; done; pull $a; grep -q "^EXEC EXIT 0" logs/colab_$a.log || { FAILED=1; echo "$(date +%H:%M) $a FAILED: $(grep -m1 Error logs/colab_$a.log)" >> $L; }; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1), pulled $(ls -la out/$a/best.pt 2>/dev/null | awk '{print $5}')" >> $L; done; }

echo "$(date +%H:%M) day chain 10b start (item 5 resume, rclone)" >> $L
timeout 60 $C sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { cd /tmp && timeout 600 $C new -s $S --gpu L4 < /dev/null >> ~/PycharmProjects/wundernn-connectome/$L 2>&1; cd ~/PycharmProjects/wundernn-connectome; }
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/wnn/src","/content/wnn/logs","/content/wnn/out","/content/wnn/wnn_connectome_starterpack")]; print("dirs ok")' | timeout 120 $C repl -s $S >> $L 2>&1
tools/colab_rclone.sh $S >> $L 2>&1 || { echo "$(date +%H:%M) RCLONE SETUP FAILED on $S, session kept" >> $L; exit 1; }
echo "$(date +%H:%M) rclone ok on $S" >> $L
for f in src/*.py wnn_connectome_starterpack/utils.py; do timeout 120 $C upload -s $S $f /content/wnn/$f < /dev/null > /dev/null 2>&1 || echo "upload failed $f" >> $L; done
cat > /tmp/wnn_day10b.py <<PY
import os, subprocess, threading, time
subprocess.run("pip -q install onnx > /content/wnn/logs/pip_onnx.log 2>&1", shell=True)   # the trainer exports ONNX at the end
t = time.time()
r = subprocess.run("rclone copy gdrive:wnn/data /content/wnn/out/data --transfers 8", shell=True)
open("/content/wnn/logs/drive_restore.log", "w").write(f"rclone rc {r.returncode} in {time.time() - t:.0f}s\n")
ARMS = "$ARMS".split()
def push():                                   # checkpoints leave the VM while training
    while True:
        for a in ARMS:
            if os.path.isdir(f"/content/wnn/out/{a}"):
                subprocess.run(f"rclone copy /content/wnn/out/{a} gdrive:wnn/out/{a}", shell=True)
        time.sleep(300)
threading.Thread(target=push, daemon=True).start()
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--soft-mask --batch 64 --eval-every 48 --save-every 8 --resume")
arm = "--hidden 128 --layers 2 --loss wmse --mse-weight 0.5 --chunk 1000 --epochs 8 --lr 1e-3 --ema-weights 0.999"  # = e8 cfg + EMA
smoke = subprocess.run(f"{base} {arm} --seed 0 --max-batches 1 --out out/smoke_day10b > /content/wnn/logs/smoke_day10b.log 2>&1", shell=True)
open("/content/wnn/logs/smoke_day10b.log", "a").write(f"\nEXEC EXIT {smoke.returncode}\n")
if smoke.returncode:
    raise SystemExit("smoke failed: see logs/smoke_day10b.log")
subprocess.run("rclone copy gdrive:wnn/out/e8noval_s0 /content/wnn/out/e8noval_s0", shell=True)   # resume from Drive (wnn10 died at ep 2)
for name, extra in [("e8noval_s0", f"{arm} --seed 0")]:                                         # item 5: no valid[192:]
    r = subprocess.run(f"{base} {extra} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    subprocess.run(f"rclone copy /content/wnn/out/{name} gdrive:wnn/out/{name}", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
timeout 60 $C rm -s $S /content/wnn/logs/smoke_day10b.log < /dev/null > /dev/null 2>&1; rm -f logs/colab_smoke_day10b_$S.log
$C exec -s $S -f /tmp/wnn_day10b.py < /dev/null >> $L 2>&1 &
echo "$(date +%H:%M) job submitted: rclone data copy (~27 min), smoke, then arms" >> $L
until timeout 60 $C download -s $S /content/wnn/logs/smoke_day10b.log logs/colab_smoke_day10b_$S.log < /dev/null > /dev/null 2>&1 && grep -q "^EXEC EXIT" logs/colab_smoke_day10b_$S.log; do sleep 60; done
grep -q "^EXEC EXIT 0" logs/colab_smoke_day10b_$S.log || { echo "$(date +%H:%M) SMOKE FAILED, session $S kept: $(grep -m1 Error logs/colab_smoke_day10b_$S.log)" >> $L; exit 1; }
echo "$(date +%H:%M) smoke ok, arms running" >> $L
wait_for $ARMS
[ $FAILED = 1 ] && { echo "$(date +%H:%M) an arm FAILED, session $S kept for inspection" >> $L; exit 1; }
$C stop -s $S < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 10b DONE, colab session stopped" >> $L
