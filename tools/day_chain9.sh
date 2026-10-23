#!/bin/bash
# Day chain 9: dz96p32 seed lottery (Fable plan 2026-10-23; dz96p32 s0 0.68329 vs s1 0.68739 single, gap 0.0041). Seeds 2, 3, 4 on the exact s1 recipe (cfg in out/dz96p32_s1/latest.pt).
# Data: rclone copy gdrive:wnn/data. Checkpoints: the job pushes out/<arm> to gdrive:wnn/out/<arm>
# every 5 min (rclone, background thread), and this script mirrors best/latest to the Mac every 5 min.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab; S=${1:-wnn9}
L=logs/day_chain9.log
ARMS="dz96p32_s2 dz96p32_s3 dz96p32_s4"
remote_done() { timeout 90 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
pull() { for a in "$@"; do mkdir -p out/$a; for f in best latest; do timeout 300 $C download -s $S /content/wnn/out/$a/$f.pt out/$a/.$f.tmp < /dev/null > /dev/null 2>&1 && mv out/$a/.$f.tmp out/$a/$f.pt; done; done; }
FAILED=0
wait_for() { for a in "$@"; do until remote_done $a; do pull $a; sleep 300; done; pull $a; grep -q "^EXEC EXIT 0" logs/colab_$a.log || { FAILED=1; echo "$(date +%H:%M) $a FAILED: $(grep -m1 Error logs/colab_$a.log)" >> $L; }; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1), pulled $(ls -la out/$a/best.pt 2>/dev/null | awk '{print $5}')" >> $L; done; }

echo "$(date +%H:%M) day chain 9 start (dz96p32 seeds 2-4, rclone)" >> $L
timeout 60 $C sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { cd /tmp && timeout 600 $C new -s $S --gpu L4 < /dev/null >> ~/PycharmProjects/wundernn-connectome/$L 2>&1; cd ~/PycharmProjects/wundernn-connectome; }
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/wnn/src","/content/wnn/logs","/content/wnn/out","/content/wnn/wnn_connectome_starterpack")]; print("dirs ok")' | timeout 120 $C repl -s $S >> $L 2>&1
tools/colab_rclone.sh $S >> $L 2>&1 || { echo "$(date +%H:%M) RCLONE SETUP FAILED on $S, session kept" >> $L; exit 1; }
echo "$(date +%H:%M) rclone ok on $S" >> $L
for f in src/*.py wnn_connectome_starterpack/utils.py; do timeout 120 $C upload -s $S $f /content/wnn/$f < /dev/null > /dev/null 2>&1 || echo "upload failed $f" >> $L; done
cat > /tmp/wnn_day9.py <<PY
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
        "--soft-mask --add-valid-from 192 --batch 64 --eval-every 48 --save-every 8 --resume")
arm = ("--hidden 96 --layers 2 --loss pearson --mse-weight 0.1 --chunk 1000 --epochs 5 --lr 1e-3 --wd 0.01 "     # = dz96p32_s1 cfg
       "--state-reset 2000 --diff --proj 32 --dz --dropout 0.1 --loss-norm chunk --ema-weights 0.999")
smoke = subprocess.run(f"{base} {arm} --seed 0 --max-batches 1 --out out/smoke_day9 > /content/wnn/logs/smoke_day9.log 2>&1", shell=True)
open("/content/wnn/logs/smoke_day9.log", "a").write(f"\nEXEC EXIT {smoke.returncode}\n")
if smoke.returncode:
    raise SystemExit("smoke failed: see logs/smoke_day9.log")
for name, extra in [(f"dz96p32_s{s}", f"{arm} --seed {s}") for s in (2, 3, 4)]:
    r = subprocess.run(f"{base} {extra} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    subprocess.run(f"rclone copy /content/wnn/out/{name} gdrive:wnn/out/{name}", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
timeout 60 $C rm -s $S /content/wnn/logs/smoke_day9.log < /dev/null > /dev/null 2>&1; rm -f logs/colab_smoke_day9_$S.log
$C exec -s $S -f /tmp/wnn_day9.py < /dev/null >> $L 2>&1 &
echo "$(date +%H:%M) job submitted: rclone data copy (~27 min), smoke, then arms" >> $L
until timeout 60 $C download -s $S /content/wnn/logs/smoke_day9.log logs/colab_smoke_day9_$S.log < /dev/null > /dev/null 2>&1 && grep -q "^EXEC EXIT" logs/colab_smoke_day9_$S.log; do sleep 60; done
grep -q "^EXEC EXIT 0" logs/colab_smoke_day9_$S.log || { echo "$(date +%H:%M) SMOKE FAILED, session $S kept: $(grep -m1 Error logs/colab_smoke_day9_$S.log)" >> $L; exit 1; }
echo "$(date +%H:%M) smoke ok, arms running" >> $L
wait_for $ARMS
[ $FAILED = 1 ] && { echo "$(date +%H:%M) an arm FAILED, session $S kept for inspection" >> $L; exit 1; }
$C stop -s $S < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 9 DONE, colab session stopped" >> $L
