#!/bin/bash
# Day chain 4: our recipe (wmse soft mask, lr 1e-3) with chunk 4000, seeds 0 and 1, 6 epochs -> fused 256-wide pair
# (the width proven at 47 min on the platform). dozken measured chunk 4000 +0.0164 over chunk 1000 (hidden-state
# degradation over long sequences); 6 epochs because chunk 4000 has 4x fewer optimiser steps per epoch.
# Data comes from Drive (MyDrive/wnn/data, copied by wnn3 at 13:54), not restaged. Needs Alex to mount Drive once:
#   colab drivemount -s wnn4
# Arm outputs live on Drive via symlinks and are mirrored to out/<arm>/ every 5 min.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab; S=${1:-wnn4}; D=/content/drive/MyDrive/wnn
L=logs/overnight_final.log
ARMS="c4000_lr1e3_s0 c4000_lr1e3_s1"
remote_done() { timeout 90 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
pull() { for a in "$@"; do mkdir -p out/$a; for f in best latest; do timeout 300 $C download -s $S /content/wnn/out/$a/$f.pt out/$a/.$f.tmp < /dev/null > /dev/null 2>&1 && mv out/$a/.$f.tmp out/$a/$f.pt; done; done; }
wait_for() { for a in "$@"; do until remote_done $a; do pull $a; sleep 300; done; pull $a; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1), pulled $(ls -la out/$a/best.pt 2>/dev/null | awk '{print $5}')" >> $L; done; }
started() { for i in $(seq 1 20); do timeout 60 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1 && grep -q "batch 1/" logs/colab_$1.log && return 0; sleep 30; done; return 1; }

echo "$(date +%H:%M) day chain 4 start (chunk 4000 pair)" >> $L
timeout 60 $C sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { cd /tmp && timeout 600 $C new -s $S --gpu L4 < /dev/null >> ~/PycharmProjects/wundernn-connectome/$L 2>&1; cd ~/PycharmProjects/wundernn-connectome; }
echo "$(date +%H:%M) $S up: waiting for Drive mount (colab drivemount -s $S)" >> $L
until timeout 60 $C log -s $S < /dev/null 2>&1 | grep -q "drive_auth_success"; do sleep 60; done
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/wnn/src","/content/wnn/logs","/content/wnn/out")]; print("dirs ok")' | timeout 120 $C repl -s $S >> $L 2>&1
for f in src/*.py; do timeout 120 $C upload -s $S $f /content/wnn/$f < /dev/null > /dev/null 2>&1 || echo "upload failed $f" >> $L; done
cat > /tmp/wnn_day4.py <<PY
import os, subprocess, time
D = "$D"
t = time.time()
r = subprocess.run(f"cp -r {D}/data /content/wnn/out/data", shell=True)
open("/content/wnn/logs/drive_restore.log", "w").write(f"cp rc {r.returncode} in {time.time() - t:.0f}s\n")
for a in "$ARMS".split():
    os.makedirs(f"{D}/out/{a}", exist_ok=True)
    if not os.path.islink(f"/content/wnn/out/{a}"):
        os.symlink(f"{D}/out/{a}", f"/content/wnn/out/{a}")
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--soft-mask --add-valid-from 192 --batch 64 --eval-every 48 --save-every 8 --resume --layers 2 "
        "--hidden 128 --loss wmse --chunk 4000 --lr 1e-3 --epochs 6")
for name, seed in [("c4000_lr1e3_s0", 0), ("c4000_lr1e3_s1", 1)]:
    r = subprocess.run(f"{base} --seed {seed} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
for try in 1 2 3; do
    $C exec -s $S -f /tmp/wnn_day4.py < /dev/null >> $L 2>&1 &
    started c4000_lr1e3_s0 && break
    echo "$(date +%H:%M) day4 arms did not start (try $try), checking restore" >> $L
    timeout 60 $C download -s $S /content/wnn/logs/drive_restore.log logs/colab_drive_restore_$S.log < /dev/null > /dev/null 2>&1; cat logs/colab_drive_restore_$S.log >> $L 2>/dev/null
done
wait_for $ARMS
$C stop -s $S < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 4 DONE, colab session stopped" >> $L
