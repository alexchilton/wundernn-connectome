#!/bin/bash
# Day chain 12a: findings section 5 aux arms on the e8 recipe (c1000_lr1e3_e8: held-out 0.68703, seed 0):
#   quiet_s0 = --aux 2 --aux-kind quiet (log rows to the next i0_p0 change + its sign; targets the age>=64 band, 0.42 vs 0.80 at age 0);
#   level_s0 = --aux 2 --aux-kind level --aux-w 0.2 (|y|-weighted mean of clip(y) over the next 2000 rows; level corr 0.82 today).
# Code comes from git branch arm-four (not main). Data: rclone gdrive:wnn/data. Both arms run in parallel on one L4.
# Disproof: held-out best <= 0.68703 + 0.001 for an arm (pair noise ~0.001); quiet also: age>=64 band within +-0.008.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab; S=${1:-wnn13}
L=logs/day_chain12a.log
ARMS="quiet_s0 level_s0"
remote_done() { timeout 90 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
pull() { local a; for a in "$@"; do mkdir -p out/$a; for f in best latest; do timeout 300 $C download -s $S /content/wnn/out/$a/$f.pt out/$a/.$f.tmp < /dev/null > /dev/null 2>&1 && mv out/$a/.$f.tmp out/$a/$f.pt; done; done; }
FAILED=0
wait_all() { for a in "$@"; do until remote_done $a; do pull $@; sleep 300; done; pull $a; grep -q "^EXEC EXIT 0" logs/colab_$a.log || { FAILED=1; echo "$(date +%H:%M) $a FAILED: $(grep -m1 Error logs/colab_$a.log)" >> $L; }; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1)" >> $L; done; }

echo "$(date +%H:%M) day chain 12a start" >> $L
SRC=$(mktemp -d); mkdir -p $SRC/src
for f in $(git ls-tree --name-only arm-four src/ | grep '\.py$'); do git show arm-four:$f > $SRC/$f; done
grep -q -- "--aux-kind" $SRC/src/train_gru.py || { echo "$(date +%H:%M) branch code missing --aux-kind, abort" >> $L; exit 1; }
timeout 60 $C sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { cd /tmp && timeout 600 $C new -s $S --gpu L4 < /dev/null >> ~/PycharmProjects/wundernn-connectome/$L 2>&1; cd ~/PycharmProjects/wundernn-connectome; }
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/wnn/src","/content/wnn/logs","/content/wnn/out","/content/wnn/wnn_connectome_starterpack")]; print("dirs ok")' | timeout 120 $C repl -s $S >> $L 2>&1
tools/colab_rclone.sh $S >> $L 2>&1 || { echo "$(date +%H:%M) RCLONE SETUP FAILED on $S, session kept" >> $L; exit 1; }
echo "$(date +%H:%M) rclone ok on $S" >> $L
for f in $SRC/src/*.py; do timeout 120 $C upload -s $S $f /content/wnn/src/$(basename $f) < /dev/null > /dev/null 2>&1 || echo "upload failed $f" >> $L; done
timeout 120 $C upload -s $S wnn_connectome_starterpack/utils.py /content/wnn/wnn_connectome_starterpack/utils.py < /dev/null > /dev/null 2>&1 || echo "upload failed utils" >> $L
cat > /tmp/wnn_day12a.py <<PY
import os, subprocess, threading, time
subprocess.run("pip -q install onnx > /content/wnn/logs/pip_onnx.log 2>&1", shell=True)
t = time.time()
r = subprocess.run("rclone copy gdrive:wnn/data /content/wnn/out/data --transfers 8", shell=True)
open("/content/wnn/logs/drive_restore.log", "w").write(f"rclone rc {r.returncode} in {time.time() - t:.0f}s\n")
ARMS = "$ARMS".split()
def push():
    while True:
        for a in ARMS:
            if os.path.isdir(f"/content/wnn/out/{a}"):
                subprocess.run(f"rclone copy /content/wnn/out/{a} gdrive:wnn/out/{a}", shell=True)
        time.sleep(300)
threading.Thread(target=push, daemon=True).start()
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--add-valid-from 192 --soft-mask --batch 64 --eval-every 48 --save-every 8 --resume "
        "--hidden 128 --layers 2 --loss wmse --mse-weight 0.5 --chunk 1000 --epochs 8 --lr 1e-3 --seed 0")   # = c1000_lr1e3_e8
extra = {"quiet_s0": "--aux 2 --aux-kind quiet --aux-w 0.3", "level_s0": "--aux 2 --aux-kind level --aux-w 0.2"}
for a, x in extra.items():
    smoke = subprocess.run(f"{base} {x} --max-batches 1 --out out/smoke_{a} > /content/wnn/logs/smoke_{a}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/smoke_{a}.log", "a").write(f"\nEXEC EXIT {smoke.returncode}\n")
    if smoke.returncode:
        raise SystemExit(f"smoke {a} failed")
open("/content/wnn/logs/smoke_day12a.log", "w").write("EXEC EXIT 0\n")
def run(a):
    r = subprocess.run(f"{base} {extra[a]} --out out/{a} > /content/wnn/logs/{a}.log 2>&1", shell=True)
    subprocess.run(f"rclone copy /content/wnn/out/{a} gdrive:wnn/out/{a}", shell=True)
    open(f"/content/wnn/logs/{a}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
th = [threading.Thread(target=run, args=(a,)) for a in ARMS]
[t_.start() for t_ in th]; [t_.join() for t_ in th]
PY
for a in $ARMS smoke_day12a; do timeout 60 $C rm -s $S /content/wnn/logs/$a.log < /dev/null > /dev/null 2>&1; rm -f logs/colab_$a.log; done
$C exec -s $S -f /tmp/wnn_day12a.py < /dev/null >> $L 2>&1 &
echo "$(date +%H:%M) job submitted: rclone data copy (~27 min), smoke x2, then both arms in parallel" >> $L
until timeout 60 $C download -s $S /content/wnn/logs/smoke_day12a.log logs/colab_smoke_day12a.log < /dev/null > /dev/null 2>&1 && grep -q "^EXEC EXIT" logs/colab_smoke_day12a.log; do
  timeout 60 $C download -s $S /content/wnn/logs/smoke_quiet_s0.log logs/colab_smoke_quiet.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT [1-9]" logs/colab_smoke_quiet.log 2>/dev/null && { echo "$(date +%H:%M) SMOKE quiet FAILED, session kept" >> $L; exit 1; }
  sleep 120; done
echo "$(date +%H:%M) smoke ok, arms running" >> $L
wait_all $ARMS
[ $FAILED = 1 ] && { echo "$(date +%H:%M) an arm FAILED, session $S kept for inspection" >> $L; exit 1; }
$C stop -s $S < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 12a DONE, colab session stopped" >> $L
