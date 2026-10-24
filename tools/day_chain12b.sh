#!/bin/bash
# Day chain 12b: distillation teachers (findings section 5), e8 recipe (c1000_lr1e3_e8: held-out 0.68703, seed 0), --distill 0.5.
#   student13_s0: target from the plain mean of the 13 checkpoints in tools/teach13.txt (held-out 0.69111; logs/teacher_mix.log),
#                 precomputed by src/dump_teacher.py (out/teach13/{train,valid}_T.npy).
#   bidir_s0:     bidirectional GRU teacher (--bidir, model.BiNet); its predictions -> out/teachbi -> studentbi_s0.
# Code: git branch arm-four. Data: rclone gdrive:wnn/data. Disproof: student held-out best <= 0.68703 + 0.001 (ceiling ~ +0.004
# for the 13-mean, chain 7 students reached +0.001); bidir teacher held-out is printed but is not a submission (not causal).
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab; S=${1:-wnn14}
L=logs/day_chain12b.log
ARMS="student13_s0 bidir_s0 studentbi_s0"
remote_done() { timeout 90 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
pull() { for a in "$@"; do mkdir -p out/$a; for f in best latest; do timeout 300 $C download -s $S /content/wnn/out/$a/$f.pt out/$a/.$f.tmp < /dev/null > /dev/null 2>&1 && mv out/$a/.$f.tmp out/$a/$f.pt; done; done; }
FAILED=0
wait_all() { for a in "$@"; do until remote_done $a; do pull $@; sleep 300; done; pull $a; grep -q "^EXEC EXIT 0" logs/colab_$a.log || { FAILED=1; echo "$(date +%H:%M) $a FAILED: $(grep -m1 Error logs/colab_$a.log)" >> $L; }; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1)" >> $L; done; }

echo "$(date +%H:%M) day chain 12b start" >> $L
SRC=$(mktemp -d); mkdir -p $SRC/src
for f in $(git ls-tree --name-only arm-four src/ | grep '\.py$'); do git show arm-four:$f > $SRC/$f; done
grep -q -- "--teacher-preds" $SRC/src/train_gru.py && [ -f $SRC/src/dump_teacher.py ] || { echo "$(date +%H:%M) branch code missing --teacher-preds, abort" >> $L; exit 1; }
for f in $(cat tools/teach13.txt); do [ -f $f ] || { echo "$(date +%H:%M) teacher $f missing locally, abort" >> $L; exit 1; }; done
timeout 60 $C sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { cd /tmp && timeout 600 $C new -s $S --gpu L4 < /dev/null >> ~/PycharmProjects/wundernn-connectome/$L 2>&1; cd ~/PycharmProjects/wundernn-connectome; }
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/wnn/src","/content/wnn/logs","/content/wnn/out","/content/wnn/teach","/content/wnn/wnn_connectome_starterpack")]; print("dirs ok")' | timeout 120 $C repl -s $S >> $L 2>&1
tools/colab_rclone.sh $S >> $L 2>&1 || { echo "$(date +%H:%M) RCLONE SETUP FAILED on $S, session kept" >> $L; exit 1; }
echo "$(date +%H:%M) rclone ok on $S" >> $L
for f in $SRC/src/*.py; do timeout 120 $C upload -s $S $f /content/wnn/src/$(basename $f) < /dev/null > /dev/null 2>&1 || echo "upload failed $f" >> $L; done
timeout 120 $C upload -s $S wnn_connectome_starterpack/utils.py /content/wnn/wnn_connectome_starterpack/utils.py < /dev/null > /dev/null 2>&1 || echo "upload failed utils" >> $L
TEACH=""
for f in $(cat tools/teach13.txt); do t=/content/wnn/teach/$(basename $(dirname $f))_$(basename $f); TEACH="$TEACH $t"
  timeout 300 $C upload -s $S $f $t < /dev/null > /dev/null 2>&1 || { echo "$(date +%H:%M) upload failed $f, abort" >> $L; exit 1; }; done
echo "$(date +%H:%M) 13 teachers uploaded" >> $L
cat > /tmp/wnn_day12b.py <<PY
import os, subprocess, threading, time
subprocess.run("pip -q install onnx > /content/wnn/logs/pip_onnx.log 2>&1", shell=True)
t = time.time()
r = subprocess.run("rclone copy gdrive:wnn/data /content/wnn/out/data --transfers 8", shell=True)
open("/content/wnn/logs/drive_restore.log", "w").write(f"rclone rc {r.returncode} in {time.time() - t:.0f}s\n")
ARMS = "$ARMS".split()
def push():
    while True:
        for a in ARMS + ["teach13", "teachbi"]:
            if os.path.isdir(f"/content/wnn/out/{a}"):
                subprocess.run(f"rclone copy /content/wnn/out/{a} gdrive:wnn/out/{a}", shell=True)
        time.sleep(300)
threading.Thread(target=push, daemon=True).start()
def sh(cmd, log):
    r = subprocess.run(f"cd /content/wnn && {cmd} >> /content/wnn/logs/{log}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/{log}.log", "a").write(f"\nSTEP EXIT {r.returncode}: {cmd[:120]}\n")
    return r.returncode
def dump(out, ckpts, log):
    rc = sum(sh(f"python -u src/dump_teacher.py out/data {n} {out} {ckpts}", log) for n in ("train", "valid"))
    subprocess.run(f"rclone copy /content/wnn/{out} gdrive:wnn/{out}", shell=True)
    open(f"/content/wnn/logs/{log}.log", "a").write(f"\nEXEC EXIT {rc}\n")
    return rc
base = ("python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--add-valid-from 192 --soft-mask --batch 64 --eval-every 48 --save-every 8 --resume "
        "--hidden 128 --layers 2 --loss wmse --mse-weight 0.5 --chunk 1000 --epochs 8 --lr 1e-3 --seed 0")   # = c1000_lr1e3_e8
extra = {"student13_s0": "--teacher-preds out/teach13 --distill 0.5", "bidir_s0": "--bidir",
         "studentbi_s0": "--teacher-preds out/teachbi --distill 0.5"}
if dump("out/teach13", "$TEACH", "dump13"):
    raise SystemExit("dump13 failed")
for a in ("student13_s0", "bidir_s0"):
    rc = sh(f"{base} {extra[a]} --max-batches 1 --out out/smoke_{a}", f"smoke_{a}")
    if rc:
        open("/content/wnn/logs/smoke_day12b.log", "w").write(f"EXEC EXIT {rc}\n"); raise SystemExit(f"smoke {a} failed")
open("/content/wnn/logs/smoke_day12b.log", "w").write("EXEC EXIT 0\n")
def run(a):
    r = subprocess.run(f"cd /content/wnn && {base} {extra[a]} --out out/{a} > /content/wnn/logs/{a}.log 2>&1", shell=True)
    subprocess.run(f"rclone copy /content/wnn/out/{a} gdrive:wnn/out/{a}", shell=True)
    open(f"/content/wnn/logs/{a}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
    return r.returncode
def bidir_then_student():
    if run("bidir_s0"):
        open("/content/wnn/logs/studentbi_s0.log", "w").write("bidir_s0 failed\nEXEC EXIT 1\n"); return
    if dump("out/teachbi", "out/bidir_s0/best.pt", "dumpbi"):
        open("/content/wnn/logs/studentbi_s0.log", "w").write("dumpbi failed\nEXEC EXIT 1\n"); return
    run("studentbi_s0")
th = [threading.Thread(target=run, args=("student13_s0",)), threading.Thread(target=bidir_then_student)]
[t_.start() for t_ in th]; [t_.join() for t_ in th]
PY
for a in $ARMS smoke_day12b dump13 dumpbi; do timeout 60 $C rm -s $S /content/wnn/logs/$a.log < /dev/null > /dev/null 2>&1; rm -f logs/colab_$a.log; done
$C exec -s $S -f /tmp/wnn_day12b.py < /dev/null >> $L 2>&1 &
echo "$(date +%H:%M) job submitted: rclone data copy (~27 min), 13-teacher dump, smoke x2, then student13 || (bidir -> dump -> studentbi)" >> $L
until timeout 60 $C download -s $S /content/wnn/logs/smoke_day12b.log logs/colab_smoke_day12b.log < /dev/null > /dev/null 2>&1 && grep -q "^EXEC EXIT" logs/colab_smoke_day12b.log; do
  timeout 60 $C download -s $S /content/wnn/logs/dump13.log logs/colab_dump13.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT [1-9]" logs/colab_dump13.log 2>/dev/null && { echo "$(date +%H:%M) DUMP13 FAILED, session kept" >> $L; exit 1; }
  sleep 120; done
grep -q "^EXEC EXIT 0" logs/colab_smoke_day12b.log || { echo "$(date +%H:%M) SMOKE FAILED, session kept" >> $L; exit 1; }
echo "$(date +%H:%M) dump13 + smoke ok ($(grep -o 'DUMP DONE.*' logs/colab_dump13.log | tr '\n' ' ')), arms running" >> $L
wait_all $ARMS
[ $FAILED = 1 ] && { echo "$(date +%H:%M) an arm FAILED, session $S kept for inspection" >> $L; exit 1; }
$C stop -s $S < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 12b DONE, colab session stopped" >> $L
