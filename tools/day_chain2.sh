#!/bin/bash
# Day chain 2 (replaces the tail of day_chain.sh, whose remote arms keep running): wait for the four day arms,
# pull them, then chunk follow-ups on the same session (data already staged), pull, stop.
# Why: day_chunk1000 0.68689 and day_lr1e3 0.68821 vs full1 0.68310 (seed spread +-0.0012); both still rising at epoch ends.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab
L=logs/overnight_final.log
remote_done() { timeout 90 $C download -s wnn /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
wait_for() { for a in "$@"; do until remote_done $a; do sleep 300; done; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1)" >> $L; done; }
pull() { for a in "$@"; do mkdir -p out/$a; timeout 300 $C download -s wnn /content/wnn/out/$a/best.pt out/$a/best.pt < /dev/null > /dev/null 2>&1; done; }
started() { for i in $(seq 1 20); do timeout 60 $C download -s wnn /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1 && grep -q "batch 1/" logs/colab_$1.log && return 0; sleep 30; done; return 1; }

echo "$(date +%H:%M) day chain 2 start" >> $L
wait_for day_pearson day_chunk1000 day_lr1e3 day_dozken
pull day_pearson day_chunk1000 day_lr1e3 day_dozken
cat > /tmp/wnn_day2.py <<'PY'
import subprocess
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--soft-mask --add-valid-from 192 --batch 64 --epochs 4 --eval-every 48 --save-every 8 --resume --seed 0 --hidden 128 --layers 2 --loss wmse")
for name, extra in [("c1000_lr1e3", "--chunk 1000 --lr 1e-3"),          # the two day wins combined (chunk1000 +0.0038, lr1e3 +0.0051)
                    ("c2000_lr1e3", "--chunk 2000 --lr 1e-3"),          # does the chunk trend continue past 1000?
                    ("c1000_lr1e3_s1", "--chunk 1000 --lr 1e-3 --seed 1"),  # seed check on the combination
                    ("c1000_lr1e3_e8", "--chunk 1000 --lr 1e-3 --epochs 8")]:  # longer schedule on the combination
    r = subprocess.run(f"{base} {extra} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
for try in 1 2 3; do
    $C exec -s wnn -f /tmp/wnn_day2.py < /dev/null >> $L 2>&1 &
    started c1000_lr1e3 && break
    echo "$(date +%H:%M) day2 arms did not start (try $try), resubmitting" >> $L
done
wait_for c1000_lr1e3 c2000_lr1e3 c1000_lr1e3_s1 c1000_lr1e3_e8
pull c1000_lr1e3 c2000_lr1e3 c1000_lr1e3_s1 c1000_lr1e3_e8
$C stop -s wnn < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 2 DONE, colab session stopped" >> $L
