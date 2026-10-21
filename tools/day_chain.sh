#!/bin/bash
# Day chain: wait for the two Fable arms (remote markers), then run the global Pearson loss and two cheap arms, pull, stop.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab
L=logs/overnight_final.log
remote_done() { timeout 90 $C download -s wnn /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
wait_for() { for a in "$@"; do until remote_done $a; do sleep 300; done; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1)" >> $L; done; }
pull() { for a in "$@"; do mkdir -p out/$a; timeout 300 $C download -s wnn /content/wnn/out/$a/best.pt out/$a/best.pt < /dev/null > /dev/null 2>&1; done; }
started() { for i in $(seq 1 20); do timeout 60 $C download -s wnn /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1 && grep -q "batch 1/" logs/colab_$1.log && return 0; sleep 30; done; return 1; }

echo "$(date +%H:%M) day chain start" >> $L
wait_for morning_norm morning_ema8
pull morning_norm morning_ema8
$C upload -s wnn src/train_gru.py /content/wnn/src/train_gru.py < /dev/null >> $L 2>&1
cat > /tmp/wnn_day.py <<'PY'
import subprocess
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--soft-mask --add-valid-from 192 --batch 64 --epochs 4 --eval-every 48 --save-every 8 --resume --seed 0 --hidden 128 --layers 2")
for name, extra in [("day_pearson", "--loss pearson --mse-weight 0.5 --chunk 500"),
                    ("day_chunk1000", "--loss wmse --chunk 1000"),
                    ("day_lr1e3", "--loss wmse --chunk 500 --lr 1e-3")]:
    r = subprocess.run(f"{base} {extra} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
for try in 1 2 3; do
    $C exec -s wnn -f /tmp/wnn_day.py < /dev/null >> $L 2>&1 &
    started day_pearson && break
    echo "$(date +%H:%M) day arms did not start (try $try), resubmitting" >> $L
done
wait_for day_pearson day_chunk1000 day_lr1e3
pull day_pearson day_chunk1000 day_lr1e3
$C stop -s wnn < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN DONE, colab session stopped" >> $L
