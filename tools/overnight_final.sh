#!/bin/bash
# Overnight chain, driven by REMOTE completion markers: each arm appends "EXEC EXIT <rc>" to its own log on Colab.
# The colab exec client can time out while the remote job keeps running (seen 2026-10-21 00:49), so the client's
# exit is never used as "done". Steps: wait arms 2 -> run time arm -> pick recipe -> 2 final seeds -> Fable's two
# ideas -> pull checkpoints -> stop the session.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab
L=logs/overnight_final.log

remote_done() {   # $1 = arm name; true once its remote log ends with an EXEC EXIT line
    timeout 90 $C download -s wnn /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1
    grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null
}
wait_for() {      # wait until every named arm is done remotely
    for a in "$@"; do
        until remote_done $a; do sleep 300; done
        echo "$(date +%H:%M) $a done: $(grep EVAL logs/colab_$a.log | tail -1 | cut -c1-80)" >> $L
    done
}
pull() {          # pull each arm's best.pt
    for a in "$@"; do mkdir -p out/$a; timeout 300 $C download -s wnn /content/wnn/out/$a/best.pt out/$a/best.pt < /dev/null > /dev/null 2>&1; done
}

echo "$(date +%H:%M) chain v3 start: time arm dropped (it builds on full_dps, which lost by 0.005)" >> $L
wait_for full_dps full_all
pull full_h192 full_l3 full_reset2k full_dps full_all

FLAGS=$(.venv/bin/python tools/pick_final.py)
echo "$(date +%H:%M) recipe: $FLAGS" >> $L; cat docs/final-recipe.md >> $L
cat > /tmp/wnn_final.py <<PY
import subprocess
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--loss wmse --soft-mask --add-valid-from 192 --batch 64 --chunk 500 --epochs 4 --eval-every 48 --save-every 8 --resume $FLAGS")
for seed in (1, 2):   # full1 is seed 0 of the base recipe
    r = subprocess.run(f"{base} --seed {seed} --out out/final_s{seed} > /content/wnn/logs/final_s{seed}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/final_s{seed}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
echo "$(date +%H:%M) final runs submitted" >> $L
$C exec -s wnn -f /tmp/wnn_final.py < /dev/null >> $L 2>&1 &
wait_for final_s1 final_s2
pull final_s1 final_s2

# Fable ideas 1 and 2 on top of the picked recipe, seed 0, paired with full1 (seed 0) when the recipe is the base one
for f in src/model.py src/train_gru.py; do $C upload -s wnn $f /content/wnn/$f < /dev/null >> $L 2>&1; done
cat > /tmp/wnn_morning.py <<PY
import subprocess
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--loss wmse --soft-mask --add-valid-from 192 --batch 64 --chunk 500 --eval-every 48 --save-every 8 --resume --seed 0 $FLAGS")
for name, extra in [("morning_norm", "--epochs 4 --loss-norm batch"), ("morning_ema8", "--epochs 8 --ema-weights 0.999")]:
    r = subprocess.run(f"{base} {extra} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\\nEXEC EXIT {r.returncode}\\n")
PY
echo "$(date +%H:%M) morning arms submitted" >> $L
$C exec -s wnn -f /tmp/wnn_morning.py < /dev/null >> $L 2>&1 &
wait_for morning_norm morning_ema8
pull morning_norm morning_ema8
ls -la out/final_s*/best.pt out/full_*/best.pt out/morning_*/best.pt >> $L 2>&1
$C stop -s wnn < /dev/null >> $L 2>&1
echo "$(date +%H:%M) OVERNIGHT DONE, colab session stopped" >> $L
