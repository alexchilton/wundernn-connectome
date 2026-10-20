#!/bin/bash
# After all of tonight's arms: pick the recipe by tools/pick_final.py, train 2 seeds on Colab, pull, stop the session.
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab
L=logs/overnight_final.log
until grep -q "^EXEC_RC=" logs/colab_arms3_exec.log 2>/dev/null; do sleep 120; done
for a in full_h192 full_l3 full_reset2k full_dps full_all full_dpst; do timeout 90 $C download -s wnn /content/wnn/logs/$a.log logs/colab_$a.log < /dev/null > /dev/null 2>&1; done
FLAGS=$(.venv/bin/python tools/pick_final.py)
echo "$(date +%H:%M) recipe: $FLAGS" >> $L; cat docs/final-recipe.md >> $L
cat > /tmp/wnn_final.py <<PY
import subprocess
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--loss wmse --soft-mask --add-valid-from 192 --batch 64 --chunk 500 --epochs 4 --eval-every 48 --save-every 8 --resume $FLAGS")
for seed in (0, 1):
    r = subprocess.run(f"{base} --seed {seed} --out out/final_s{seed} > /content/wnn/logs/final_s{seed}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/final_s{seed}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
echo "$(date +%H:%M) final runs start" >> $L
$C exec -s wnn -f /tmp/wnn_final.py < /dev/null >> $L 2>&1; echo "$(date +%H:%M) EXEC_RC=$?" >> $L
for s in 0 1; do timeout 90 $C download -s wnn /content/wnn/logs/final_s$s.log logs/colab_final_s$s.log < /dev/null > /dev/null 2>&1; mkdir -p out/final_s$s; timeout 300 $C download -s wnn /content/wnn/out/final_s$s/best.pt out/final_s$s/best.pt < /dev/null >> $L 2>&1; grep EVAL logs/colab_final_s$s.log | tail -2 >> $L; done
for a in full_h192 full_l3 full_reset2k full_all; do mkdir -p out/$a; [ -s out/$a/best.pt ] || timeout 300 $C download -s wnn /content/wnn/out/$a/best.pt out/$a/best.pt < /dev/null > /dev/null 2>&1; done
ls -la out/final_s*/best.pt out/full_*/best.pt >> $L 2>&1
$C stop -s wnn < /dev/null >> $L 2>&1
echo "$(date +%H:%M) OVERNIGHT DONE, colab session stopped" >> $L
