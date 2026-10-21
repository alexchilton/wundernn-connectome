#!/bin/bash
# Day chain 3: session wnn died ~10:20 with day_pearson/chunk1000/lr1e3 checkpoints unpulled. Fresh session wnn2:
# stage data, then the combination arms and the dozken control. Each arm's best.pt is pulled as soon as it ends,
# Every arm's best.pt and latest.pt are mirrored to out/<arm>/ every 5 min while it trains, so a dying session
# loses at most 5 minutes; resume on a new session by uploading latest.pt (the arms run with --resume).
cd ~/PycharmProjects/wundernn-connectome
C=~/.local/bin/colab; S=${1:-wnn3}
L=logs/overnight_final.log
remote_done() { timeout 90 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1; grep -q "^EXEC EXIT" logs/colab_$1.log 2>/dev/null; }
# pull to a temp name first so a failed or partial download never overwrites the last good local copy
pull() { for a in "$@"; do mkdir -p out/$a; for f in best latest; do timeout 300 $C download -s $S /content/wnn/out/$a/$f.pt out/$a/.$f.tmp < /dev/null > /dev/null 2>&1 && mv out/$a/.$f.tmp out/$a/$f.pt; done; done; }
# mirror best.pt and latest.pt (the --resume state) every 5 min while the arm trains, not only at its end
wait_for() { for a in "$@"; do until remote_done $a; do pull $a; sleep 300; done; pull $a; echo "$(date +%H:%M) $a done: best $(grep -o 'WP [0-9.]*' logs/colab_$a.log | awk '{print $2}' | sort -r | head -1), pulled $(ls -la out/$a/best.pt 2>/dev/null | awk '{print $5}')" >> $L; done; }
started() { for i in $(seq 1 20); do timeout 60 $C download -s $S /content/wnn/logs/$1.log logs/colab_$1.log < /dev/null > /dev/null 2>&1 && grep -q "batch 1/" logs/colab_$1.log && return 0; sleep 30; done; return 1; }

echo "$(date +%H:%M) day chain 3 start (wnn died ~10:20)" >> $L
timeout 60 $C sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { cd /tmp && timeout 600 $C new -s $S --gpu L4 < /dev/null >> ~/PycharmProjects/wundernn-connectome/$L 2>&1; cd ~/PycharmProjects/wundernn-connectome; }
if ! { timeout 60 $C download -s $S /content/wnn/stage.log logs/colab_stage_$S.log < /dev/null > /dev/null 2>&1 && grep -q "START" logs/colab_stage_$S.log; }; then
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/wnn/src","/content/wnn/tools","/content/wnn/out/mask1","/content/wnn/logs","/content/wnn/wnn_connectome_starterpack")]; print("dirs ok")' | timeout 120 $C repl -s $S >> $L 2>&1
for f in src/*.py tools/fetch_zip_member.py tools/colab_stage.py; do timeout 120 $C upload -s $S $f /content/wnn/$f < /dev/null > /dev/null 2>&1 || echo "upload failed $f" >> $L; done
timeout 300 $C upload -s $S out/mask1/best.pt /content/wnn/out/mask1/best.pt < /dev/null > /dev/null 2>&1 || echo "upload failed mask1" >> $L
echo "$(date +%H:%M) staging" >> $L
$C exec -s $S -f tools/colab_stage.py < /dev/null >> $L 2>&1 &
fi
until timeout 60 $C download -s $S /content/wnn/stage.log logs/colab_stage_$S.log < /dev/null > /dev/null 2>&1 && grep -q "STAGE DONE" logs/colab_stage_$S.log; do sleep 180; done
echo "$(date +%H:%M) staged: $(tail -2 logs/colab_stage_$S.log | cut -c1-150 | tr '\n' ' ')" >> $L
# Drive (mounted once per session by Alex with `colab drivemount -s <session>`): arm outputs live on Drive via symlinks, and the
# prepared memmaps are copied there, so a dead session loses neither checkpoints nor the 40-minute staging.
timeout 900 $C exec -s $S -f tools/colab_drive.py < /dev/null >> $L 2>&1
echo "$(date +%H:%M) drive: $(timeout 60 $C download -s $S /content/wnn/drive_links.txt /dev/stdout < /dev/null 2>/dev/null | tail -1)" >> $L
cat > /tmp/wnn_day3.py <<'PY'
import subprocess
base = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
        "--soft-mask --add-valid-from 192 --batch 64 --epochs 4 --eval-every 48 --save-every 8 --resume --seed 0 --hidden 128 --layers 2")
for name, extra in [("c1000_lr1e3_e8", "--loss wmse --chunk 1000 --lr 1e-3 --epochs 8"),       # both day wins, 8 epochs (both were still rising at epoch 4)
                    ("day_dozken", "--hidden 192 --proj 128 --diff --loss pearson --mse-weight 0.5 --state-reset 2000 "
                                   "--chunk 1000 --lr 1e-3 --ema-weights 0.999 --epochs 5"),   # control, rerun (died at epoch 2)
                    ("c2000_lr1e3_e8", "--loss wmse --chunk 2000 --lr 1e-3 --epochs 8"),       # does the chunk trend continue?
                    ("c1000_lr1e3_e8_s1", "--loss wmse --chunk 1000 --lr 1e-3 --epochs 8 --seed 1")]:  # seed check, and the second fused member
    r = subprocess.run(f"{base} {extra} --out out/{name} > /content/wnn/logs/{name}.log 2>&1", shell=True)
    open(f"/content/wnn/logs/{name}.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
for try in 1 2 3; do
    $C exec -s $S -f /tmp/wnn_day3.py < /dev/null >> $L 2>&1 &
    started c1000_lr1e3_e8 && break
    echo "$(date +%H:%M) day3 arms did not start (try $try), resubmitting" >> $L
done
wait_for c1000_lr1e3_e8 day_dozken c2000_lr1e3_e8 c1000_lr1e3_e8_s1
# appended 16:25: the dozken-ish recipe at 128 wide (day_dozken 192 fused with e8 = 0.69017 held-out but 42 us/call,
# ~75 min on the platform); at 128 the fused pair is 256 wide like the uploads that ran 44-47 min
cat > /tmp/wnn_day3b.py <<'PY'
import os, subprocess
D = "/content/drive/MyDrive/wnn"
os.makedirs(f"{D}/out/dzish128", exist_ok=True)
if not os.path.islink("/content/wnn/out/dzish128"):
    os.symlink(f"{D}/out/dzish128", "/content/wnn/out/dzish128")
cmd = ("cd /content/wnn && python -u src/train_gru.py --data out/data --train-name train --valid-name valid --n-valid 192 "
       "--soft-mask --add-valid-from 192 --batch 64 --eval-every 48 --save-every 8 --resume --seed 0 --layers 2 "
       "--hidden 128 --proj 128 --diff --loss pearson --mse-weight 0.5 --state-reset 2000 --chunk 1000 --lr 1e-3 "
       "--ema-weights 0.999 --epochs 5 --out out/dzish128 > /content/wnn/logs/dzish128.log 2>&1")
r = subprocess.run(cmd, shell=True)
open("/content/wnn/logs/dzish128.log", "a").write(f"\nEXEC EXIT {r.returncode}\n")
PY
$C exec -s $S -f /tmp/wnn_day3b.py < /dev/null >> $L 2>&1 &
started dzish128 || echo "$(date +%H:%M) dzish128 did not start" >> $L
wait_for dzish128
$C stop -s $S < /dev/null >> $L 2>&1
echo "$(date +%H:%M) DAY CHAIN 3 DONE, colab session stopped" >> $L
