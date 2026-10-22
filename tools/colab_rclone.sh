#!/bin/bash
# Give a Colab session Drive access with Alex's own rclone token instead of `colab drivemount` (no browser step).
# One-time on the Mac: rclone config create gdrive drive scope=drive
# Usage: tools/colab_rclone.sh <session>   -> installs rclone on the VM, uploads ~/.config/rclone/rclone.conf,
# checks `rclone lsd gdrive:wnn`; exit 0 only if the VM can list Drive.
# On the VM afterwards:  rclone copy gdrive:wnn/data /content/wnn/out/data --transfers 8
#                        rclone copy /content/wnn/out/<arm> gdrive:wnn/out/<arm>      (checkpoint push)
C=~/.local/bin/colab; S=$1
[ -n "$S" ] || { echo "usage: $0 <session>"; exit 2; }
cat > /tmp/wnn_rclone_setup.py <<'PY'
import os, subprocess
os.makedirs("/root/.config/rclone", exist_ok=True)
if subprocess.run("which rclone", shell=True, capture_output=True).returncode:
    subprocess.run("cd /tmp && curl -sSL -o rc.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip && "
                   "unzip -oq rc.zip && cp rclone-*-linux-amd64/rclone /usr/local/bin/ && chmod 755 /usr/local/bin/rclone",
                   shell=True, check=True)
r = subprocess.run("rclone lsd gdrive:wnn", shell=True, capture_output=True, text=True)
print(r.stdout); print("RCLONE OK" if r.returncode == 0 and " data" in r.stdout else "RCLONE FAILED " + r.stderr[-300:])
PY
echo 'import os; os.makedirs("/root/.config/rclone", exist_ok=True); print("dir ok")' | timeout 120 $C repl -s $S < /dev/stdin > /dev/null 2>&1
timeout 120 $C upload -s $S ~/.config/rclone/rclone.conf /root/.config/rclone/rclone.conf < /dev/null > /dev/null 2>&1 || { echo "conf upload failed"; exit 1; }
out=$(timeout 300 $C exec -s $S -f /tmp/wnn_rclone_setup.py < /dev/null 2>&1)
echo "$out" | grep -E "RCLONE|data|out"
echo "$out" | grep -q "RCLONE OK"
