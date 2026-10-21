# Colab staging for a fresh session: starter-pack fetch (zip range requests), memmap prep, soft-mask labels.
# Run by tools/colab_new_session.sh; progress in /content/wnn/stage.log, ends with "STAGE DONE".
import os, subprocess, time
W = "/content/wnn"
def note(m):
    with open(f"{W}/stage.log", "a") as f: f.write(time.strftime("%H:%M:%S ") + m + "\n")
def run(cmd, log):
    note(f"RUN {cmd}")
    r = subprocess.run(f"cd {W} && {cmd} > {W}/{log} 2>&1", shell=True)
    note(f"EXIT {r.returncode} {log}: " + open(f"{W}/{log}").read()[-300:].replace("\n", " | "))
    return r.returncode
note(f"START disk free {os.statvfs('/content').f_bavail * os.statvfs('/content').f_frsize / 1e9:.0f} GB")
os.makedirs(f"{W}/wnn_connectome_starterpack/datasets", exist_ok=True); os.makedirs(f"{W}/out/data", exist_ok=True)
Z = "https://files.wundernn.io/wnn_connectome_starterpack.zip"
for f in ("utils.py",):
    run(f"python -u tools/fetch_zip_member.py {Z} wnn_connectome_starterpack/{f} wnn_connectome_starterpack/{f}", f"fetch_{f}.log")
for f in ("valid.parquet", "train.parquet"):
    p = f"wnn_connectome_starterpack/datasets/{f}"
    if not os.path.exists(f"{W}/{p}"):
        run(f"python -u tools/fetch_zip_member.py {Z} {p} {p}", f"fetch_{f}.log")
run("python -u src/prep.py --parquet wnn_connectome_starterpack/datasets/valid.parquet --out out/data --name valid", "prep_valid.log")
run("python -u src/prep.py --parquet wnn_connectome_starterpack/datasets/train.parquet --out out/data --name train", "prep_train.log")
os.remove(f"{W}/wnn_connectome_starterpack/datasets/train.parquet")
run("python -u src/mask_model.py label --data out/data --out out/mask1 --name train", "label_train.log")
note("STAGE DONE")
