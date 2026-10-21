# Run on a staged Colab session with Drive mounted: arm out/ dirs become symlinks into Drive, and the prepared
# memmaps are copied to Drive in the background (log /content/wnn/drive_copy.log, ends with DATA COPIED).
import os, subprocess
D = "/content/drive/MyDrive/wnn"
os.makedirs(f"{D}/out", exist_ok=True)
for a in ("c1000_lr1e3_e8", "day_dozken", "c2000_lr1e3_e8", "c1000_lr1e3_e8_s1"):
    src, dst = f"/content/wnn/out/{a}", f"{D}/out/{a}"
    os.makedirs(dst, exist_ok=True)
    if not os.path.islink(src):
        if os.path.isdir(src): os.rmdir(src)
        os.symlink(dst, src)
open("/content/wnn/drive_links.txt","w").write(str({a: os.readlink(f"/content/wnn/out/{a}") for a in os.listdir("/content/wnn/out") if os.path.islink(f"/content/wnn/out/{a}")})+"\n")
# after staging: copy the prepared memmaps and the stage log to Drive, in the background
subprocess.Popen(["bash", "-c", f"until grep -q 'STAGE DONE' /content/wnn/stage.log; do sleep 30; done; "
                  f"mkdir -p {D}/data && cp -r /content/wnn/out/data/. {D}/data/ && cp /content/wnn/stage.log {D}/ && "
                  f"echo \"$(date) DATA COPIED $(du -sh {D}/data)\" >> /content/wnn/drive_copy.log"],
                 stdout=open("/content/wnn/drive_copy.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
print("copy job started")
