"""Fetch train[:192] (X, Y, P) from gdrive:wnn/data by byte range (128-byte .npy headers, float16, 20000 rows)."""
import subprocess, numpy as np
N, L = 192, 20000
for name, tail in (("X", (112,)), ("Y", (2,)), ("P", ())):
    shape = (N, L) + tail
    nbytes = int(np.prod(shape)) * 2
    raw = subprocess.run(["rclone", "cat", "--offset", "128", "--count", str(nbytes), f"gdrive:wnn/data/train_{name}.npy"],
                         capture_output=True, check=True).stdout
    assert len(raw) == nbytes, (name, len(raw), nbytes)
    a = np.lib.format.open_memmap(f"out/data/train192_{name}.npy", mode="w+", dtype=np.float16, shape=shape)
    a[:] = np.frombuffer(raw, dtype=np.float16).reshape(shape); a.flush()
    print(name, shape, "ok", flush=True)
