"""Extract one member of a remote ZIP using HTTP range requests (resumable, verified by CRC)."""
import re, io, os, sys, zipfile, zlib, struct, subprocess

URL, MEMBER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

def get(start, end):
    for attempt in range(20):
        r = subprocess.run(["curl", "-sSf", "--max-time", "300", "-r", f"{start}-{end}", URL], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == end - start + 1:
            return r.stdout
        print(f"retry {attempt} {start}-{end}: rc={r.returncode} got={len(r.stdout)} {r.stderr[-200:]}", flush=True)
    raise SystemExit("range fetch failed")

SIZE = int(re.search(rb"content-length: (\d+)", subprocess.run(["curl", "-sI", URL], capture_output=True).stdout, re.I).group(1))

class HttpFile(io.RawIOBase):
    def __init__(self):
        self.size = SIZE; self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0: return b""
        d = get(self.pos, min(self.pos + n, self.size) - 1); self.pos += len(d); return d

zf = zipfile.ZipFile(HttpFile())
info = zf.getinfo(MEMBER)
print(f"{MEMBER}: compressed {info.compress_size} uncompressed {info.file_size} method {info.compress_type} crc {info.CRC:08x}", flush=True)
hdr = get(info.header_offset, info.header_offset + 29)
n, m = struct.unpack("<HH", hdr[26:30])
start = info.header_offset + 30 + n + m
part = OUT + ".part"; done = os.path.getsize(part) if os.path.exists(part) else 0
if info.compress_type != 0 and done:
    print("deflated member: restarting from 0 (resume needs stored)", flush=True); done = 0
CH = 64 << 20
dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
with open(part, "ab" if done else "wb") as f:
    pos = done
    while pos < info.compress_size:
        d = get(start + pos, start + min(pos + CH, info.compress_size) - 1); pos += len(d)
        f.write(dec.decompress(d) if dec else d)
        print(f"{pos/1e9:.2f}/{info.compress_size/1e9:.2f} GB", flush=True)
crc = 0
with open(part, "rb") as f:
    while b := f.read(CH): crc = zlib.crc32(b, crc)
assert crc == info.CRC and os.path.getsize(part) == info.file_size, f"CRC/size mismatch {crc:08x}"
os.replace(part, OUT); print("OK crc", f"{crc:08x}", flush=True)
