"""Pick the final recipe from tonight's full-data arms by a rule fixed before the results (2026-10-20 23:20).

Each change joins only if its best held-out WP beats its comparison run by more than the margin:
  reset 2000          full_reset2k vs full1      > 0.005
  diff+proj+skip      full_dps     vs full1      > 0.005
  EMA (10,100,1000)   full_all     vs full_dps   > 0.005
  time feature        full_dpst    vs full_dps   > 0.003
  width 192           full_h192    vs full1      > 0.005
  depth 3             full_l3      vs full1      > 0.005
Prints the flags for train_gru.py and writes docs/final-recipe.md.
"""
import re, sys
def best(name):
    try:
        v = [float(x) for x in re.findall(r"EVAL .* WP ([0-9.]+)", open(f"logs/colab_{name}.log").read())]
        return max(v) if v else None
    except FileNotFoundError:
        return None
b = {n: best(n) for n in ["full1", "full_h192", "full_l3", "full_reset2k", "full_dps", "full_all", "full_dpst"]}
def wins(a, ref, m):
    return b[a] is not None and b[ref] is not None and b[a] - b[ref] > m
flags, lines = ["--hidden", "128", "--layers", "2"], []
rules = [("reset 2000", "full_reset2k", "full1", .005, ["--state-reset", "2000"]),
         ("diff+proj+skip", "full_dps", "full1", .005, ["--diff", "--proj", "128", "--skip"]),
         ("EMA 10,100,1000", "full_all", "full_dps", .005, ["--ema", "10,100,1000"]),
         ("time feature", "full_dpst", "full_dps", .003, ["--time"])]
for label, a, ref, m, f in rules:
    ok = wins(a, ref, m) and (label not in ("EMA 10,100,1000", "time feature") or wins("full_dps", "full1", .005))
    lines.append(f"{label}: {a} {b[a]} vs {ref} {b[ref]} -> {'IN' if ok else 'out'}")
    if ok: flags += f
if wins("full_h192", "full1", .005) and not wins("full_l3", "full_h192", 0):
    flags[1] = "192"; lines.append(f"width: full_h192 {b['full_h192']} vs full1 {b['full1']} -> IN")
elif wins("full_l3", "full1", .005):
    flags[3] = "3"; lines.append(f"depth: full_l3 {b['full_l3']} vs full1 {b['full1']} -> IN")
else:
    lines.append(f"width/depth: h192 {b['full_h192']}, l3 {b['full_l3']} vs full1 {b['full1']} -> out")
open("docs/final-recipe.md", "w").write("# Final recipe, picked automatically (rule in tools/pick_final.py)\n\n" + "\n".join(lines) + "\n\nflags: " + " ".join(flags) + "\n")
print(" ".join(flags))
