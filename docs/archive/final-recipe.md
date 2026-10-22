# Final recipe, picked automatically (rule in tools/pick_final.py)

reset 2000: full_reset2k 0.68378 vs full1 0.6831 -> out
diff+proj+skip: full_dps 0.67807 vs full1 0.6831 -> out
EMA 10,100,1000: full_all 0.67829 vs full_dps 0.67807 -> out
time feature: full_dpst None vs full_dps 0.67807 -> out
width/depth: h192 0.68121, l3 0.68075 vs full1 0.6831 -> out

flags: --hidden 128 --layers 2
