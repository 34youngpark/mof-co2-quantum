#!/usr/bin/env python3
import re, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
def parse(files):
    pts = {}
    for f in files:
        for l in open(os.path.expanduser(f)):
            m = re.search(r"nops=\s*(\d+).*ΔFCI=\s*([-\d.]+)", l)
            if m: pts[int(m.group(1))] = float(m.group(2))
    k = sorted(pts); return k, [pts[i] for i in k]
runs = {"b: MOF+CO$_2$ (carbamate)": ["~/adapt_b20_part1.out", "~/adapt_b20_part2.out", "~/adapt_b20.out"],
        "c: MOF+CO$_2$ (pore)": ["~/adapt_c20_part1.out", "~/adapt_c20.out"],
        "a: MOF": ["~/adapt_a20_part1.out", "~/adapt_a20.out"],
        "CO$_2$ 8q": ["~/adapt_CO2.out"]}
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
for lab, fs in runs.items():
    x, y = parse(fs)
    if x: ax.plot(x, y, "-", lw=1.4, label=f"{lab}")
ax.axhline(4.2, ls="--", c="gray", lw=0.8); ax.text(2, 5, "chemical accuracy", fontsize=7, color="gray")
ax.set_yscale("log"); ax.set_xlabel("ADAPT operators"); ax.set_ylabel(r"$E_{\rm VQE}-E_{\rm CASCI}$ [kJ/mol]")
ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("ADAPT-VQE convergence, (10e,10o) spaces", fontsize=9)
plt.tight_layout(); plt.savefig("paper/figs/fig4_adapt.png"); print("→ paper/figs/fig4_adapt.png")