#!/usr/bin/env python3
"""Fig. 1: a/b/c/d 구조 (c축 방향 투영, 2x2 복제로 육각 채널 표시), fragment 원자 강조."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ase.io import read
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
FRAG = {"a_MOF": [215, 209], "b_MOF_CO2_bound": [194, 195, 216, 217, 218],
        "c_MOF_CO2_free": [216, 217, 218, 215, 209], "d_CO2_box": [0, 1, 2]}
TITLE = {"a_MOF": "(a) MOF", "b_MOF_CO2_bound": "(b) MOF + CO$_2$ (carbamate)",
         "c_MOF_CO2_free": "(c) MOF + CO$_2$ (in pore)", "d_CO2_box": "(d) CO$_2$"}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), dpi=200, gridspec_kw={"width_ratios": [1, 1, 1, 0.55]})
for ax, (name, frag) in zip(axes, FRAG.items()):
    at0 = read(f"structures/02_relaxed/{name}_relaxed.cif")
    at = at0.repeat((2, 2, 1)) if name != "d_CO2_box" else at0
    n0 = len(at0); pos = at.positions; z = at.numbers
    for i in np.argsort(pos[:, 2]):
        f = i in frag                      # 첫 복사본(원본 셀)만 강조
        r = covalent_radii[z[i]] * (2.0 if f else 0.9)
        ax.scatter(pos[i, 0], pos[i, 1], s=(r*36)**2/10, c=[jmol_colors[z[i]]],
                   edgecolors="k" if f else "gray", linewidths=1.2 if f else 0.25,
                   alpha=1.0 if (f or i < n0) else 0.35, zorder=pos[i, 2])
    if name != "d_CO2_box":
        a1, a2 = at0.cell[0][:2], at0.cell[1][:2]
        poly = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
        ax.plot(poly[:, 0], poly[:, 1], "k--", lw=0.7)
    ax.set_aspect("equal"); ax.set_title(TITLE[name], fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if name == "d_CO2_box": ax.set_xlim(4.5, 10.5); ax.set_ylim(4.5, 10.5)
plt.tight_layout(); plt.savefig("paper/figs/fig1_structures.png"); plt.savefig("paper/figs/fig1_structures.pdf")
print("→ paper/figs/fig1_structures.png")