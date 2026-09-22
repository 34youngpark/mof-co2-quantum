#!/usr/bin/env python3
"""Fig. 3: threshold vs ΔE (a) + 계별 상관 회수 (b). thr_{a,b,c,d}.npz 사용."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
KJ = 2625.4996
r = {s: np.load(f"thr_{s}.npz") for s in "abcd"}
thr = r["b"]["thr"]; m = thr <= 1e-4                      # pre-convergence 구간 제외
hf1 = (r["b"]["E_hf"]-r["a"]["E_hf"]-r["d"]["E_hf"])*KJ; hf2 = (r["b"]["E_hf"]-r["c"]["E_hf"])*KJ
c1 = (r["b"]["E_mp2"]-r["a"]["E_mp2"]-r["d"]["E_mp2"])*KJ; c2 = (r["b"]["E_mp2"]-r["c"]["E_mp2"])*KJ
f1 = (r["b"]["E_full"]-r["a"]["E_full"]-r["d"]["E_full"])*KJ; f2 = (r["b"]["E_full"]-r["c"]["E_full"])*KJ
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
ax[0].semilogx(thr[m], hf1+c1[m], "o-", label=r"$\Delta E_1$ (Eq. 1)")
ax[0].semilogx(thr[m], hf2+c2[m], "s-", label=r"$\Delta E_2$ (Eq. 2)")
ax[0].axhline(hf1+f1, ls="--", c="C0", lw=0.8); ax[0].axhline(hf2+f2, ls="--", c="C1", lw=0.8)
ax[0].invert_xaxis(); ax[0].set_xlabel("NO occupation threshold"); ax[0].set_ylabel(r"$\Delta E$ (HF+MP2) [kJ/mol]")
ax[0].legend(fontsize=8); ax[0].set_title("(a) Adsorption energy convergence", fontsize=9)
for s, lab in zip("abcd", ["a: MOF", "b: MOF+CO$_2$ (bound)", "c: MOF+CO$_2$ (free)", "d: CO$_2$"]):
    ax[1].semilogx(thr[m], (r[s]["E_mp2"][m]-r[s]["E_full"])*KJ, "o-", ms=3, label=lab)
ax[1].invert_xaxis(); ax[1].set_xlabel("NO occupation threshold"); ax[1].set_ylabel(r"$E_{corr}(\tau) - E_{corr}(\infty)$ [kJ/mol]")
ax[1].legend(fontsize=7); ax[1].set_title("(b) Per-system correlation recovery", fontsize=9)
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig3_convergence.png"); plt.savefig("fig3_convergence.pdf")
print(f"limits: dE1={hf1+f1:.2f} dE2={hf2+f2:.2f} kJ/mol → fig3_convergence.png/pdf")