#!/usr/bin/env python3
"""5.2 (6): thr_{a,b,c,d}.npz → ΔE1, ΔE2 수렴 표 (Rocca Table 2 대응)"""
import numpy as np
KJ = 2625.4996
r = {s: np.load(f'thr_{s}.npz') for s in 'abcd'}
thr = r['b']['thr']
hf1 = (r['b']['E_hf'] - r['a']['E_hf'] - r['d']['E_hf']) * KJ
hf2 = (r['b']['E_hf'] - r['c']['E_hf']) * KJ
print(f"HF only:  dE1 = {hf1:7.2f}   dE2 = {hf2:7.2f}  kJ/mol\n")
print(" threshold |  n_tot a    b    c    d |  dE1(HF+MP2)  dE2(HF+MP2) | corr1   corr2")
for k, t in enumerate(thr):
    c1 = (r['b']['E_mp2'][k] - r['a']['E_mp2'][k] - r['d']['E_mp2'][k]) * KJ
    c2 = (r['b']['E_mp2'][k] - r['c']['E_mp2'][k]) * KJ
    n = [int(r[s]['n_tot'][k]) for s in 'abcd']
    print(f" {t:8.0e} | {n[0]:6d} {n[1]:4d} {n[2]:4d} {n[3]:4d} | {hf1+c1:11.2f}  {hf2+c2:11.2f} | {c1:6.2f}  {c2:6.2f}")
c1 = (r['b']['E_full'] - r['a']['E_full'] - r['d']['E_full']) * KJ
c2 = (r['b']['E_full'] - r['c']['E_full']) * KJ
print(f" full-vir |                        | {hf1+c1:11.2f}  {hf2+c2:11.2f} | {c1:6.2f}  {c2:6.2f}")