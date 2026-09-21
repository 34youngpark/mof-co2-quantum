import numpy as np
KJ = 2625.4996
r = {s: np.load(f'thr_{s}.npz') for s in 'abcd'}
thr = r['b']['thr']
print(" threshold |   E_MP2(a)      E_MP2(b)      E_MP2(c)      E_MP2(d)   | b-c (kJ/mol)")
for k, t in enumerate(thr):
    e = [float(r[s]['E_mp2'][k]) for s in 'abcd']
    print(f" {t:8.0e} | {e[0]:12.6f}  {e[1]:12.6f}  {e[2]:12.6f}  {e[3]:12.6f} | {(e[1]-e[2])*KJ:8.2f}")
e = [float(r[s]['E_full']) for s in 'abcd']
print(f" full-vir | {e[0]:12.6f}  {e[1]:12.6f}  {e[2]:12.6f}  {e[3]:12.6f} | {(e[1]-e[2])*KJ:8.2f}")
