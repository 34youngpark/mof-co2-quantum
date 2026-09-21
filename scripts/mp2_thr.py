#!/usr/bin/env python3
"""
5.2 (6): threshold별 잘린 NO 비점유 공간에서 MP2 → 수렴 표 재료
입력: chk + as_no_*.npz (⑤ 출력)
사용: python mp2_thr.py --chk ... --no as_no_b.npz --cderi ..._cderi.h5 --out thr_b.npz
"""
import argparse, time
import numpy as np
from pyscf import lib
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.lib import chkfile as pbcchk

ap = argparse.ArgumentParser()
ap.add_argument('--chk', required=True); ap.add_argument('--no', required=True)
ap.add_argument('--cderi', required=True); ap.add_argument('--out', default='thr.npz')
ap.add_argument('--thresholds', default='1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6')
ap.add_argument('--max-mem', type=int, default=20000)
args = ap.parse_args(); t0 = time.time()
thr = [float(x) for x in args.thresholds.split(',')]

cell = pbcchk.load_cell(args.chk); cell.build(False, False); cell.max_memory = args.max_mem
d = lib.chkfile.load(args.chk, 'scf')
mo_coeff = np.asarray(d['mo_coeff']); mo_e = np.asarray(d['mo_energy']); e_hf = float(d['e_tot'])
if mo_coeff.ndim == 3: mo_coeff, mo_e = mo_coeff[0], mo_e[0]
mo_coeff = mo_coeff.real; mo_e = mo_e.real
S = cell.pbc_intor('int1e_ovlp').real
F_ao = S @ mo_coeff @ np.diag(mo_e) @ mo_coeff.T @ S

no = np.load(args.no)
C_act, e_act = no['C_act_occ'], no['e_act_occ']; nact = C_act.shape[1]
C_no, occ_no = no['C_no_vir'], no['occ_no']
n_max = int((occ_no > min(thr)).sum())
C_max = C_no[:, :n_max]
print(f"[load] nact_occ={nact} n_vir_max={n_max} (thr={min(thr):.0e})  E_HF={e_hf:.8f}")

mf = pbcscf.RHF(cell).density_fit(); mf.max_memory = args.max_mem
mf.with_df._cderi = args.cderi
eri_max = np.asarray(mf.with_df.ao2mo([C_act, C_max, C_act, C_max], compact=False)).real
eri_max = eri_max.reshape(nact, n_max, nact, n_max)          # NO 기저 [i,a,j,b]
print(f"[ERI] done ({(time.time()-t0)/60:.1f} min)")

def mp2_in(n):
    Csub = C_max[:, :n]
    f = Csub.T @ F_ao @ Csub
    e_v, U = np.linalg.eigh(f)                               # 재정규화
    eri = np.einsum('iajb,ac,bd->icjd', eri_max[:, :n, :, :n], U, U, optimize=True)
    den = (e_act[:,None,None,None] + e_act[None,None,:,None]
           - e_v[None,:,None,None] - e_v[None,None,None,:])
    T = eri / den
    return float(np.einsum('iajb,iajb->', T, 2*eri - eri.transpose(0,3,2,1)))

rows = []
print(" threshold  n_vir  n_tot   E_MP2(active)     dE vs full-vir (kJ/mol)")
E_full = float(no['E_mp2_act'])
for t in thr:
    n = int((occ_no > t).sum()); E = mp2_in(n)
    rows.append((t, n, n+nact, E))
    print(f" {t:8.0e}  {n:5d}  {n+nact:5d}   {E:14.8f}   {(E-E_full)*2625.5:10.2f}")
print(f" full-vir   {len(occ_no):5d}  {len(occ_no)+nact:5d}   {E_full:14.8f}         0.00")
np.savez(args.out, thr=np.array(thr), n_vir=np.array([r[1] for r in rows]),
         n_tot=np.array([r[2] for r in rows]), E_mp2=np.array([r[3] for r in rows]),
         E_full=E_full, E_hf=e_hf, nact=nact)
print(f"[save] {args.out}  ({(time.time()-t0)/60:.1f} min)")