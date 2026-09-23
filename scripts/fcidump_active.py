#!/usr/bin/env python3
"""
5.2 (7) v2: cderi 2회 읽기로 여러 활성 공간 크기(28q, 20q ...)를 한 번에 생성.
사용: python -u scripts/fcidump_active.py --chk ... --no as_no_b.npz --cderi ... --name b --spaces 7,7 5,5
출력: quantum/reference/FCIDUMP.<name>.<nq>q, as_final_<name>_<nq>q.npz
원리: 활성 점유 16 + NO 상위 80 = 96오비탈 4-index 적분을 한 번 뽑고(읽기 1), 전체 밀도의 J/K를
      한 번 계산(읽기 2)한 뒤, 활성 공간 h1/E_core는 그 안의 선형대수로 구성.
      E_core는 E_HF(cas 행렬식) == E_HF(exxdiv=None) 이 되도록 정의 (v1의 [check]=0 조건과 동일).
"""
import argparse, time, numpy as np
from pyscf import lib, fci, ao2mo
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.lib import chkfile as pbcchk
from pyscf.tools import fcidump

ap = argparse.ArgumentParser()
ap.add_argument('--chk', required=True); ap.add_argument('--no', required=True)
ap.add_argument('--cderi', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--spaces', nargs='+', default=['7,7', '5,5'], help='nocc,nvir 목록')
ap.add_argument('--nvir-pool', type=int, default=80)
ap.add_argument('--max-mem', type=int, default=20000)
ap.add_argument('--no-casci', action='store_true', help='큰 공간: CASCI 생략 (CCSD(T)용 FCIDUMP만)')
a = ap.parse_args(); t0 = time.time(); KJ = 2625.5
spaces = [tuple(int(v) for v in s.split(',')) for s in a.spaces]

# --- HF, ④⑤ 결과 로드 ---
cell = pbcchk.load_cell(a.chk); cell.build(False, False); cell.max_memory = a.max_mem
d = lib.chkfile.load(a.chk, 'scf')
mo, moe, moo = np.asarray(d['mo_coeff']), np.asarray(d['mo_energy']), np.asarray(d['mo_occ'])
if mo.ndim == 3: mo, moe, moo = mo[0], moe[0], moo[0]
mo, moe = mo.real, moe.real
nocc = int(round(moo.sum()/2)); nao = mo.shape[0]
S = cell.pbc_intor('int1e_ovlp').real
F = S @ mo @ np.diag(moe) @ mo.T @ S
no = np.load(a.no)
C_act, e_act = no['C_act_occ'], no['e_act_occ']
C_no, occ_no = no['C_no_vir'], no['occ_no']
nact = C_act.shape[1]; nvp = min(a.nvir_pool, len(occ_no))
print(f"[{a.name}] nao={nao} nocc={nocc} act_occ={nact} vir_pool={nvp}", flush=True)

mf = pbcscf.RHF(cell).density_fit(); mf.max_memory = a.max_mem
mf.with_df._cderi = a.cderi; mf.exxdiv = None

# --- 읽기 1: pool 4-index 적분 ---
C_pool = np.hstack([C_act, C_no[:, :nvp]]); npool = C_pool.shape[1]
eri_pool = ao2mo.restore(1, np.asarray(mf.with_df.ao2mo(C_pool, compact=False)).real.reshape(npool, npool, npool, npool), npool)
F_pool = C_pool.T @ F @ C_pool
print(f"[ERI] pool {npool}^4 = {eri_pool.nbytes/1e9:.2f} GB  ({(time.time()-t0)/60:.1f} min)", flush=True)

# --- 점유 결손 (pool 안에서 MP2) ---
o, v = slice(0, nact), slice(nact, npool)
e_v, Uv = np.linalg.eigh(F_pool[v, v])
eri_ov = np.einsum('iajb,ac,bd->icjd', eri_pool[o, v, o, v], Uv, Uv, optimize=True)
den = e_act[:,None,None,None] + e_act[None,None,:,None] - e_v[None,:,None,None] - e_v[None,None,None,:]
T = eri_ov / den
doo = -(2*np.einsum('iajb,kajb->ik', T, T) - np.einsum('iajb,kbja->ik', T, T))
dep, Uo = np.linalg.eigh(doo); order = np.argsort(dep); dep, Uo = dep[order], Uo[:, order]
print(f"[occ] depletion (all {nact}): {np.round(-dep, 4)}", flush=True)
del eri_ov, T

# --- 읽기 2: 전체 밀도의 J/K (exxdiv=None) ---
hcore = mf.get_hcore(); enuc = mf.energy_nuc()
dm_full = 2*mo[:, :nocc] @ mo[:, :nocc].T
vj, vk = mf.get_jk(cell, dm_full); veff_full = vj - 0.5*vk
e_hf_noexx = enuc + np.einsum('ij,ji', dm_full, hcore) + 0.5*np.einsum('ij,ji', dm_full, veff_full)
h_full_pool = C_pool.T @ (hcore + veff_full) @ C_pool
print(f"[JK] done ({(time.time()-t0)/60:.1f} min)  E_HF(exxdiv=None)={e_hf_noexx:.8f}  "
      f"E_HF(chk,ewald)={float(d['e_tot']):.8f}  madelung shift={float(d['e_tot'])-e_hf_noexx:.6f} Ha", flush=True)

# --- 활성 공간별 h1 / E_core / CASCI / FCIDUMP ---
def recan(X):
    e, U = np.linalg.eigh(X.T @ F_pool @ X); return X @ U, e

for nocc_sel, nvir_sel in spaces:
    nocc_sel = min(nocc_sel, nact)
    Xo = np.zeros((npool, nocc_sel)); Xo[o, :] = Uo[:, :nocc_sel]
    Xv = np.zeros((npool, nvir_sel)); Xv[nact:nact+nvir_sel, :] = np.eye(nvir_sel)
    Xo, e_o = recan(Xo); Xv, e_vs = recan(Xv)
    X = np.hstack([Xo, Xv]); ncas = X.shape[1]; nelecas = 2*nocc_sel; nq = 2*ncas
    C_cas = C_pool @ X
    # h1 = <p|hcore + veff_full|q> − J/K(활성 점유 밀도)  (pool 적분으로 계산)
    Dcas = 2*Xo @ Xo.T
    veff_cas_pool = np.einsum('pqij,ij->pq', eri_pool, Dcas) - 0.5*np.einsum('piqj,ij->pq', eri_pool, Dcas)
    h1 = X.T @ (h_full_pool - veff_cas_pool) @ X
    eri_cas = np.einsum('pqrs,pa,qb,rc,sd->abcd', eri_pool, X, X, X, X, optimize=True)
    # E_core: 활성 HF 행렬식 에너지가 전체 HF(exxdiv=None)와 같도록
    dmc = np.zeros((ncas, ncas)); dmc[:nocc_sel, :nocc_sel] = 2*np.eye(nocc_sel)
    e_cas_hf = (np.einsum('ij,ij', h1, dmc) + 0.5*np.einsum('ijkl,ij,kl', eri_cas, dmc, dmc)
                - 0.25*np.einsum('ijkl,il,kj', eri_cas, dmc, dmc))
    ecore = e_hf_noexx - e_cas_hf
    if a.no_casci:
        e_casci = float('nan')
    else:
        cis = fci.direct_spin1.FCI(); cis.max_cycle = 200; cis.conv_tol = 1e-9
        e_casci, ci = cis.kernel(h1, eri_cas, ncas, nelecas, ecore=ecore)
    path = f"quantum/reference/FCIDUMP.{a.name}.{nq}q"
    fcidump.from_integrals(path, h1, eri_cas, ncas, nelecas, nuc=ecore, ms=0)
    np.savez(f"as_final_{a.name}_{nq}q.npz", C_cas=C_cas, ncas=ncas, nelecas=nelecas,
             e_casci=e_casci, e_hf_cas=e_hf_noexx, e_hf_chk=float(d['e_tot']), ecore=ecore,
             e_occ=e_o, e_vir=e_vs, dep=dep[:nocc_sel], occ_no_sel=occ_no[:nvir_sel])
    print(f"[{nq}q] ({nelecas}e,{ncas}o)  occ e={np.round(e_o,3)}  vir e={np.round(e_vs,3)}", flush=True)
    print(f"      E_CASCI={e_casci:.8f}  E_HF(cas)={e_hf_noexx:.8f}  corr_in_cas={(e_casci-e_hf_noexx)*KJ:.2f} kJ/mol"
          f"  ecore={ecore:.6f}  → {path}", flush=True)
print(f"[done] {(time.time()-t0)/60:.1f} min")