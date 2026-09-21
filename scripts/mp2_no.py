#!/usr/bin/env python3
"""
5.2 (5): MP2 natural orbital로 비점유 공간 압축
입력: HF chk + as_occ_*.npz (④ 출력)
출력: as_no_*.npz (C_no_vir, occ_no, threshold별 개수, E_MP2_act)

사용:
  python mp2_no.py --chk calc/pbc_hf/b_MOF_CO2_bound_gth-dzvp.chk --occ as_occ_b.npz \
                   --cderi scratch/cderi_b.h5 --out as_no_b.npz --max-mem 40000
--cderi 파일이 있으면 읽고, 없으면 만들어서 그 경로에 저장 (⑥⑦에서 재사용)
"""
import argparse, os, time
import numpy as np
from pyscf import lib
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.lib import chkfile as pbcchk

ap = argparse.ArgumentParser()
ap.add_argument('--chk', required=True)
ap.add_argument('--occ', required=True, help='④ 출력 npz')
ap.add_argument('--cderi', required=True, help='DF 텐서 저장/재사용 경로 (.h5)')
ap.add_argument('--out', default='as_no.npz')
ap.add_argument('--max-mem', type=int, default=40000, help='MB')
ap.add_argument('--thresholds', default='1e-2,1e-3,1e-4,1e-5,1e-6')
args = ap.parse_args()
t0 = time.time()

# --- HF 로드 ---
cell = pbcchk.load_cell(args.chk); cell.build(False, False)
cell.max_memory = args.max_mem
d = lib.chkfile.load(args.chk, 'scf')
mo_coeff = np.asarray(d['mo_coeff']); mo_occ = np.asarray(d['mo_occ']); mo_e = np.asarray(d['mo_energy'])
if mo_coeff.ndim == 3: mo_coeff, mo_occ, mo_e = mo_coeff[0], mo_occ[0], mo_e[0]
mo_coeff = mo_coeff.real; mo_e = mo_e.real
nocc = int(round(mo_occ.sum()/2)); nao = mo_coeff.shape[0]
C_vir = mo_coeff[:, nocc:]; e_vir = mo_e[nocc:]; nvir = C_vir.shape[1]

# --- ④ 결과 로드 ---
occ = np.load(args.occ)
C_act = occ['C_act_occ']; e_act = occ['e_act_occ']; nact = C_act.shape[1]
print(f"[load] nao={nao} nocc={nocc} nact_occ={nact} nvir={nvir}")

# --- density fitting 텐서 ---
mf = pbcscf.RHF(cell).density_fit()
mf.max_memory = args.max_mem
if os.path.exists(args.cderi):
    mf.with_df._cderi = args.cderi
    print(f"[DF] reuse {args.cderi}")
else:
    mf.with_df._cderi_to_save = args.cderi
    print(f"[DF] building -> {args.cderi} (시간 걸림)")
    mf.with_df.build()
print(f"[DF] ready  ({(time.time()-t0)/60:.1f} min)")

# --- (ia|jb), i,j: 활성 점유 / a,b: 전체 비점유 ---
eri = mf.with_df.ao2mo([C_act, C_vir, C_act, C_vir], compact=False)
eri = np.asarray(eri).real.reshape(nact, nvir, nact, nvir)   # [i,a,j,b]
print(f"[ERI] (ia|jb) {eri.nbytes/1e9:.1f} GB  ({(time.time()-t0)/60:.1f} min)")

# --- MP2 진폭과 활성 점유 MP2 에너지 ---
denom = (e_act[:,None,None,None] + e_act[None,None,:,None]
         - e_vir[None,:,None,None] - e_vir[None,None,None,:])
T = eri / denom                                               # t[i,a,j,b]
E_mp2 = np.einsum('iajb,iajb->', T, 2*eri - eri.transpose(0,3,2,1))
print(f"[MP2] E_corr(active occ, full vir) = {E_mp2:.8f} Ha")
del eri, denom

# --- 비점유 블록 1-RDM (spin-summed) ---
dvv = 2*np.einsum('icja,icjb->ab', T, T) - np.einsum('icja,ibjc->ab', T, T)
D = dvv + dvv.T
del T, dvv
occ_no, U = np.linalg.eigh(D)
occ_no, U = occ_no[::-1], U[:, ::-1]                          # 내림차순
C_no_vir = C_vir @ U
print(f"[NO] sum occ = {occ_no.sum():.5f}, max = {occ_no[0]:.5f}")

# --- threshold별 개수 (⑥ 수렴 표 재료) ---
thr = [float(x) for x in args.thresholds.split(',')]
counts = {t: int((occ_no > t).sum()) for t in thr}
print("[thr]  threshold  n_vir_kept  (+ n_act_occ = total active)")
for t in thr:
    print(f"       {t:8.0e}   {counts[t]:5d}        {counts[t]+nact:5d}")

np.savez(args.out, C_act_occ=C_act, e_act_occ=e_act, C_core_occ=occ['C_core_occ'],
         C_no_vir=C_no_vir, occ_no=occ_no, E_mp2_act=E_mp2, nact_occ=nact,
         thr=np.array(thr), counts=np.array([counts[t] for t in thr]))
print(f"[save] {args.out}  total {(time.time()-t0)/60:.1f} min")