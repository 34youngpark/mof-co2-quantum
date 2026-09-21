#!/usr/bin/env python3
"""
5.2 (4): 주기 HF chk -> 점유 오비탈 국소화(Pipek-Mezey, Gamma) -> 흡착 부위 오비탈 선택 -> 재정규화
Rocca/Schaefer 워크플로우의 VASP 단계 3~7 (INCAR.wan / INCAR.srt / INCAR.HF.diag.recan)에 대응.

사용:
    python localize_occ.py --chk hf_mofco2.chk --frag-mode auto --out as_occ_mofco2.npz
    python localize_occ.py --chk hf_mof.chk    --frag-atoms 12,13,45,46,101 --out as_occ_mof.npz

frag-mode auto: CO2 원자(C+O2, 원소와 연결성으로 탐지) 및 아민 N 주변 cutoff(기본 3.0 A) 안의
원자를 fragment로 자동 선정. 화학흡착 계(carbamate)에서는 C-N 결합이 이미 생겼으므로
CO2 탄소에서 cutoff 내 N까지 자동 포함됨. 반드시 출력된 fragment 원자 목록을 눈으로 확인할 것.

출력 npz:
    C_loc_all   : 국소화된 전체 점유 오비탈 (AO x Nocc)
    sel_idx     : 선택된 국소 오비탈 인덱스
    C_act_occ   : 선택 + 재정규화된 활성 점유 오비탈 (AO x n_act_occ)
    e_act_occ   : 재정규화 후 오비탈 에너지 (Fock 고유값)
    C_core_occ  : 나머지(frozen) 점유 오비탈, 재정규화됨
    frag_atoms  : fragment 원자 인덱스
    pop_table   : (Nocc x 2) [fragment 점유율, 전체 노름]
다음 단계(5.2 (5) MP2 NO)는 C_act_occ만 점유로 넣고 나머지를 frozen 처리해서
비점유 블록 밀도행렬을 만들면 됨.
"""
import argparse, sys
import numpy as np
from pyscf import lib
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.lib import chkfile as pbcchk
from pyscf import lo

COV_R = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'Mg':1.41}  # covalent radii (A)

def load_mf(chkfile_path):
    cell = pbcchk.load_cell(chkfile_path)
    cell.build(False, False)
    scf_dat = lib.chkfile.load(chkfile_path, 'scf')
    mo_coeff = np.asarray(scf_dat['mo_coeff'])
    mo_occ   = np.asarray(scf_dat['mo_occ'])
    mo_e     = np.asarray(scf_dat['mo_energy'])
    # KRHF(gamma) 저장 형식이면 (1, nao, nmo) -> squeeze
    if mo_coeff.ndim == 3:
        assert mo_coeff.shape[0] == 1, "Gamma-only chk만 지원 (k>1이면 k-mesh용 확장 필요)"
        mo_coeff, mo_occ, mo_e = mo_coeff[0], mo_occ[0], mo_e[0]
    if np.iscomplexobj(mo_coeff):
        if np.abs(mo_coeff.imag).max() < 1e-8:
            mo_coeff = mo_coeff.real.copy()
        else:
            raise RuntimeError("MO에 유의미한 허수부가 있음 - Gamma점 실수 오비탈이 아님")
    return cell, mo_coeff, mo_occ, mo_e

def auto_fragment(cell, cutoff=3.0):
    """CO2(C=O 2개인 탄소) + 아민 N + 그 주변 cutoff 내 원자 자동 선정"""
    coords = cell.atom_coords() * lib.param.BOHR  # A
    syms = [cell.atom_symbol(i) for i in range(cell.natm)]
    # 최소영상 거리
    a = cell.lattice_vectors() * lib.param.BOHR
    def mind(i, j):
        d = coords[j] - coords[i]
        best = np.inf
        for n1 in (-1,0,1):
            for n2 in (-1,0,1):
                for n3 in (-1,0,1):
                    v = d + n1*a[0] + n2*a[1] + n3*a[2]
                    best = min(best, np.linalg.norm(v))
        return best
    # CO2 탄소 탐지: O 이웃(<=1.45A)이 정확히 2개이고 다른 C 이웃이 없는 C
    seeds = []
    for i, s in enumerate(syms):
        if s == 'C':
            o_nb = [j for j in range(cell.natm) if syms[j]=='O' and i!=j and mind(i,j)<1.45]
            c_nb = [j for j in range(cell.natm) if syms[j]=='C' and i!=j and mind(i,j)<1.7]
            if len(o_nb)==2 and len(c_nb)==0:
                seeds += [i]+o_nb
    # 모든 N (아민; ampd에는 N 2개/아민)도 후보 seed
    seeds += [i for i,s in enumerate(syms) if s=='N']
    seeds = sorted(set(seeds))
    frag = set(seeds)
    for i in seeds:
        for j in range(cell.natm):
            if mind(i,j) < cutoff:
                frag.add(j)
    return sorted(frag), seeds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chk', required=True)
    ap.add_argument('--frag-mode', choices=['auto','manual'], default='auto')
    ap.add_argument('--frag-atoms', default='', help='manual: 0-기준 원자 인덱스, 콤마구분')
    ap.add_argument('--cutoff', type=float, default=3.0, help='auto seed 주변 포함 반경(A)')
    ap.add_argument('--pop-thresh', type=float, default=0.4,
                    help='fragment Mulliken 점유율이 이 값 이상인 국소 오비탈 선택')
    ap.add_argument('--out', default='as_occ.npz')
    ap.add_argument('--cube-top', type=int, default=0, help='상위 n개 국소 오비탈 cube 출력(검증용)')
    args = ap.parse_args()

    cell, mo_coeff, mo_occ, mo_e = load_mf(args.chk)
    nocc = int(round(mo_occ.sum()/2))
    print(f"[load] natm={cell.natm} nao={mo_coeff.shape[0]} nocc={nocc}")

    if args.frag_mode == 'auto':
        frag_atoms, seeds = auto_fragment(cell, args.cutoff)
        print(f"[frag] seeds(CO2/N)={seeds}")
    else:
        frag_atoms = [int(x) for x in args.frag_atoms.split(',') if x.strip()!='']
    print(f"[frag] fragment atoms ({len(frag_atoms)}):")
    for i in frag_atoms:
        print(f"    {i:4d} {cell.atom_symbol(i):>2s}  {np.round(cell.atom_coord(i)*lib.param.BOHR,3)}")

    # --- Pipek-Mezey 국소화 (Gamma, 실수 오비탈이므로 분자용 루틴 사용 가능) ---
    C_occ = mo_coeff[:, :nocc]
    S = cell.pbc_intor('int1e_ovlp')
    if np.iscomplexobj(S): S = S.real
    pm = lo.PipekMezey(cell, C_occ)
    pm.pop_method = 'mulliken'
    pm.conv_tol = 1e-7
    C_loc = pm.kernel()
    print("[PM] localization done.")

    # --- 국소 오비탈별 fragment Mulliken 점유율 ---
    ao_slices = cell.aoslice_by_atom()
    frag_ao = np.concatenate([np.arange(ao_slices[a,2], ao_slices[a,3]) for a in frag_atoms])
    SC = S @ C_loc
    pop = np.einsum('mi,mi->mi', C_loc, SC)  # AO x Norb, Mulliken 기여
    pop_frag = pop[frag_ao].sum(axis=0)
    pop_tot  = pop.sum(axis=0)  # ~1
    frac = pop_frag / pop_tot

    order = np.argsort(-frac)
    print("\n[rank] localized occ orbitals by fragment population")
    print(" idx   frag_pop   main atoms (pop>=0.15)")
    atom_pop = np.zeros((cell.natm, C_loc.shape[1]))
    for a in range(cell.natm):
        sl = np.arange(ao_slices[a,2], ao_slices[a,3])
        atom_pop[a] = pop[sl].sum(axis=0)
    for i in order[:min(60, len(order))]:
        mains = [f"{cell.atom_symbol(a)}{a}:{atom_pop[a,i]:.2f}"
                 for a in np.argsort(-atom_pop[:,i])[:4] if atom_pop[a,i] >= 0.15]
        mark = '*' if frac[i] >= args.pop_thresh else ' '
        print(f" {mark}{i:4d}   {frac[i]:7.3f}   {' '.join(mains)}")

    sel = np.where(frac >= args.pop_thresh)[0]
    print(f"\n[select] {len(sel)} orbitals with frag_pop >= {args.pop_thresh}")

    # --- 재정규화: 선택/비선택 부분공간 각각에서 Fock 대각화 ---
    F_ao = S @ mo_coeff @ np.diag(mo_e) @ mo_coeff.T @ S
    def recan(Csub):
        f = Csub.T @ F_ao @ Csub
        e, u = np.linalg.eigh(f)
        return Csub @ u, e
    C_act, e_act = recan(C_loc[:, sel])
    rest = np.setdiff1d(np.arange(nocc), sel)
    C_core, e_core = recan(C_loc[:, rest]) if len(rest) else (np.zeros((S.shape[0],0)), np.array([]))
    print(f"[recan] active occ energies (Ha): {np.round(e_act,4)}")

    # --- 검증: span 보존 (선택+나머지 = 원래 점유 공간) ---
    P0 = C_occ @ C_occ.T
    P1 = C_act @ C_act.T + (C_core @ C_core.T if len(rest) else 0)
    err = np.abs(S @ (P0-P1) @ S).max()
    print(f"[check] occupied-space projector diff = {err:.2e} (1e-8 이하이어야 함)")

    np.savez(args.out, C_loc_all=C_loc, sel_idx=sel, C_act_occ=C_act, e_act_occ=e_act,
             C_core_occ=C_core, e_core_occ=e_core, frag_atoms=np.array(frag_atoms),
             pop_table=np.stack([frac, pop_tot], axis=1))
    print(f"[save] {args.out}")

    if args.cube_top > 0:
        from pyscf.tools import cubegen
        for k, i in enumerate(order[:args.cube_top]):
            cubegen.orbital(cell, f"locorb_{i}.cube", C_loc[:, i])
            print(f"[cube] locorb_{i}.cube (frag_pop={frac[i]:.3f})")

if __name__ == '__main__':
    main()