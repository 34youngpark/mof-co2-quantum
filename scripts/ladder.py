#!/usr/bin/env python3
"""5.4 (2): 활성 공간 내 방법 사다리 HF/MP2/CCSD/CCSD(T)/CASCI + 방법별 ΔE2. FCIDUMP 기반, 초 단위."""
import sys, numpy as np
from pyscf import gto, scf, mp, cc
from pyscf.tools import fcidump
KJ = 2625.5
rows = {}
for nq in ("20q", "28q"):
    for s in "abcd":
        path = f"quantum/reference/FCIDUMP.{s}.{nq}"
        try: z = np.load(f"as_final_{s}_{nq}.npz")
        except FileNotFoundError: continue
        mf = fcidump.to_scf(path, molpro_orbsym=False)
        n = mf.mol.nao; nocc = mf.mol.nelectron // 2
        dm0 = np.zeros((n, n)); dm0[:nocc, :nocc] = 2*np.eye(nocc)      # 활성 HF 행렬식에서 출발
        mf.kernel(dm0=dm0)
        e_hf = mf.e_tot
        e_mp2 = mp.MP2(mf).run().e_tot
        mycc = cc.CCSD(mf).run(); e_ccsd = mycc.e_tot; e_ccsdt = e_ccsd + mycc.ccsd_t()
        e_cas = float(z["e_casci"])
        rows[(nq, s)] = dict(HF=e_hf, MP2=e_mp2, CCSD=e_ccsd, CCSDT=e_ccsdt, CASCI=e_cas)
        print(f"[{nq} {s}] HF={e_hf:.6f} (cas HF {float(z['e_hf_cas']):.6f})  MP2={e_mp2:.6f}  CCSD={e_ccsd:.6f}  "
              f"CCSD(T)={e_ccsdt:.6f}  CASCI={e_cas:.6f}  | CCSD−CASCI={(e_ccsd-e_cas)*KJ:6.2f}  (T)−CASCI={(e_ccsdt-e_cas)*KJ:6.2f} kJ/mol")
print("\nΔE2 = E(b) − E(c) by method (kJ/mol):")
for nq in ("20q", "28q"):
    if (nq, "b") in rows and (nq, "c") in rows:
        b, c = rows[(nq, "b")], rows[(nq, "c")]
        print(f"  {nq}: " + "  ".join(f"{m}={ (b[m]-c[m])*KJ:7.2f}" for m in ("HF", "MP2", "CCSD", "CCSDT", "CASCI")))