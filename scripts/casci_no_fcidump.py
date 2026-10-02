#!/usr/bin/env python
"""FCIDUMP를 CASCI 자연 오비탈(NO) 기저로 회전한다.

사용법:
  python scripts/casci_no_fcidump.py FCIDUMP [FCIDUMP ...] [--suffix _no]

입력 하나마다:
  1) 활성 공간 FCI(=CASCI) -> 스핀합 1-RDM -> 대각화 = NO (점유수 내림차순)
  2) h1, (pq|rs)를 NO 기저로 회전 (ECORE 그대로) -> <입력><suffix> 저장
  3) 검증: 저장한 파일을 다시 읽어 FCI가 회전 전과 같은지, 1-RDM이 대각인지
  4) 지표: NO 점유수, 참조 행렬식(앞쪽 오비탈부터 채운 것)의 에너지와 가중치 c0^2
     (회전 전 = HF 행렬식, 회전 후 = NO 행렬식)
  NO 점유수와 회전 행렬은 <출력>.npz 에 저장 (그림용).
"""
import argparse
import sys

import numpy as np
from pyscf import ao2mo, fci
from pyscf.tools import fcidump

HA2KJ = 2625.499639


def load(path):
    d = fcidump.read(path, verbose=False)
    norb = int(d["NORB"])
    ne = int(d["NELEC"])
    ms2 = int(d.get("MS2", 0))
    nelec = ((ne + ms2) // 2, (ne - ms2) // 2)
    h1 = np.asarray(d["H1"], dtype=float)
    eri = ao2mo.restore(1, np.asarray(d["H2"], dtype=float), norb)
    return h1, eri, norb, nelec, ms2, float(d["ECORE"])


def solve(h1, eri, norb, nelec, ecore):
    s = fci.direct_spin1.FCI()
    s.verbose = 0
    s.conv_tol = 1e-11
    s.max_cycle = 300
    e, c = s.kernel(h1, eri, norb, nelec, ecore=ecore)
    if not s.converged:
        print("  [경고] FCI 미수렴")
    dm1 = s.make_rdm1(c, norb, nelec)
    ss = fci.spin_op.spin_square(c, norb, nelec)[0]
    # 참조 행렬식: 주소 (0,0) = alpha, beta 모두 앞쪽 오비탈부터 채운 것
    c0 = np.zeros_like(c)
    c0[0, 0] = 1.0
    e_ref = s.energy(h1, eri, c0, norb, nelec) + ecore
    return e, dm1, ss, e_ref, float(c[0, 0] ** 2)


def run(path, suffix):
    out = path + suffix
    print("=" * 72)
    print("입력:", path)
    h1, eri, norb, nelec, ms2, ecore = load(path)
    ndocc = min(nelec)
    print("  NORB=%d  NELEC=%d  MS2=%d  (%d큐비트)" % (norb, sum(nelec), ms2, 2 * norb))

    e, dm1, ss, e_ref, w0 = solve(h1, eri, norb, nelec, ecore)
    print("  E(CASCI)          = %.8f Ha   <S^2> = %.4f" % (e, ss))
    print("  E(참조 행렬식)    = %.8f Ha   참조-CASCI = %8.2f kJ/mol   c0^2 = %.4f"
          % (e_ref, (e_ref - e) * HA2KJ, w0))

    occ, U = np.linalg.eigh(dm1)
    idx = np.argsort(-occ)
    occ, U = occ[idx], U[:, idx]
    for k in range(norb):  # 부호 고정 (재현성)
        if U[np.argmax(np.abs(U[:, k])), k] < 0:
            U[:, k] *= -1.0

    h1n = U.T @ h1 @ U
    erin = ao2mo.incore.full(ao2mo.restore(8, eri, norb), U)
    fcidump.from_integrals(out, h1n, erin, norb, sum(nelec), nuc=ecore, ms=ms2)

    # ---- 검증: 쓴 파일을 다시 읽어서 ----
    h1r, erir, norb_r, nelec_r, _, ecore_r = load(out)
    e2, dm2, ss2, e_ref2, w02 = solve(h1r, erir, norb_r, nelec_r, ecore_r)
    de = e2 - e
    offd = np.abs(dm2 - np.diag(np.diag(dm2))).max()
    dd = np.abs(np.diag(dm2) - occ).max()

    print("  --- NO 기저 (%s) ---" % out)
    print("  E(CASCI)          = %.8f Ha   회전 전후 차이 = %.1e Ha" % (e2, de))
    print("  E(참조 행렬식)    = %.8f Ha   참조-CASCI = %8.2f kJ/mol   c0^2 = %.4f"
          % (e_ref2, (e_ref2 - e2) * HA2KJ, w02))
    print("  1-RDM 비대각 최대 = %.1e   대각-점유수 최대 차 = %.1e" % (offd, dd))
    print("  NO 점유수:")
    print("    강점유(앞 %d개): " % ndocc + " ".join("%.4f" % x for x in occ[:ndocc]))
    print("    약점유(나머지) : " + " ".join("%.4f" % x for x in occ[ndocc:]))
    print("    약점유 합 = %.4f e    0.02<n<1.98 인 NO = %d개    0.10<n<1.90 인 NO = %d개"
          % (occ[ndocc:].sum(),
             int(((occ > 0.02) & (occ < 1.98)).sum()),
             int(((occ > 0.10) & (occ < 1.90)).sum())))

    ok = abs(de) < 1e-8 and offd < 1e-5 and dd < 1e-5
    print("  검증:", "PASS" if ok else "FAIL")
    np.savez(out + ".npz", occ=occ, U=U, e_casci=e, e_ref_in=e_ref, e_ref_no=e_ref2,
             c0sq_in=w0, c0sq_no=w02)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fcidump", nargs="+")
    ap.add_argument("--suffix", default="_no")
    a = ap.parse_args()
    ok = all([run(p, a.suffix) for p in a.fcidump])
    print("=" * 72)
    print("전체:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()