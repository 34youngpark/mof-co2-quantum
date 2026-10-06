#!/usr/bin/env python3
"""활성 공간 FCIDUMP → CASCI 1-RDM 자연 오비탈 기저로 회전한 FCIDUMP (<src>_no)"""
import sys, numpy as np
from pyscf import ao2mo, fci
from pyscf.tools import fcidump
KJ = 2625.5
src = sys.argv[1]; d = fcidump.read(src, verbose=False)
n, nelec, ecore = d["NORB"], d["NELEC"], d["ECORE"]
h1 = d["H1"]; eri = ao2mo.restore(1, d["H2"], n)
cis = fci.direct_spin1.FCI(); cis.conv_tol = 1e-10
e0, ci = cis.kernel(h1, eri, n, nelec, ecore=ecore)
dm1 = cis.make_rdm1(ci, n, nelec)
occ, U = np.linalg.eigh(dm1); occ, U = occ[::-1], U[:, ::-1]
h1n = U.T @ h1 @ U
erin = np.einsum('pqrs,pa,qb,rc,sd->abcd', eri, U, U, U, U, optimize=True)
e1, _ = cis.kernel(h1n, erin, n, nelec, ecore=ecore)
no = nelec // 2; dm = np.zeros((n, n)); dm[:no, :no] = 2*np.eye(no)
e_det = ecore + np.einsum('ij,ij', h1n, dm) + 0.5*np.einsum('ijkl,ij,kl', erin, dm, dm) - 0.25*np.einsum('ijkl,il,kj', erin, dm, dm)
dst = src + "_no"; fcidump.from_integrals(dst, h1n, erin, n, nelec, nuc=ecore, ms=0)
print(f"NO occupations: {np.round(occ, 4)}")
print(f"CASCI before {e0:.8f}  after {e1:.8f}  diff {(e1-e0)*KJ:.2e} kJ/mol (0이어야 함)")
print(f"reference det in NO basis: ΔCASCI={(e_det-e0)*KJ:.2f} kJ/mol  → {dst}")
