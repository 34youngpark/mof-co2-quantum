#!/usr/bin/env python3
import sys, numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump
src = sys.argv[1]; d = fcidump.read(src, verbose=False)
n, nelec = d["NORB"], d["NELEC"]; nocc = nelec // 2
h1 = d["H1"]; eri = ao2mo.restore(1, d["H2"], n)
order = []
for k in range(n - nocc):
    if nocc - 1 - k >= 0: order.append(nocc - 1 - k)
    order.append(nocc + k)
order += [i for i in range(n) if i not in order]
order = np.array(order)
h1n = h1[np.ix_(order, order)]
erin = eri[np.ix_(order, order, order, order)]
occ_new = [int(np.where(order == i)[0][0]) for i in range(nocc)]
dst = src + "_il"
fcidump.from_integrals(dst, h1n, erin, n, nelec, nuc=d["ECORE"], ms=0)
print(f"{dst}\norder(old idx)={order.tolist()}\nocc(new idx)={sorted(occ_new)}")
