"""
주기 HF + MP2 파이프라인 확인용: 15 Å 박스 속 CO2.
사용: OMP_NUM_THREADS=8 python scripts/pbc_hf_test.py
"""
import time
from ase.io import read
from pyscf.pbc import gto, scf, mp

atoms = read("structures/02_relaxed/d_CO2_box_relaxed.cif")
cell = gto.Cell()
cell.atom = [(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)]
cell.a = atoms.cell[:]                 # Å
cell.unit = "Angstrom"
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pbe"
cell.exp_to_discard = 0.1              # 주기계에서 너무 퍼진 함수 제거
cell.verbose = 4
cell.max_memory = 20000                # MB
cell.build()
print("기저함수 수:", cell.nao, " 전자 수:", cell.nelectron)

t = time.time()
mf = scf.RHF(cell).density_fit()       # Γ점, Gaussian density fitting
mf.conv_tol = 1e-8
e_hf = mf.kernel()
print("HF 에너지  = %.8f Ha  (%.1f s)" % (e_hf, time.time() - t))

t = time.time()
pt = mp.RMP2(mf)
e_corr, _ = pt.kernel()
print("MP2 상관   = %.8f Ha  (%.1f s)" % (e_corr, time.time() - t))
print("MP2 총에너지 = %.8f Ha" % (e_hf + e_corr))