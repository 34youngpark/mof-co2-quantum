"""
주기 HF (Γ점, GDF) — 216/219원자 MOF 계.
사용: OMP_NUM_THREADS=24 python scripts/pbc_hf_mof.py <name> <basis>
  name : a_MOF | b_MOF_CO2_bound | c_MOF_CO2_free | d_CO2_box
  basis: gth-szv | gth-dzvp
결과: calc/pbc_hf/<name>_<basis>.chk (오비탈 저장, 뒤 단계에서 재사용), 같은 이름 .log
"""
import sys, os, time
from ase.io import read
from pyscf.pbc import gto, scf
from pyscf import lib

name, basis = sys.argv[1], sys.argv[2]
os.makedirs("calc/pbc_hf", exist_ok=True)
tag = f"calc/pbc_hf/{name}_{basis}"

atoms = read(f"structures/02_relaxed/{name}_relaxed.cif")
cell = gto.Cell()
cell.atom = [(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)]
cell.a = atoms.cell[:]
cell.unit = "Angstrom"
cell.basis = basis
cell.pseudo = "gth-pbe"
cell.exp_to_discard = 0.1
cell.verbose = 4
cell.output = tag + ".log"
cell.max_memory = 100000               # MB (워크스테이션 125 GB 중 100 GB)
cell.build()
print(f"{name} {basis}: 원자 {cell.natm}, 기저함수 {cell.nao}, 전자 {cell.nelectron}, 스레드 {lib.num_threads()}")

t0 = time.time()
mf = scf.RHF(cell).density_fit()
mf.with_df._cderi_to_save = tag + "_cderi.h5"   # 3중심 적분을 디스크에
mf.chkfile = tag + ".chk"
mf.conv_tol = 1e-7
mf.max_cycle = 100
mf.diis_space = 10
e_hf = mf.kernel()
dt = time.time() - t0
print(f"HF = {e_hf:.8f} Ha   converged={mf.converged}   {dt/60:.1f} min")
print(f"HOMO-LUMO gap = {(mf.mo_energy[cell.nelectron//2]-mf.mo_energy[cell.nelectron//2-1])*27.2114:.2f} eV")
with open("calc/pbc_hf/summary.txt", "a") as f:
    f.write(f"{name}\t{basis}\tnao={cell.nao}\tE={e_hf:.8f}\tconv={mf.converged}\t{dt/60:.1f}min\n")