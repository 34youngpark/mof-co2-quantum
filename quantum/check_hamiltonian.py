"""
P3 검증: FCIDUMP → 큐비트 해밀토니안이 맞는지 확인.
  - 세 계의 큐비트 수·항 수·HF 행렬식 에너지 (Rocca 기준값과 비교)
  - CO2(8큐비트)는 4전자 부분공간에서 정확 대각화 → FCI 기준값과 비교
Rocca의 기준값은 FCIDUMP의 ECORE 상수를 뺀 값이므로, 비교할 때 ecore를 뺀다.
사용: python quantum/check_hamiltonian.py [--gpu]
"""
import sys, os, time, numpy as np, cudaq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum.qnp.hamiltonian import qubit_hamiltonian, hf_bitstring, fermion_operator
from openfermion import get_sparse_operator
from openfermion.linalg import jw_number_restrict_operator, get_ground_state

cudaq.set_target("nvidia" if "--gpu" in sys.argv else "qpp-cpu")
REF = {"CO2": (-25.71575269, -25.76203867), "MOF": (-52.86078332, -55.27452186), "MOFCO2": (None, None)}
KJ = 2625.5

@cudaq.kernel
def hf_state(n: int, occ: list[int]):
    q = cudaq.qvector(n)
    for i in range(n):
        if occ[i] == 1:
            x(q[i])

for name, (hf_ref, fci_ref) in REF.items():
    path = f"quantum/reference/reduced_FCIDUMP.{name}"
    t = time.time()
    H, nq, ne, ecore = qubit_hamiltonian(path)
    tb = time.time() - t
    if nq > 8 and "--gpu" not in sys.argv:
        print(f"{name:7s} qubits={nq:2d} nelec={ne:2d} terms={H.term_count:6d}  (build {tb:.1f}s; HF 기대값은 --gpu 에서)")
        continue
    e_hf = cudaq.observe(hf_state, H, nq, hf_bitstring(nq, ne)).expectation() - ecore
    print(f"{name:7s} qubits={nq:2d} nelec={ne:2d} terms={H.term_count:6d}  E_HF={e_hf:.8f}  ref={hf_ref}  "
          f"diff={(e_hf-hf_ref)*KJ if hf_ref else float('nan'):.1e} kJ/mol  ({time.time()-t:.1f}s)")
    if nq == 8:
        M = get_sparse_operator(fermion_operator(path), n_qubits=nq)
        M = jw_number_restrict_operator(M, ne, nq)
        e0 = get_ground_state(M)[0] - ecore
        print(f"        4전자 부분공간 정확 대각화 = {e0:.8f}  FCI ref = {fci_ref}  diff = {(e0-fci_ref)*KJ:.1e} kJ/mol")