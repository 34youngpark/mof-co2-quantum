"""
P4 검증: QNP 회로에 Rocca의 최적 파라미터를 넣어 에너지를 평가 → FCI 기준값과 비교.
(최적화 없이 회로 구현이 맞는지만 확인. Rocca Table 3: CO2 2.1e-8, MOF 3.3, MOFCO2 3.4 kJ/mol)
사용: python quantum/check_qnp.py [--gpu] [CO2|MOF|MOFCO2 ...]
"""
import sys, os, time, cudaq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum.qnp.hamiltonian import qubit_hamiltonian
from quantum.qnp.ansatz import qnp_ansatz, load_rocca_params, n_gates

cudaq.set_target("nvidia", option="fp64") if "--gpu" in sys.argv else cudaq.set_target("qpp-cpu")
KJ = 2625.5
ROCCA = os.path.expanduser("~/external/quantum_simulation_MOF/qnp_vqe_qiskit_simulator")
CASES = {"CO2": (4, "CO2", -25.76203867), "MOF": (10, "MOF", -55.27452186), "MOFCO2": (18, "MOFCO2", None)}
names = [a for a in sys.argv[1:] if a in CASES] or (["CO2"] if "--gpu" not in sys.argv else list(CASES))

for name in names:
    n_layers, sub, fci = CASES[name]
    H, nq, ne, ecore = qubit_hamiltonian(f"quantum/reference/reduced_FCIDUMP.{name}")
    theta, phi = load_rocca_params(f"{ROCCA}/{sub}/optimized_params.txt", nq, n_layers)
    t = time.time()
    e = cudaq.observe(qnp_ansatz, H, nq, ne, n_layers, theta, phi).expectation() - ecore
    dt = time.time() - t
    print(f"{name:7s} {nq:2d}q layers={n_layers:2d} gates={n_gates(nq, n_layers):3d} params={2*n_gates(nq, n_layers):3d}  "
          f"E_QNP={e:.8f}  FCI={fci}  " + (f"diff={(e-fci)*KJ:.2e} kJ/mol" if fci else "") + f"  ({dt:.1f}s/observe)")