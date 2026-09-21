#!/usr/bin/env python3
"""
P5: QNP-VQE 최적화 루프 (SciPy L-BFGS-B + basin-hopping), CUDA-Q.
사용:
  python quantum/vqe_qnp.py CO2 --gpu                          # HF(0)에서 시작, 기본 설정
  python quantum/vqe_qnp.py MOF --gpu --bh 5 --maxiter 300     # basin-hopping 5회
  python quantum/vqe_qnp.py CO2 --init rocca                   # Rocca 최적값에서 시작(수렴 검증용)
  python quantum/vqe_qnp.py b --fcidump path/FCIDUMP.b --layers 18 --fci -74.7   # 우리 계
체크포인트: vqe_<name>.npz (best_x, best_e, history). --resume 로 이어서.
"""
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cudaq
from scipy.optimize import minimize, basinhopping
from quantum.qnp.hamiltonian import qubit_hamiltonian
from quantum.qnp.ansatz import qnp_ansatz, load_rocca_params, n_gates

KJ = 2625.5
ROCCA = os.path.expanduser("~/external/quantum_simulation_MOF/qnp_vqe_qiskit_simulator")
CASES = {"CO2": (4, -25.76203867), "MOF": (10, -55.27452186), "MOFCO2": (18, -74.70598653)}

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--fcidump"); ap.add_argument("--layers", type=int); ap.add_argument("--fci", type=float)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--init", default="zero", choices=["zero", "rocca", "random"])
ap.add_argument("--maxiter", type=int, default=200, help="L-BFGS-B 반복 상한")
ap.add_argument("--bh", type=int, default=0, help="basin-hopping 홉 수 (0=끄기)")
ap.add_argument("--step", type=float, default=0.3, help="basin-hopping 섭동 크기 (rad)")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
a = ap.parse_args()

cudaq.set_target("nvidia", option="fp64") if a.gpu else cudaq.set_target("qpp-cpu")
if a.name in CASES:
    n_layers, fci = CASES[a.name]; fcidump = f"quantum/reference/reduced_FCIDUMP.{a.name}"
else:
    n_layers, fci, fcidump = a.layers, a.fci, a.fcidump
    assert n_layers and fcidump, "우리 계는 --fcidump, --layers 필수"
if a.layers: n_layers = a.layers

H, nq, ne, ecore = qubit_hamiltonian(fcidump)
ng = n_gates(nq, n_layers); npar = 2 * ng
out = f"vqe_{a.name}{a.tag}.npz"
print(f"[{a.name}] {nq}q ne={ne} layers={n_layers} gates={ng} params={npar}  FCI={fci}")

def energy(x):
    return cudaq.observe(qnp_ansatz, H, nq, ne, n_layers, list(x[:ng]), list(x[ng:])).expectation() - ecore

# --- 초기값 ---
rng = np.random.default_rng(a.seed)
if a.resume and os.path.exists(out):
    x0 = np.load(out)["best_x"]; print(f"[init] resume from {out}")
elif a.init == "rocca":
    th, ph = load_rocca_params(f"{ROCCA}/{a.name}/optimized_params.txt", nq, n_layers); x0 = np.array(th + ph)
elif a.init == "random":
    x0 = rng.uniform(-0.1, 0.1, npar)
else:
    x0 = np.zeros(npar)                                     # = HF 상태
e0 = energy(x0); t_obs = time.time(); energy(x0); t_obs = time.time() - t_obs
print(f"[init] E={e0:.8f}  ({t_obs:.2f}s/observe → 1 gradient ≈ {t_obs*(npar+1)/60:.1f} min)")

# --- 로그/체크포인트 ---
best = {"e": e0, "x": x0.copy()}; hist = []; t0 = time.time(); nfev = [0]
def f(x):
    e = energy(x); nfev[0] += 1
    if e < best["e"]:
        best["e"], best["x"] = e, x.copy()
        np.savez(out, best_x=best["x"], best_e=best["e"], hist=np.array(hist), nq=nq, layers=n_layers)
    return e
def cb(x, *_):
    e = best["e"]; hist.append((time.time() - t0, nfev[0], e))
    msg = f"  iter {len(hist):4d}  nfev={nfev[0]:6d}  E_best={e:.8f}  {(time.time()-t0)/60:6.1f} min"
    if fci: msg += f"  ΔFCI={(e-fci)*KJ:8.3f} kJ/mol"
    print(msg, flush=True)

opts = dict(method="L-BFGS-B", options={"maxiter": a.maxiter, "ftol": 1e-12, "gtol": 1e-6})
if a.bh == 0:
    res = minimize(f, x0, callback=cb, **opts)
else:
    def bh_cb(x, e, accepted):
        print(f"[hop] E={e:.8f} accepted={accepted}  best={best['e']:.8f}", flush=True)
    res = basinhopping(f, x0, niter=a.bh, stepsize=a.step, seed=a.seed,
                       minimizer_kwargs=dict(callback=cb, **opts), callback=bh_cb)

np.savez(out, best_x=best["x"], best_e=best["e"], hist=np.array(hist), nq=nq, layers=n_layers)
e = best["e"]
print(f"\n[done] E_VQE={e:.8f}  nfev={nfev[0]}  {(time.time()-t0)/60:.1f} min  → {out}")
if fci: print(f"       ΔFCI = {(e-fci)*KJ:.4f} kJ/mol  (chemical accuracy 4.2)")