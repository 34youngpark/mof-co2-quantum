#!/usr/bin/env python3
"""
P5: QNP-VQE 최적화 루프 (SciPy L-BFGS-B + basin-hopping), CUDA-Q.
  python quantum/vqe_qnp.py CO2 --gpu                                # Rocca 참조계, observe
  python quantum/vqe_qnp.py MOF --gpu --fast --bh 5                  # 고속 평가기 + basin-hopping
  python quantum/vqe_qnp.py b --fcidump quantum/reference/FCIDUMP.b.28q --layers 18 \
         --fci -1304.xx --fast --fp32 --tag _28q                      # 우리 계 (fci는 총에너지, ecore 자동 차감)
  python quantum/vqe_qnp.py b ... --layers 8  --tag _28q_L8          # 1단계: 8레이어
  python quantum/vqe_qnp.py b ... --layers 18 --init-from vqe_b_28q_L8.npz --tag _28q   # 2단계: warm start
  python quantum/vqe_qnp.py b ... --fast --init-from vqe_b_28q.npz --eval-only            # fp64 최종 재평가
체크포인트: vqe_<name><tag>.npz (best_x, best_e, hist). --resume 로 이어서.
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
ap.add_argument("--gpu", action="store_true"); ap.add_argument("--fp32", action="store_true")
ap.add_argument("--fast", action="store_true", help="상태벡터+CI 수축 고속 평가기 (시뮬레이터 전용)")
ap.add_argument("--init", default="zero", choices=["zero", "rocca", "random"])
ap.add_argument("--init-from", help="이전 체크포인트 npz (레이어 적어도 됨: 새 레이어는 0으로 채움)")
ap.add_argument("--eval-only", action="store_true", help="초기값 에너지만 평가하고 종료")
ap.add_argument("--maxiter", type=int, default=200); ap.add_argument("--bh", type=int, default=0)
ap.add_argument("--step", type=float, default=0.3); ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
a = ap.parse_args()

if a.fp32: cudaq.set_target("nvidia")
elif a.gpu: cudaq.set_target("nvidia", option="fp64")
else: cudaq.set_target("qpp-cpu")

if a.name in CASES:
    n_layers, fci = CASES[a.name]; fcidump = f"quantum/reference/reduced_FCIDUMP.{a.name}"; fci_is_total = False
else:
    n_layers, fci, fcidump, fci_is_total = a.layers, a.fci, a.fcidump, True
    assert n_layers and fcidump, "우리 계는 --fcidump, --layers 필수"
if a.layers: n_layers = a.layers

H, nq, ne, ecore = qubit_hamiltonian(fcidump)
if fci is not None and fci_is_total: fci -= ecore          # 내부 관례: ECORE 제외 전자 에너지
ng = n_gates(nq, n_layers); npar = 2 * ng
out = f"vqe_{a.name}{a.tag}.npz"
print(f"[{a.name}] {nq}q ne={ne} layers={n_layers} gates={ng} params={npar}  FCI(elec)={fci}  ecore={ecore:.6f}")

if a.fast:
    from quantum.qnp.fast_energy import FastEnergy
    fe = FastEnergy(fcidump, n_layers)
    def energy(x): return fe.energy(x[:ng], x[ng:])
else:
    def energy(x):
        return cudaq.observe(qnp_ansatz, H, nq, ne, n_layers,
                             [float(v) for v in x[:ng]], [float(v) for v in x[ng:]]).expectation() - ecore

# --- 초기값 ---
rng = np.random.default_rng(a.seed)
if a.resume and os.path.exists(out):
    x0 = np.load(out)["best_x"]; print(f"[init] resume from {out}")
elif a.init_from:
    z = np.load(a.init_from); xo = z["best_x"]; lo = int(z["layers"]); ngo = n_gates(nq, lo)
    assert int(z["nq"]) == nq and lo <= n_layers and len(xo) == 2 * ngo
    x0 = np.concatenate([xo[:ngo], np.zeros(ng - ngo), xo[ngo:], np.zeros(ng - ngo)])
    print(f"[init] warm start from {a.init_from} ({lo} → {n_layers} layers, {2*ngo} → {npar} params)")
elif a.init == "rocca":
    th, ph = load_rocca_params(f"{ROCCA}/{a.name}/optimized_params.txt", nq, n_layers); x0 = np.array(th + ph)
elif a.init == "random":
    x0 = rng.uniform(-0.1, 0.1, npar)
else:
    x0 = np.zeros(npar)
t = time.time(); e0 = energy(x0); t_obs = time.time() - t
print(f"[init] E={e0:.8f}" + (f"  (total {e0+ecore:.8f})" if fci_is_total else "") +
      f"  ΔFCI={(e0-fci)*KJ:.3f} kJ/mol" * (fci is not None) +
      f"  ({t_obs:.2f}s/eval → 1 gradient ≈ {t_obs*(npar+1)/60:.1f} min)")
if a.eval_only: sys.exit(0)

# --- 로그/체크포인트 ---
best = {"e": e0, "x": x0.copy()}; hist = []; t0 = time.time(); nfev = [0]
def save(): np.savez(out, best_x=best["x"], best_e=best["e"], hist=np.array(hist), nq=nq, layers=n_layers, ecore=ecore)
def f(x):
    e = energy(x); nfev[0] += 1
    if e < best["e"]: best["e"], best["x"] = e, x.copy(); save()
    return e
def cb(x, *_):
    e = best["e"]; hist.append((time.time() - t0, nfev[0], e))
    msg = f"  iter {len(hist):4d}  nfev={nfev[0]:6d}  E_best={e:.8f}  {(time.time()-t0)/60:6.1f} min"
    if fci is not None: msg += f"  ΔFCI={(e-fci)*KJ:8.3f} kJ/mol"
    print(msg, flush=True)

opts = dict(method="L-BFGS-B", options={"maxiter": a.maxiter, "ftol": 1e-12, "gtol": 1e-6})
if a.bh == 0:
    minimize(f, x0, callback=cb, **opts)
else:
    def bh_cb(x, e, accepted): print(f"[hop] E={e:.8f} accepted={accepted}  best={best['e']:.8f}", flush=True)
    basinhopping(f, x0, niter=a.bh, stepsize=a.step, seed=a.seed,
                 minimizer_kwargs=dict(callback=cb, **opts), callback=bh_cb)
save(); e = best["e"]
print(f"\n[done] E_VQE={e:.8f}" + (f"  (total {e+ecore:.8f})" if fci_is_total else "") +
      f"  nfev={nfev[0]}  {(time.time()-t0)/60:.1f} min  → {out}")
if fci is not None: print(f"       ΔFCI = {(e-fci)*KJ:.4f} kJ/mol  (chemical accuracy 4.2)")