#!/usr/bin/env python3
"""
5.3 (7): ADAPT-VQE (UCCSD 풀, cudaq_algorithms) — QNP 비교용. Owens Table I 형식 비용 기록.
사용: python quantum/adapt_vqe.py CO2            (CPU, 8q 검증)
      python quantum/adapt_vqe.py MOF --gpu --max-ops 40
      python quantum/adapt_vqe.py b --fcidump quantum/reference/FCIDUMP.b.20q --fci <E_CASCI> --gpu
"""
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cudaq
from scipy.optimize import minimize
from cudaq_algorithms.stateprep import make_uccsd_operator_pool
from quantum.qnp.hamiltonian import qubit_hamiltonian

KJ = 2625.5
CASES = {"CO2": -25.76203867, "MOF": -55.27452186, "MOFCO2": -74.70598653}
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--fcidump"); ap.add_argument("--fci", type=float)
ap.add_argument("--gpu", action="store_true"); ap.add_argument("--max-ops", type=int, default=30)
ap.add_argument("--grad-tol", type=float, default=1e-3); ap.add_argument("--maxiter", type=int, default=100)
ap.add_argument("--tag", default="")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
cudaq.set_target("nvidia", option="fp64") if a.gpu else cudaq.set_target("qpp-cpu")
if a.name in CASES: fcidump, fci, is_total = f"quantum/reference/reduced_FCIDUMP.{a.name}", CASES[a.name], False
else: fcidump, fci, is_total = a.fcidump, a.fci, True
H, nq, ne, ecore = qubit_hamiltonian(fcidump)
if fci is not None and is_total: fci -= ecore
pool = make_uccsd_operator_pool(nq, ne, 0)

def terms_of(op):
    """SpinOperator → [(pauli_word_str, coef)]"""
    out = []
    for t in op:
        c = t.evaluate_coefficient() if hasattr(t, "evaluate_coefficient") else t.get_coefficient()
        c = complex(c).real
        w = t.get_pauli_word(nq) if hasattr(t, "get_pauli_word") else str(t.to_string(False)).split()[-1]
        out.append((w, c))
    return out
pool_terms = [terms_of(p) for p in pool]
print(f"[{a.name}] {nq}q ne={ne} pool={len(pool)} ops  FCI(elec)={fci}", flush=True)

@cudaq.kernel
def adapt_kernel(nq: int, ne: int, theta: list[float], idx: list[int], coefs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(nq)
    for i in range(ne): x(q[i])
    for k in range(len(words)):
        exp_pauli(theta[idx[k]] * coefs[k], q, words[k])

sel = []            # 선택된 풀 인덱스
_ck = f"adapt_{a.name}{a.tag}.npz"
_resume = a.resume and os.path.exists(_ck)
def flat(sel):
    idx, coefs, words = [], [], []
    for j, k in enumerate(sel):
        for w, c in pool_terms[k]:
            idx.append(j); coefs.append(c); words.append(cudaq.pauli_word(w))
    return idx, coefs, words
nfev = [0]
def energy(theta, sel):
    idx, coefs, words = flat(sel); nfev[0] += 1
    return cudaq.observe(adapt_kernel, H, nq, ne, [float(v) for v in theta], idx, coefs, words).expectation() - ecore

if _resume:
    _z = np.load(_ck); sel = [int(k) for k in _z["sel"]]; theta = np.array(_z["theta"]); hist = [tuple(h) for h in _z["hist"]]
    e = energy(theta, sel); print(f"[resume] {_ck}: nops={len(sel)}", flush=True)
else:
    theta = np.zeros(0); hist = []; e = energy(theta, sel)
t0 = time.time()
print(f"[init] E={e:.8f}" + (f"  ΔFCI={(e-fci)*KJ:.3f} kJ/mol" if fci is not None else ""), flush=True)
eps = 1e-3
for step in range(len(sel), a.max_ops):
    # 기울기: 후보 op를 끝에 붙이고 ±eps 유한차분
    g = np.zeros(len(pool))
    for k in range(len(pool)):
        if k in sel: continue
        th = np.append(theta, 0.0); s2 = sel + [k]
        th[-1] = eps;  ep = energy(th, s2)
        th[-1] = -eps; em = energy(th, s2)
        g[k] = (ep - em) / (2 * eps)
    kbest = int(np.argmax(np.abs(g))); gmax = abs(g[kbest])
    if gmax < a.grad_tol:
        print(f"[stop] max|grad|={gmax:.2e} < tol", flush=True); break
    sel.append(kbest); theta = np.append(theta, 0.0)
    res = minimize(lambda th: energy(th, sel), theta, method="L-BFGS-B",
                   options={"maxiter": a.maxiter, "ftol": 1e-12, "gtol": 1e-6})
    theta, e = res.x, res.fun
    hist.append((len(sel), e, nfev[0], time.time() - t0))
    print(f"  step {step+1:3d}  op={kbest:4d}  |g|={gmax:.2e}  nops={len(sel):3d}  E={e:.8f}"
          + (f"  ΔFCI={(e-fci)*KJ:8.3f} kJ/mol" if fci is not None else "")
          + f"  nfev={nfev[0]:6d}  {(time.time()-t0)/60:.1f} min", flush=True)
    np.savez(f"adapt_{a.name}{a.tag}.npz", sel=np.array(sel), theta=theta, e=e, hist=np.array(hist), ecore=ecore)
print(f"\n[done] E_ADAPT={e:.8f}  nops={len(sel)}  nfev={nfev[0]}  {(time.time()-t0)/60:.1f} min"
      + (f"  ΔFCI={(e-fci)*KJ:.4f} kJ/mol" if fci is not None else ""), flush=True)