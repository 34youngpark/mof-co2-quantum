"""
고속 에너지 평가기: QNP 상태벡터 → 전자수 보존 부분공간(CI 벡터) → PySCF FCI 루틴으로 <Ψ|H|Ψ>.
cudaq.observe(수만 개 Pauli 항 개별 평가) 대비 수십~수백 배 빠름. 시뮬레이터 전용.
반환 에너지는 vqe_qnp/check_qnp 관례와 동일하게 ECORE 제외(관측값 − ecore와 같은 값).
"""
import time, gc, numpy as np, cudaq
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1
from pyscf.tools import fcidump
from .ansatz import qnp_ansatz


class FastEnergy:
    def __init__(self, fcidump_path, n_layers, verbose=True):
        d = fcidump.read(fcidump_path, verbose=False)
        self.norb, nelec = d["NORB"], d["NELEC"]
        self.nq, self.ne, self.nl = 2 * self.norb, nelec, n_layers
        self.na = self.nb = nelec // 2
        self.ecore = d["ECORE"]
        h1 = d["H1"]; eri = ao2mo.restore(8, d["H2"], self.norb)
        self.h2e = direct_spin1.absorb_h1e(h1, eri, self.norb, (self.na, self.nb), 0.5)
        # 행렬식 문자열 → 큐비트 정수 인덱스 (interleaved: 2p=α, 2p+1=β)
        sa = np.asarray(cistring.make_strings(range(self.norb), self.na), dtype=np.int64)
        sb = np.asarray(cistring.make_strings(range(self.norb), self.nb), dtype=np.int64)
        self.shape = (len(sa), len(sb))
        self.idx = {}
        for endian in ("little", "big"):
            pos = (lambda q: q) if endian == "little" else (lambda q: self.nq - 1 - q)
            qa = np.zeros(len(sa), np.int64); qb = np.zeros(len(sb), np.int64)
            for p in range(self.norb):
                qa |= ((sa >> p) & 1) << pos(2 * p)
                qb |= ((sb >> p) & 1) << pos(2 * p + 1)
            self.idx[endian] = (qa[:, None] | qb[None, :]).ravel()
        # 엔디언 판별: HF 상태(파라미터 0)의 진폭 1 위치로
        ng = self._ngates(); z = [0.0] * ng
        psi = self._state(z, z)
        hf_a = int(self.na and 0)  # placeholder
        for endian, ix in self.idx.items():
            c = psi[ix]
            if abs(np.abs(c).max() - 1.0) < 1e-3:
                self.endian = endian; break
        else:
            raise RuntimeError("HF 진폭을 부분공간에서 찾지 못함 - 큐비트 순서 확인")
        self.ix = self.idx[self.endian]
        del psi, c; gc.collect()
        if verbose:
            print(f"[FastEnergy] {self.nq}q ne={self.ne} dets={self.shape[0]}x{self.shape[1]}"
                  f"={self.shape[0]*self.shape[1]:,} / {2**self.nq:,}  endian={self.endian}")

    def _ngates(self):
        from .ansatz import n_gates
        return n_gates(self.nq, self.nl)

    def _state(self, theta, phi):
        st = cudaq.get_state(qnp_ansatz, self.nq, self.ne, self.nl, [float(x) for x in theta], [float(x) for x in phi])
        arr = np.array(st)
        del st; gc.collect()                      # GPU 상태벡터 즉시 해제 (28q: 4.3 GB)
        return arr

    def energy(self, theta, phi, return_leak=False):
        psi = self._state(theta, phi)
        c = psi[self.ix]
        if np.abs(c.imag).max() > 1e-4:
            raise RuntimeError("CI 진폭에 허수부 - 실수 회로가 아님")
        c = np.ascontiguousarray(c.real.reshape(self.shape), dtype=np.float64)
        norm2 = float(np.vdot(c, c).real)
        del psi; gc.collect()
        hc = direct_spin1.contract_2e(self.h2e, c, self.norb, (self.na, self.nb))
        e = float(np.vdot(c, hc).real) / norm2          # ECORE 제외 (관례 일치)
        return (e, 1.0 - norm2) if return_leak else e


if __name__ == "__main__":
    # 검증: cudaq.observe 와 비교 + 시간 측정.  사용: python -m quantum.qnp.fast_energy CO2 [MOF] [MOFCO2] [--gpu]
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from quantum.qnp.hamiltonian import qubit_hamiltonian
    from quantum.qnp.ansatz import n_gates
    cudaq.set_target("nvidia") if "--fp32" in sys.argv else (cudaq.set_target("nvidia", option="fp64") if "--gpu" in sys.argv else cudaq.set_target("qpp-cpu"))
    CASES = {"CO2": 4, "MOF": 10, "MOFCO2": 18}
    rng = np.random.default_rng(0)
    for name in [a for a in sys.argv[1:] if a in CASES]:
        path = f"quantum/reference/reduced_FCIDUMP.{name}"; nl = CASES[name]
        fe = FastEnergy(path, nl)
        ng = n_gates(fe.nq, nl)
        th, ph = list(rng.uniform(-0.5, 0.5, ng)), list(rng.uniform(-0.5, 0.5, ng))
        t = time.time(); e_fast, leak = fe.energy(th, ph, return_leak=True); t_fast = time.time() - t
        line = f"{name:7s} fast={e_fast:.10f}  leak={leak:.2e}  {t_fast:.2f}s"
        if name != "MOFCO2" or "--full" in sys.argv:
            H, nq, ne, ecore = qubit_hamiltonian(path)
            t = time.time(); e_obs = cudaq.observe(qnp_ansatz, H, nq, ne, nl, th, ph).expectation() - ecore; t_obs = time.time() - t
            line += f"  | observe={e_obs:.10f}  {t_obs:.1f}s  | diff={abs(e_fast-e_obs)*2625.5:.2e} kJ/mol  speedup x{t_obs/t_fast:.0f}"
        print(line)