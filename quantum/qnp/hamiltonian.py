"""
FCIDUMP → Jordan-Wigner 큐비트 해밀토니안 (CUDA-Q SpinOperator).

큐비트 라벨은 interleaved: qubit 2p = 오비탈 p의 α, qubit 2p+1 = 오비탈 p의 β.
단, Jordan-Wigner 변환 자체는 Rocca(qiskit-nature)와 같이 블록 순서(α0..α(n-1), β0..β(n-1))로
수행한 뒤 큐비트 번호만 interleaved로 바꾼다. 그래서 Z-string이 같은 스핀 큐비트에만 걸린다.
페르미온 모드 순서가 다르면 같은 회로가 다른 에너지를 주므로, Rocca의 최적 파라미터를
재현하려면 이 관례를 반드시 지켜야 한다. (2026-09-18 검증: 이 관례로 CO2 FCI 재현)
"""
import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump
from openfermion import InteractionOperator, jordan_wigner
import cudaq


def read_fcidump(path):
    d = fcidump.read(path, verbose=False)
    norb, nelec = d["NORB"], d["NELEC"]
    h1 = d["H1"]
    eri = ao2mo.restore(1, d["H2"], norb)          # (pq|rs) chemist, 4-index
    return h1, eri, d["ECORE"], norb, nelec


def fermion_operator(path):
    """FCIDUMP → openfermion InteractionOperator. 스핀오비탈 인덱스는 블록 순서:
    s = p (α), s = norb + p (β)."""
    h1, eri, ecore, norb, nelec = read_fcidump(path)
    n = norb
    h1s = np.zeros((2 * n, 2 * n))
    h1s[:n, :n] = h1
    h1s[n:, n:] = h1
    # H = Σ h_pq a†p aq + ½ Σ (pq|rs) a†pσ a†rτ asτ aqσ  (chemist)
    h2s = np.zeros((2 * n,) * 4)
    for so, to in [(0, 0), (0, n), (n, 0), (n, n)]:
        h2s[so:so + n, to:to + n, to:to + n, so:so + n] += eri.transpose(0, 2, 3, 1)
    return InteractionOperator(ecore, h1s, 0.5 * h2s)


def qubit_hamiltonian(path):
    """FCIDUMP 경로 → (cudaq.SpinOperator, nqubits, nelec, ecore)"""
    h1, eri, ecore, norb, nelec = read_fcidump(path)
    qop = jordan_wigner(fermion_operator(path))
    nq = 2 * norb
    relabel = lambda s: 2 * s if s < norb else 2 * (s - norb) + 1   # 블록 → interleaved
    H = cudaq.SpinOperator.empty()
    for term, coef in qop.terms.items():
        w = ["I"] * nq
        for idx, p in term:
            w[relabel(idx)] = p
        H += coef.real * cudaq.SpinOperator.from_word("".join(w))
    return H, 2 * norb, nelec, ecore


def hf_bitstring(nqubits, nelec):
    """HF 행렬식: 낮은 공간 오비탈부터 α,β 채움 → 처음 nelec개 큐비트가 1"""
    return [1] * nelec + [0] * (nqubits - nelec)