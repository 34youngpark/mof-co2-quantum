"""
FCIDUMP → Jordan-Wigner 큐비트 해밀토니안 (CUDA-Q SpinOperator).
큐비트 순서는 interleaved: qubit 2p = 오비탈 p의 α, qubit 2p+1 = 오비탈 p의 β.
(Rocca의 QNP 회로가 4큐비트 = 인접한 공간 오비탈 2개 단위로 작동하므로 이 순서가 필요)
"""
import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump
from openfermion import InteractionOperator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
import cudaq


def read_fcidump(path):
    d = fcidump.read(path, verbose=False)
    norb, nelec = d["NORB"], d["NELEC"]
    h1 = d["H1"]
    eri = ao2mo.restore(1, d["H2"], norb)          # (pq|rs) chemist, 4-index
    return h1, eri, d["ECORE"], norb, nelec


def fermion_operator(path):
    """FCIDUMP → openfermion InteractionOperator (spin-orbital, interleaved α/β)"""
    h1, eri, ecore, norb, nelec = read_fcidump(path)
    two = np.asarray(eri.transpose(0, 2, 3, 1), order="C")   # openfermion 규약
    h1s, h2s = spinorb_from_spatial(h1, two)
    return InteractionOperator(ecore, h1s, 0.5 * h2s)


def qubit_hamiltonian(path):
    """FCIDUMP 경로 → (cudaq.SpinOperator, nqubits, nelec, ecore)"""
    h1, eri, ecore, norb, nelec = read_fcidump(path)
    qop = jordan_wigner(fermion_operator(path))
    nq = 2 * norb
    H = cudaq.SpinOperator.empty()
    for term, coef in qop.terms.items():
        w = ["I"] * nq
        for idx, p in term:
            w[idx] = p
        H += coef.real * cudaq.SpinOperator.from_word("".join(w))
    return H, 2 * norb, nelec, ecore


def hf_bitstring(nqubits, nelec):
    """HF 행렬식: 낮은 공간 오비탈부터 α,β 채움 → 처음 nelec개 큐비트가 1"""
    return [1] * nelec + [0] * (nqubits - nelec)