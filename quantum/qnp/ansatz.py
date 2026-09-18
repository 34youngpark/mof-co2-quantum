"""
QNP (quantum-number-preserving) ansatz — Anselmetti et al. 2021 — CUDA-Q 구현.
게이트 분해와 fabric 배치는 Rocca et al. 공개 코드(qnp_utils.py, Qiskit)를 그대로 옮긴 것.
큐비트 순서: interleaved (2p = 오비탈 p의 α, 2p+1 = β). 4큐비트 게이트 = 공간 오비탈 2개.
"""
import cudaq
import numpy as np


@cudaq.kernel
def qnp_px(theta: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit):
    """전자쌍 교환 회전 QNP_PX(θ)"""
    pi = 3.141592653589793
    x.ctrl(q3, q1)
    x.ctrl(q3, q0)
    h(q3)
    s(q1)
    rz(-pi / 2, q0)
    x.ctrl(q1, q0)
    x.ctrl(q3, q2)
    rz(pi / 2, q0)
    ry(theta / 8, q2)
    ry(-theta / 8, q3)
    z.ctrl(q0, q3)
    x.ctrl(q0, q2)
    ry(-theta / 8, q3)
    ry(theta / 8, q2)
    x.ctrl(q1, q2)
    x.ctrl(q1, q3)
    rz(pi / 2, q1)
    ry(-theta / 8, q2)
    ry(theta / 8, q3)
    x.ctrl(q0, q2)
    z.ctrl(q0, q3)
    ry(-theta / 8, q2)
    ry(theta / 8, q3)
    x.ctrl(q3, q2)
    h(q3)
    x.ctrl(q3, q1)
    s(q3)
    rz(-pi / 2, q1)
    x.ctrl(q1, q0)


@cudaq.kernel
def qnp_or(phi: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit):
    """오비탈 회전 QNP_OR(φ): α, β 각각 Givens rotation"""
    h(q0)
    h(q1)
    x.ctrl(q0, q2)
    x.ctrl(q1, q3)
    ry(phi / 2, q0)
    ry(phi / 2, q1)
    ry(phi / 2, q2)
    ry(phi / 2, q3)
    x.ctrl(q1, q3)
    x.ctrl(q0, q2)
    h(q1)
    h(q0)


@cudaq.kernel
def qnp_ansatz(nq: int, nelec: int, n_layers: int, theta: list[float], phi: list[float]):
    """HF 초기화 + QNP fabric. theta[i], phi[i]는 전역 게이트 번호 i의 파라미터."""
    q = cudaq.qvector(nq)
    for i in range(nelec):
        x(q[i])
    ig = 0
    for nl in range(2 * n_layers):
        if nl % 2 == 0:
            ngate = nq // 4
            start = 0
        else:
            ngate = (nq - 1) // 4
            if nq % 2 != 0:
                start = nq % 4
            else:
                start = 2
        for g in range(ngate):
            s0 = start + 4 * g
            qnp_px(theta[ig], q[s0], q[s0 + 1], q[s0 + 2], q[s0 + 3])
            qnp_or(phi[ig], q[s0], q[s0 + 1], q[s0 + 2], q[s0 + 3])
            ig += 1


def gate_layout(nq, n_layers):
    """sublayer별 게이트 수 목록 (Rocca count_QNP_params 와 동일)"""
    out = []
    for nl in range(2 * n_layers):
        out.append(nq // 4 if nl % 2 == 0 else (nq - 1) // 4)
    return out


def n_gates(nq, n_layers):
    return sum(gate_layout(nq, n_layers))


def load_rocca_params(path, nq, n_layers):
    """Rocca optimized_params.txt → (theta, phi). 파일 값은 ×2, 순서는 sublayer별 [θ...][φ...]"""
    vals = 2.0 * np.array([float(v) for v in open(path).read().split()])
    theta, phi = [], []
    k = 0
    for ng in gate_layout(nq, n_layers):
        theta += list(vals[k:k + ng]); k += ng
        phi += list(vals[k:k + ng]); k += ng
    assert k == len(vals), (k, len(vals))
    return theta, phi