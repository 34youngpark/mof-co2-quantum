import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService()
for b in s.backends():
    print(b.name, b.num_qubits, "pending jobs:", b.status().pending_jobs)
    p = b.properties()
    g2 = [g.parameters[0].value for g in p.gates if len(g.qubits) == 2]
    ro = [p.readout_error(q) for q in range(b.num_qubits)]
    print("  2q gate err median: %.4f   readout err median: %.4f" % (np.median(g2), np.median(ro)))
