import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService(channel="ibm_quantum_platform")
for inst in s.instances():
    print("=== instance:", inst['name'], "| plan:", inst['plan'])
    try:
        for b in s.backends(instance=inst['crn']):
            st = b.status()
            print(" ", b.name, b.num_qubits, "q | operational:", st.operational, "| pending:", st.pending_jobs)
            try:
                p = b.properties()
                g2 = [g.parameters[0].value for g in p.gates if len(g.qubits) == 2]
                ro = [p.readout_error(q) for q in range(b.num_qubits)]
                print("    2q err median %.4f | readout err median %.4f" % (np.median(g2), np.median(ro)))
            except Exception as e:
                print("    (properties 못 읽음:", type(e).__name__, ")")
    except Exception as e:
        print("  backends 조회 실패:", type(e).__name__, str(e)[:120])
