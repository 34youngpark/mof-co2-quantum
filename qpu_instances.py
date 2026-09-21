from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService(channel="ibm_quantum_platform")   # instance 없이 접속
for i in s.instances():
    print(i)
