from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService(channel="ibm_quantum_platform")
for inst in s.instances():
    try:
        print(inst['name'], "→", s.usage(instance=inst['crn']) if 'instance' in s.usage.__code__.co_varnames else s.usage())
    except Exception as e:
        print(inst['name'], "usage 조회 실패:", type(e).__name__, str(e)[:100])
