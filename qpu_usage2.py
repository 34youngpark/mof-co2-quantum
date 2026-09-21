from qiskit_ibm_runtime import QiskitRuntimeService
s0 = QiskitRuntimeService(channel="ibm_quantum_platform")
for inst in s0.instances():
    s = QiskitRuntimeService(channel="ibm_quantum_platform", instance=inst['crn'])
    u = s.usage()
    print(inst['name'], "| plan:", inst['plan'])
    print("   allocated %d min | used %d min | remaining %d min | period %s ~ %s" % (
        u['usage_allocation_seconds']//60, u['usage_consumed_seconds']//60,
        u['usage_remaining_seconds']//60, u['usage_period']['start_time'][:10], u['usage_period']['end_time'][:10]))
