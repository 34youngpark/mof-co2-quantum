"""
calc/pbc_hf/summary.txt 에서 gth-dzvp HF 에너지를 읽어 Eq.1, Eq.2 계산.
사용: python scripts/hf_binding.py
"""
HA, KJ = 27.211386, 96.485
E = {}
for line in open("calc/pbc_hf/summary.txt"):
    f = line.split()
    if f[1] == "gth-dzvp":
        E[f[0]] = float(f[3].split("=")[1])
need = ["a_MOF", "b_MOF_CO2_bound", "c_MOF_CO2_free", "d_CO2_box"]
miss = [n for n in need if n not in E]
if miss: raise SystemExit(f"아직 없음: {miss}")
dE1 = (E["b_MOF_CO2_bound"] - E["a_MOF"] - E["d_CO2_box"]) * HA
dE2 = (E["b_MOF_CO2_bound"] - E["c_MOF_CO2_free"]) * HA
print(f"HF/gth-dzvp  Eq.1 ΔE1 = {dE1:+.3f} eV = {dE1*KJ:+.1f} kJ/mol")
print(f"HF/gth-dzvp  Eq.2 ΔE2 = {dE2:+.3f} eV = {dE2*KJ:+.1f} kJ/mol")
print(f"vdW-DF2 기준: ΔE1 = -0.664 eV, ΔE2 = -0.612 eV → HF와의 차이가 회수해야 할 상관에너지")