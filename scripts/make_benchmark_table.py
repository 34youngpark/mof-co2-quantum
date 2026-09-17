"""
calc/benchmark/*/*/espresso.pwo + vdW-DF2 relax 결과 → results/dft_benchmark.csv
사용: python scripts/make_benchmark_table.py
"""
import re, csv, os

RY, KJ, EXP = 13.605693, 96.485, -0.70          # eV/Ry, kJ/mol per eV, 실험 ΔE (eV)
names = ["a_MOF", "b_MOF_CO2_bound", "c_MOF_CO2_free", "d_CO2_box"]

def energy(pwo):
    txt = open(pwo, errors="ignore").read()
    if "JOB DONE" not in txt: return None
    return float(re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)", txt, re.M)[-1])

rows = []
funcs = [("vdW-DF2", "calc/relax_vdwdf2"), ("PBE", "calc/benchmark/pbe"),
         ("PBE-D3", "calc/benchmark/pbe-d3"), ("vdW-DF", "calc/benchmark/vdw-df")]
for label, base in funcs:
    E = {n: energy(f"{base}/{n}/espresso.pwo") for n in names}
    if None in E.values():
        print(f"{label}: 누락 {[n for n in names if E[n] is None]}"); continue
    dE1 = (E["b_MOF_CO2_bound"] - E["a_MOF"] - E["d_CO2_box"]) * RY
    dE2 = (E["b_MOF_CO2_bound"] - E["c_MOF_CO2_free"]) * RY
    rows.append([label, *[f"{E[n]:.6f}" for n in names],
                 f"{dE1:.3f}", f"{dE2:.3f}", f"{dE1*KJ:.1f}", f"{(dE1-EXP)*KJ:+.1f}"])

os.makedirs("results", exist_ok=True)
hdr = ["functional", "E_a_Ry", "E_b_Ry", "E_c_Ry", "E_d_Ry", "dE1_eV", "dE2_eV", "dE1_kJmol", "dE1-exp_kJmol"]
with open("results/dft_benchmark.csv", "w", newline="") as f:
    csv.writer(f).writerows([hdr, *rows, ["experiment", "", "", "", "", f"{EXP:.2f}", "", f"{EXP*KJ:.1f}", "0"]])

print(f"{'범함수':10s} {'ΔE1 (eV)':>10s} {'ΔE2 (eV)':>10s} {'ΔE1 (kJ/mol)':>13s} {'실험 대비':>10s}")
for r in rows: print(f"{r[0]:10s} {r[5]:>10s} {r[6]:>10s} {r[7]:>13s} {r[8]:>10s}")
print(f"{'실험':10s} {EXP:>10.2f} {'':>10s} {EXP*KJ:>13.1f} {'0':>10s}")
print("\n저장: results/dft_benchmark.csv")