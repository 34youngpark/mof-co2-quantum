"""
vdW-DF2 relax 4개의 결과를 검증하고 흡착에너지를 계산한다.
  - 최적화 구조를 structures/02_relaxed/*_relaxed.cif 로 저장
  - 기하 검사 (a 이동량, b 카바메이트, c CO2–N 거리)
  - Eq.1, Eq.2 계산 → results/vdwdf2_energies.csv
사용: python scripts/analyze_relax.py   (레포 루트에서)
"""
import os, csv
import numpy as np
from ase.io import read, write

RY = 13.605693          # eV
KJ = 96.485             # kJ/mol per eV
os.makedirs("structures/02_relaxed", exist_ok=True)
os.makedirs("results", exist_ok=True)

names = ["a_MOF", "b_MOF_CO2_bound", "c_MOF_CO2_free", "d_CO2_box"]
S, E = {}, {}
for n in names:
    pwo = f"calc/relax_vdwdf2/{n}/espresso.pwo"
    assert "JOB DONE" in open(pwo).read(), f"{n} 미완료"
    s = read(pwo, format="espresso-out", index=-1)
    S[n] = s
    E[n] = s.get_potential_energy() / RY          # Ry
    write(f"structures/02_relaxed/{n}_relaxed.cif", s)
    print(f"{n:18s} 원자 {len(s):3d}  E = {E[n]:.8f} Ry")

print("\n--- 기하 검사 ---")
a0 = read("structures/01_built/a_MOF.cif")
d = np.linalg.norm(S["a_MOF"].positions - a0.positions, axis=1)
print(f"a: 문헌 구조 대비 이동 최대 {d.max():.3f} Å, 평균 {d.mean():.3f} Å  (0.1 이하면 설정 일치)")

b = S["b_MOF_CO2_bound"]
Mg = [k for k in range(len(b)) if b[k].symbol == "Mg"]
db = b.get_distances(216, range(len(b)), mic=True)
near = sorted(range(len(b)), key=lambda k: db[k])[1:4]
print("b: C216 최근접 3개:", ", ".join(f"{b[k].symbol}{k} {db[k]:.3f}" for k in near))
print(f"b: 카바메이트 O–Mg 최단 {min(b.get_distances(k, Mg, mic=True).min() for k in (194, 195)):.2f} Å")

c = S["c_MOF_CO2_free"]
Ns = [k for k in range(len(c)) if c[k].symbol == "N"]
dcn = c.get_distances(216, Ns, mic=True).min()
print(f"c: CO2 C ↔ 최근접 N {dcn:.2f} Å  (5 이상이어야 '자유' 상태 유지)")
print(f"c: CO2 C–O {c.get_distance(216, 217, mic=True):.3f} / {c.get_distance(216, 218, mic=True):.3f} Å")

print("\n--- 흡착에너지 (vdW-DF2) ---")
dE1 = (E["b_MOF_CO2_bound"] - E["a_MOF"] - E["d_CO2_box"]) * RY
dE2 = (E["b_MOF_CO2_bound"] - E["c_MOF_CO2_free"]) * RY
dEphys = (E["c_MOF_CO2_free"] - E["a_MOF"] - E["d_CO2_box"]) * RY
print(f"Eq.1  ΔE1 = E(b)-E(a)-E(d) = {dE1:+.3f} eV = {dE1*KJ:+.1f} kJ/mol   (실험 ≈ -0.70 eV)")
print(f"Eq.2  ΔE2 = E(b)-E(c)      = {dE2:+.3f} eV = {dE2*KJ:+.1f} kJ/mol")
print(f"참고  E(c)-E(a)-E(d)       = {dEphys:+.3f} eV  (기공 안 CO2의 물리흡착)")

with open("results/vdwdf2_energies.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["functional", "E_a_Ry", "E_b_Ry", "E_c_Ry", "E_d_Ry", "dE1_eV", "dE2_eV", "dE1_kJmol", "dE1_minus_exp_kJmol"])
    w.writerow(["vdW-DF2", *[f"{E[n]:.8f}" for n in names], f"{dE1:.4f}", f"{dE2:.4f}", f"{dE1*KJ:.1f}", f"{dE1*KJ+70:.1f}"])
print("\n저장: structures/02_relaxed/*.cif, results/vdwdf2_energies.csv")