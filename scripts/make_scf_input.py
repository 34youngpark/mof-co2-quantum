"""
02_relaxed 구조로 single-point(scf) 입력 생성.
사용: python scripts/make_scf_input.py <functional> <name>
  functional: pbe | pbe-d3 | vdw-df | vdw-df2 | scan
"""
import sys, os
from ase.io import read, write

func, name = sys.argv[1], sys.argv[2]
atoms = read(f"structures/02_relaxed/{name}_relaxed.cif")
outdir = f"calc/benchmark/{func}/{name}"
os.makedirs(outdir, exist_ok=True)

pseudos = {
    "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
    "N":  "N.pbe-n-radius_5.UPF",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
}
FUNC = {
    "pbe":     {"input_dft": "pbe"},
    "pbe-d3":  {"input_dft": "pbe", "vdw_corr": "dft-d3"},
    "vdw-df":  {"input_dft": "vdw-df"},
    "vdw-df2": {"input_dft": "vdw-df2"},
    "scan":    {"input_dft": "scan"},
}
kpts = (1, 1, 1) if name.startswith("d_") else (1, 1, 2)
input_data = {
    "control":   {"calculation": "scf", "prefix": name, "outdir": "./tmp",
                  "pseudo_dir": os.path.expanduser("~/pseudo"), "disk_io": "none"},
    "system":    {"ecutwfc": 60, "ecutrho": 480, "nspin": 1, "occupations": "fixed", **FUNC[func]},
    "electrons": {"conv_thr": 1e-7, "mixing_beta": 0.3, "electron_maxstep": 200},
}
write(f"{outdir}/espresso.pwi", atoms, format="espresso-in",
      input_data=input_data, pseudopotentials=pseudos, kpts=kpts)
print("생성:", f"{outdir}/espresso.pwi")