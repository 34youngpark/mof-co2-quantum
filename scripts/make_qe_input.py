"""
structures/01_built/*.cif → calc/relax_vdwdf2/<name>/espresso.pwi
사용: python scripts/make_qe_input.py a_MOF   (레포 루트에서)
"""
import sys, os
from ase.io import read
from ase.calculators.espresso import Espresso

name = sys.argv[1]                      # a_MOF / b_MOF_CO2_bound / c_MOF_CO2_free / d_CO2_box
atoms = read(f"structures/01_built/{name}.cif")
outdir = f"calc/relax_vdwdf2/{name}"
os.makedirs(outdir, exist_ok=True)

pseudos = {
    "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
    "N":  "N.pbe-n-radius_5.UPF",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
}

is_box = name.startswith("d_")
kpts = (1, 1, 1) if is_box else (1, 1, 2)

"""
structures/01_built/*.cif → calc/relax_vdwdf2/<name>/espresso.pwi
사용: python scripts/make_qe_input.py a_MOF   (레포 루트에서)
"""
import sys, os
from ase.io import read, write

name = sys.argv[1]                      # a_MOF / b_MOF_CO2_bound / c_MOF_CO2_free / d_CO2_box
atoms = read(f"structures/01_built/{name}.cif")
outdir = f"calc/relax_vdwdf2/{name}"
os.makedirs(outdir, exist_ok=True)

pseudos = {
    "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
    "N":  "N.pbe-n-radius_5.UPF",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
}

is_box = name.startswith("d_")
kpts = (1, 1, 1) if is_box else (1, 1, 2)

input_data = {
    "control":   {"calculation": "relax", "prefix": name, "outdir": "./tmp",
                  "pseudo_dir": os.path.expanduser("~/pseudo"),
                  "tprnfor": True, "nstep": 300, "disk_io": "low"},
    "system":    {"ecutwfc": 60, "ecutrho": 480, "input_dft": "vdw-df2",
                  "nspin": 1, "occupations": "fixed"},
    "electrons": {"conv_thr": 1e-7, "mixing_beta": 0.3, "electron_maxstep": 200},
    "ions":      {"ion_dynamics": "bfgs"},
}

write(f"{outdir}/espresso.pwi", atoms, format="espresso-in",
      input_data=input_data, pseudopotentials=pseudos, kpts=kpts)
print("생성:", f"{outdir}/espresso.pwi   (원자 {len(atoms)}개, k-mesh {kpts})")