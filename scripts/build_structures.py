"""
Build the four Stage 5.1 structures from the two Siegelman 2019 CIFs in structures/00_literature.
  a: bare 2-ampd-MOF (si_004 as is, expanded to P1)                              216 atoms
  b: a with one amine column replaced by the carbamate (+CO2) geometry of si_005 219 atoms
  c: a + linear CO2 at the pore center                                           219 atoms
  d: CO2 in a 15 Å box                                                             3 atoms
Run from the repo root:  python scripts/build_structures.py
"""
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

LIT = "structures/00_literature/"
OUT = "structures/01_built/"

def fragments(atoms, cutoff=1.7):
    """Find fragments connected by covalent bonds (<1.7 Å). A fragment containing N = diamine (+CO2)."""
    i, j = neighbor_list("ij", atoms, cutoff)
    n = len(atoms)
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    frags = {}
    for k, l in enumerate(lab):
        frags.setdefault(l, []).append(k)
    return [sorted(f) for f in frags.values() if any(atoms[k].symbol == "N" for k in f)]

def nearest_Mg(atoms, idx_list):
    """Which Mg column a fragment belongs to: index of the Mg closest to the fragment's N atoms."""
    Mg = [k for k in range(len(atoms)) if atoms[k].symbol == "Mg"]
    best = None
    for n in [k for k in idx_list if atoms[k].symbol == "N"]:
        d = atoms.get_distances(n, Mg, mic=True)
        if best is None or d.min() < best[0]:
            best = (d.min(), Mg[int(d.argmin())])
    return best[1]

# ---------- (a) ----------
a = read(LIT + "ja9b05567_si_004.cif")          # symmetry expansion → 216 atoms
a.set_pbc(True)
assert len(a) == 216, len(a)
write(OUT + "a_MOF.cif", a)

# ---------- (b) ----------
b6 = read(LIT + "ja9b05567_si_005.cif")         # 6 CO2 inserted, 234 atoms
b6.set_pbc(True)
fr_a = fragments(a)      # 6 fragments, 22 atoms each
fr_b = fragments(b6)     # 6 fragments, 25 atoms each (diamine 22 + CO2 3)
assert len(fr_a) == 6 and all(len(f) == 22 for f in fr_a), [len(f) for f in fr_a]
assert len(fr_b) == 6 and all(len(f) == 25 for f in fr_b), [len(f) for f in fr_b]

mg_a = [nearest_Mg(a, f) for f in fr_a]
mg_b = [nearest_Mg(b6, f) for f in fr_b]
assert sorted(mg_a) == sorted(mg_b), (mg_a, mg_b)

# Pick one column (lowest Mg index) and replace its diamine in a with the carbamate fragment from b6
col = min(mg_a)
frag_a = fr_a[mg_a.index(col)]
frag_b = fr_b[mg_b.index(col)]

b = a.copy()
del b[frag_a]                                   # remove that column's 22 diamine atoms from a → 194
frac = b6.get_scaled_positions()[frag_b]        # fractional coordinates of the b6 fragment
piece = Atoms([b6[k].symbol for k in frag_b], scaled_positions=frac, cell=a.cell, pbc=True)
b += piece                                      # place it directly in a's cell → 219
assert len(b) == 219, len(b)
write(OUT + "b_MOF_CO2_bound.cif", b)

# ---------- (c) ----------
# Pore center: the point in the ab plane farthest from all atoms (mid-cell along c)
c = a.copy()
best = None
for fx in np.linspace(0, 1, 121, endpoint=False):
    for fy in np.linspace(0, 1, 121, endpoint=False):
        p = a.cell.cartesian_positions([[fx, fy, 0.5]])[0]
        probe = a.copy(); probe += Atoms("X", positions=[p])
        d = probe.get_distances(len(probe) - 1, range(len(a)), mic=True).min()
        if best is None or d > best[0]:
            best = (d, p)
center = best[1]
co2 = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]])
co2.rotate(90, "y")                             # align the CO2 axis parallel to the c axis (pore direction)
co2.translate(center)
c += co2
assert len(c) == 219
write(OUT + "c_MOF_CO2_free.cif", c)

# ---------- (d) ----------
d = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]], cell=[15, 15, 15], pbc=True)
d.center()
write(OUT + "d_CO2_box.cif", d)

# ---------- Checks ----------
print("a:", len(a), a.get_chemical_formula())
print("b:", len(b), b.get_chemical_formula(), "| Mg index of replaced column:", col)
print("c:", len(c), c.get_chemical_formula(), "| nearest atom to pore center %.2f Å" % best[0])
print("d:", len(d), d.get_chemical_formula())
for name, at in [("a", a), ("b", b), ("c", c)]:
    i, j = neighbor_list("ij", at, 0.8)
    print(f"{name}: overlapping pairs within 0.8 Å = {len(i)//2}")
iC = len(c) - 3
Ns = [k for k in range(len(a)) if a[k].symbol == "N"]
print("c: CO2 C ↔ nearest N = %.2f Å (target ≥5)" % c.get_distances(iC, Ns, mic=True).min())