"""
structures/00_literature 의 Siegelman 2019 CIF 두 개에서 5.1단계 구조 4개를 만든다.
  a: 빈 2-ampd-MOF (si_004 그대로, P1로 전개)                        216원자
  b: a 에서 아민 컬럼 1개를 si_005의 카바메이트(+CO2) 기하로 교체     219원자
  c: a + 기공 중앙에 직선형 CO2                                        219원자
  d: 15 Å 박스 속 CO2                                                    3원자
실행: 레포 루트에서  python scripts/build_structures.py
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
    """공유결합(<1.7 Å)으로 이어진 조각들을 찾는다. N을 포함한 조각 = 다이아민(+CO2)."""
    i, j = neighbor_list("ij", atoms, cutoff)
    n = len(atoms)
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    frags = {}
    for k, l in enumerate(lab):
        frags.setdefault(l, []).append(k)
    return [sorted(f) for f in frags.values() if any(atoms[k].symbol == "N" for k in f)]

def nearest_Mg(atoms, idx_list):
    """조각이 어느 Mg 컬럼에 속하는지: 조각의 N에 가장 가까운 Mg 인덱스."""
    Mg = [k for k in range(len(atoms)) if atoms[k].symbol == "Mg"]
    best = None
    for n in [k for k in idx_list if atoms[k].symbol == "N"]:
        d = atoms.get_distances(n, Mg, mic=True)
        if best is None or d.min() < best[0]:
            best = (d.min(), Mg[int(d.argmin())])
    return best[1]

# ---------- (a) ----------
a = read(LIT + "ja9b05567_si_004.cif")          # 대칭 전개 → 216원자
a.set_pbc(True)
assert len(a) == 216, len(a)
write(OUT + "a_MOF.cif", a)

# ---------- (b) ----------
b6 = read(LIT + "ja9b05567_si_005.cif")         # CO2 6개 삽입, 234원자
b6.set_pbc(True)
fr_a = fragments(a)      # 6개, 각 22원자
fr_b = fragments(b6)     # 6개, 각 25원자 (다이아민 22 + CO2 3)
assert len(fr_a) == 6 and all(len(f) == 22 for f in fr_a), [len(f) for f in fr_a]
assert len(fr_b) == 6 and all(len(f) == 25 for f in fr_b), [len(f) for f in fr_b]

mg_a = [nearest_Mg(a, f) for f in fr_a]
mg_b = [nearest_Mg(b6, f) for f in fr_b]
assert sorted(mg_a) == sorted(mg_b), (mg_a, mg_b)

# 컬럼 하나(가장 낮은 Mg 인덱스)를 골라 a의 다이아민을 b6의 카바메이트 조각으로 교체
col = min(mg_a)
frag_a = fr_a[mg_a.index(col)]
frag_b = fr_b[mg_b.index(col)]

b = a.copy()
del b[frag_a]                                   # a에서 그 컬럼 다이아민 22원자 제거 → 194
frac = b6.get_scaled_positions()[frag_b]        # b6 조각의 분율 좌표
piece = Atoms([b6[k].symbol for k in frag_b], scaled_positions=frac, cell=a.cell, pbc=True)
b += piece                                      # a의 셀에 그대로 얹음 → 219
assert len(b) == 219, len(b)
write(OUT + "b_MOF_CO2_bound.cif", b)

# ---------- (c) ----------
# 기공 중심: ab 평면에서 모든 원자로부터 가장 먼 점 (c 방향은 셀 중간)
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
co2.rotate(90, "y")                             # CO2 축을 c축(기공 방향)에 평행하게
co2.translate(center)
c += co2
assert len(c) == 219
write(OUT + "c_MOF_CO2_free.cif", c)

# ---------- (d) ----------
d = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]], cell=[15, 15, 15], pbc=True)
d.center()
write(OUT + "d_CO2_box.cif", d)

# ---------- 검사 ----------
print("a:", len(a), a.get_chemical_formula())
print("b:", len(b), b.get_chemical_formula(), "| 교체한 컬럼 Mg 인덱스:", col)
print("c:", len(c), c.get_chemical_formula(), "| 기공 중심까지 최근접 원자 %.2f Å" % best[0])
print("d:", len(d), d.get_chemical_formula())
for name, at in [("a", a), ("b", b), ("c", c)]:
    i, j = neighbor_list("ij", at, 0.8)
    print(f"{name}: 0.8 Å 이내 겹친 쌍 = {len(i)//2}")
iC = len(c) - 3
Ns = [k for k in range(len(a)) if a[k].symbol == "N"]
print("c: CO2 C ↔ 가장 가까운 N = %.2f Å (≥5 목표)" % c.get_distances(iC, Ns, mic=True).min())