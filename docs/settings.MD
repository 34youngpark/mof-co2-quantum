# 5.1단계 계산 조건 (settings)

논문 Methods 초안용. 2026-09-17 기준.

## 소프트웨어
- Quantum ESPRESSO pw.x v7.5 (conda-forge 빌드)
- ASE (구조 생성·입력 작성), Python 3.11, conda 환경 `mof`
- 하드웨어: Intel Xeon Silver 4214R ×2 (물리 24코어), 125 GB RAM, MPI 24 프로세스 (CO₂ 박스는 8)

## 구조
- 출처: Siegelman et al., *J. Am. Chem. Soc.* 2019, 141, 13171 (DOI 10.1021/jacs.9b05567) SI
  - `ja9b05567_si_004.cif`: Mg₂(dobpdc)(2-ampd)₂, 216원자
  - `ja9b05567_si_005.cif`: Mg₂(dobpdc)(2-ampd–CO₂)₂, 234원자 (CO₂ 6개)
- 셀: a = b = 21.9537 Å, c = 7.0093 Å, α = β = 90°, γ = 120° (si_004 값, 모든 계산에서 고정)
- 대칭: P1로 전개 (CO₂ 1개/셀이라 P3₁21 대칭 깨짐)
- (a) 빈 MOF 216원자 = si_004
- (b) CO₂ 결합 MOF 219원자 = (a)에서 아민 컬럼 1개를 si_005의 카바메이트 조각(다이아민 22 + CO₂ 3)으로 분율좌표 교체. CO₂ loading 1/6
- (c) CO₂ 자유 MOF 219원자 = (a) + 직선형 CO₂를 기공 중심(모든 원자로부터 최원점, c축 평행)에 배치
- (d) CO₂ 단독 = 15 Å 정육면체 박스, Γ점
- 생성 스크립트: `scripts/build_structures.py`

## 슈도포텐셜
SSSP Efficiency v1.3 (PBE)

| 원소 | 파일 |
|---|---|
| Mg | Mg.pbe-n-kjpaw_psl.0.3.0.UPF |
| C | C.pbe-n-kjpaw_psl.1.0.0.UPF |
| H | H.pbe-rrkjus_psl.1.0.0.UPF |
| N | N.pbe-n-radius_5.UPF |
| O | O.pbe-n-kjpaw_psl.0.1.UPF |

## 공통 수치 설정
| 항목 | 값 |
|---|---|
| ecutwfc / ecutrho | 60 / 480 Ry (SSSP 권장 최댓값) |
| k-point | 1×1×2 (a, b, c), Γ (d) |
| nspin | 1 (closed shell) |
| occupations | fixed |
| conv_thr | 1×10⁻⁷ Ry |
| mixing_beta | 0.3 |
| nosym | true (single-point 계산; relax는 기본값) |

## 구조 최적화 (relax)
- 범함수: vdW-DF2 (`input_dft='vdw-df2'`)
- 원자 위치만 최적화, 셀 고정 (`calculation='relax'`, BFGS, 기본 수렴 기준 etot 1e-4 Ry, force 1e-3 Ry/Bohr)
- 스크립트: `scripts/make_qe_input.py`, `scripts/run_all_relax.sh`
- 결과

| 계 | BFGS 스텝 | 소요 | E (Ry) |
|---|---|---|---|
| (a) | 6 | 1 h 38 m | −2833.919300 |
| (b) | 92 (재시작 1회 포함, 총 96) | ~6 h | −2938.132172 |
| (c) | 13 | 3 h 17 m | −2938.087183 |
| (d) | 2 | 5 m | −104.164085 |

- 검증
  - (a) 문헌 VASP 구조 대비 원자 이동 최대 0.016 Å, 평균 0.005 Å → QE/SSSP 설정이 원논문(VASP, 1000 eV) 결과를 재현
  - (b) 카바메이트 C–N 1.392 Å, C–O 1.280/1.290 Å, O–Mg 2.07 Å, 원래 N–Mg 3.86 Å(끊어짐); 카바메이트 N에 H 1개, 이웃 피페리딘 N에 H 2개(암모늄) → Siegelman 메커니즘 재현
  - (c) CO₂ C–N 최단 7.93 Å 유지, C–O 1.173 Å (기체와 동일)
  - (d) C–O 1.173 Å

## 흡착에너지 정의
- Eq.1: ΔE₁ = E(b) − E(a) − E(d)
- Eq.2: ΔE₂ = E(b) − E(c)
- 단위: 1 Ry = 13.605693 eV, 1 eV = 96.485 kJ/mol
- 실험 기준값: ≈ −70 kJ/mol (−0.70 eV), Siegelman 2019
- 주의: 계산값은 0 K 전자 에너지. 영점진동·열 보정 미포함

## 범함수 벤치마크 (single-point, vdW-DF2 최적화 구조 고정)
| 범함수 | QE 설정 |
|---|---|
| vdW-DF2 | `input_dft='vdw-df2'` (relax 결과 사용) |
| PBE | `input_dft='pbe'` |
| PBE-D3 | `input_dft='pbe'`, `vdw_corr='dft-d3'` |
| vdW-DF | `input_dft='vdw-df'` |

- 제외: SCAN (conda QE 빌드에 libxc 없음), HSE06 (216원자 하이브리드는 워크스테이션에서 비현실적)
- 스크립트: `scripts/make_scf_input.py`, `scripts/run_benchmark.sh`, `scripts/make_benchmark_table.py`
- 결과: `results/dft_benchmark.csv`

| 범함수 | ΔE₁ (eV) | ΔE₂ (eV) | ΔE₁ (kJ/mol) | ΔE₁ − 실험 (kJ/mol) |
|---|---|---|---|---|
| vdW-DF2 | −0.664 | −0.612 | −64.0 | +3.5 |
| PBE | −0.698 | −0.687 | −67.3 | +0.2 |
| PBE-D3 | −0.973 | −0.914 | −93.9 | −26.3 |
| vdW-DF | −0.734 | −0.614 | −70.8 | −3.3 |
| 실험 | −0.70 | – | −67.5 | 0 |

- 해석
  - 범함수 간 편차 30 kJ/mol (화학적 정확도 4.2 kJ/mol의 7배)
  - PBE의 실험 일치는 우연: ΔE₁−ΔE₂(기공 물리흡착)가 0.01 eV로 vdW 계열(0.05~0.12 eV)보다 작아 분산 누락이 드러나며, 카바메이트 결합 과대평가와 상쇄된 결과
  - PBE-D3는 D3 후처리의 과대결합 경향
  - vdW-DF2는 열 보정 크기 안에서 실험과 일치 → 5.2단계 입력 구조로 채택

## 알려진 문제와 조치
- `nohup`만으로는 mpirun이 터미널 종료 시 죽음 → `setsid nohup … < /dev/null &` 사용
- single-point에서 (a),(c)가 `sym_rho_init_shell: lone vector`로 실패 → 대칭 사용 끔 (`nosym=.true.`)
- (b) relax가 BFGS 소보폭으로 92스텝 소요; 에너지 단조 감소, 이웃 다이아민(H32, H38, C138, C144)이 0.3~0.5 Å 재배치


## 5.1단계 범함수 벤치마크에 대한 메모
SCAN: 제외. conda-forge QE 빌드가 libxc 미포함 (pw.x 에러 "libxc needed for this functional", ldd 확인). 
벤치마크는 PBE, PBE-D3, vdW-DF, vdW-DF2 4종으로 확정.