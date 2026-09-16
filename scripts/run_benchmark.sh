#!/bin/bash
# 범함수 × 구조 4개 scf. 이미 JOB DONE 이면 건너뜀.
cd ~/mof-co2-quantum
if [ "$(pgrep -c pw.x)" -gt 0 ]; then echo "pw.x 실행 중. 중단."; exit 1; fi
for func in pbe pbe-d3 vdw-df scan; do
  for name in a_MOF b_MOF_CO2_bound c_MOF_CO2_free d_CO2_box; do
    d=calc/benchmark/$func/$name
    if grep -q "JOB DONE" $d/espresso.pwo 2>/dev/null; then continue; fi
    python scripts/make_scf_input.py $func $name
    cd $d
    echo "=== $func/$name 시작: $(date)" >> ~/mof-co2-quantum/calc/benchmark/run.log
    np=24; [[ $name == d_* ]] && np=8
    mpirun -np $np pw.x -in espresso.pwi > espresso.pwo 2>&1
    echo "=== $func/$name 종료: $(date)  $(grep -c 'JOB DONE' espresso.pwo) done" >> ~/mof-co2-quantum/calc/benchmark/run.log
    cd ~/mof-co2-quantum
  done
done