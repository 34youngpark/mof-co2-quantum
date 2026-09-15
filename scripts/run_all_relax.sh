#!/bin/bash
# 사용: scripts/run_all_relax.sh a_MOF c_MOF_CO2_free   (인자 없으면 a b c 전부)
# 이미 JOB DONE 인 계산은 건너뛴다. pw.x 가 돌고 있으면 시작하지 않는다.
cd ~/mof-co2-quantum
if [ "$(pgrep -c pw.x)" -gt 0 ]; then echo "이미 pw.x 실행 중. 중단."; exit 1; fi
names=${@:-a_MOF b_MOF_CO2_bound c_MOF_CO2_free}
for name in $names; do
    d=calc/relax_vdwdf2/$name
    if grep -q "JOB DONE" $d/espresso.pwo 2>/dev/null; then echo "$name 완료됨, 건너뜀"; continue; fi
    python scripts/make_qe_input.py $name
    cd $d
    echo "=== $name 시작: $(date)" >> ../run_all.log
    mpirun -np 24 pw.x -in espresso.pwi > espresso.pwo 2>&1
    echo "=== $name 종료: $(date)  $(grep -c 'JOB DONE' espresso.pwo) done" >> ../run_all.log
    cd ~/mof-co2-quantum
done