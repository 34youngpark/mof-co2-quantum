#!/bin/bash
# a → b → c 순서로 vdW-DF2 relax를 차례로 실행. 레포 루트에서 실행.
cd ~/mof-co2-quantum
for name in a_MOF b_MOF_CO2_bound c_MOF_CO2_free; do
    python scripts/make_qe_input.py $name
    cd calc/relax_vdwdf2/$name
    echo "=== $name 시작: $(date)" >> ../run_all.log
    mpirun -np 24 pw.x -in espresso.pwi > espresso.pwo 2>&1
    echo "=== $name 종료: $(date)  $(grep -c 'JOB DONE' espresso.pwo) done" >> ../run_all.log
    cd ~/mof-co2-quantum
done