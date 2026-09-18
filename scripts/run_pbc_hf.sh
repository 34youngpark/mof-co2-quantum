#!/bin/bash
# 네 계의 주기 HF (gth-dzvp)를 차례로. 완료된 것은 건너뜀.
cd ~/mof-co2-quantum
export OMP_NUM_THREADS=24
for name in d_CO2_box a_MOF b_MOF_CO2_bound c_MOF_CO2_free; do
  if grep -q "^$name	gth-dzvp" calc/pbc_hf/summary.txt 2>/dev/null; then echo "$name 완료됨"; continue; fi
  python scripts/pbc_hf_mof.py $name gth-dzvp > calc/pbc_hf/${name}_gth-dzvp.out 2>&1
done
