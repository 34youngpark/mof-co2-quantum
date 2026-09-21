#!/bin/bash
#SBATCH -p main -c 4 -t 06:00:00 -J mp2_b -o /home/seyoung/mp2_b.out --mem=40G
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mof
cd ~/mof-co2-quantum
export OMP_NUM_THREADS=4
python scripts/mp2_no.py --chk calc/pbc_hf/b_MOF_CO2_bound_gth-dzvp.chk --occ as_occ_b.npz \
    --cderi scratch/cderi_b.h5 --out as_no_b.npz --max-mem 35000