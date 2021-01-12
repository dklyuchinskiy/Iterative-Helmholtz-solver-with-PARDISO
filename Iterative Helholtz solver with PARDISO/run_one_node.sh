#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# hyperthreading off
#SBATCH --threads-per-core=1
# set max wallclock time
#SBATCH --time=06:00:00
# set name of job
#SBATCH --job-name=nu4ppw10spg50
# set queue name
#SBATCH -p broadwell
#SBATCH --ntasks-per-node=1
# run the application

module purge
module load intel/2017.4.196 intel/2019/compilers

export KMP_AFFINITY=nowarnings,compact,1,0,granularity=fine,verbose
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

srun a.out
