#!/bin/bash
#SBATCH --account=pc2-mitarbeiter
#SBATCH --job-name=fftw_openmp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH -t 1-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --switches=1

module reset
module load devel/CMake/3.20.1-GCCcore-10.3.0
module load toolchain/gompi/2021a
module load numlib/FFTW/3.3.9-gompi-2021a

export OMP_PLACES=cores     
export OMP_PROC_BIND=close

OUTDIR="../data/patient/mpi1nodes/result"
WISDIR="../wisdom/mpi1nodes"
ITER=100
BATCH=1
THREAD=38

echo "SLURM_JOB_ID : $SLURM_JOB_ID"
echo "N            : $@"
echo "Processes    : $SLURM_NTASKS"
echo "Threads      : ${THREAD}"
echo "Iter         : ${ITER}"
echo "Batch        : ${BATCH}"
echo ""
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
  echo "FFT Size : $arg $arg $arg"
  OUTFILE="${OUTDIR}/fftw_${arg}_p1_t${THREAD}_b${BATCH}.log"
  WISDOMFILE="${WISDIR}/fftw_${arg}_p1_t${THREAD}_b${BATCH}.patient"

  echo "Wisdom path: ${WISDOMFILE}"
  echo "Output path: ${OUTFILE}"

  #srun ../build/hybrid_many --num=$arg -i ${iter} -t ${thread} --batch=${batch} --wisdomfile=${wisdomfile} > ${outfile}
  mpirun -n 1 ../build/hybrid_many --num=$arg -i ${ITER} -t ${THREAD} --batch=${BATCH} --wisdomfile=${WISDOMFILE} > ${OUTFILE}
done
