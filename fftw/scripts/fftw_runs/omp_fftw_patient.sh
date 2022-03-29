#!/bin/bash
#SBATCH --account=pc2-mitarbeiter
#SBATCH --job-name=fftw_openmp
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH -t 3-00:00:00

module reset
module load numlib/FFTW/3.3.9-gompi-2021a
module load toolchain/intel/2021a
module load devel/CMake/3.20.1-GCCcore-10.3.0

OUTDIR="../../data/patient/singlenode/wisonly"
WISDIR="../../wisdom/singlenode/all"
ITER=200
BATCH=1

echo "SLURM_JOB_ID : $SLURM_JOB_ID"
echo "Iter         : ${ITER}"
echo "Batch        : ${BATCH}"
echo ""
for arg in {403..450}
do
  echo "FFT Size      : $arg $arg $arg"
  for THREAD in {36..40}
  do
    echo "Thread       : ${THREAD}"
    mkdir -p ${OUTDIR}/${arg} ${WISDIR}/${arg}
    
    OUTFILE="${OUTDIR}/${arg}/fftw_${arg}_t${THREAD}_b${BATCH}.log"
    WISDOMFILE="${WISDIR}/${arg}/fftw_${arg}_t${THREAD}_b${BATCH}.patient"

    echo "Wisdom path    : ${WISDOMFILE}"
    echo "Output path    : ${OUTFILE}"

    ../../build/openmp_many --num=$arg -i ${ITER} -t ${THREAD} --expm=2 --batch=${BATCH} --wisdomfile=${WISDOMFILE} > ${OUTFILE}
  done
done
