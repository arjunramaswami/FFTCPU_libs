#!/bin/bash
#SBATCH --account=pc2-mitarbeiter
#SBATCH --job-name=fftw_openmp
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH -t 2-00:00:00

module reset
#module load devel/CMake/3.20.1-GCCcore-10.3.0
module load toolchain/intel/2021a

ITER=200
BATCH=1
OUTDIR="../../data/singlenodeicc"

echo "SLURM_JOB_ID : $SLURM_JOB_ID"
echo "Iter         : ${ITER}"
echo "Batch        : ${BATCH}"
echo ""
for arg in {16..127}
do
  echo "FFT Size      : $arg $arg $arg"
  for THREAD in {20..40}
  do
    echo "Thread       : ${THREAD}"
    mkdir -p ${OUTDIR}/${arg}
    
    OUTFILE="${OUTDIR}/${arg}/mkl_${arg}_t${THREAD}_b${BATCH}.log"
    echo "Output path    : ${OUTFILE}"

    KMP_AFFINITY=granularity=core,compact ../../iccbuild/mkl_openmp_many --num=$arg -i ${ITER} -t ${THREAD} --expm=2 --batch=${BATCH} > ${OUTFILE}
  done
done
