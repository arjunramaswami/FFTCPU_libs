#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mpi_fftw
#SBATCH -p short
#SBATCH -t 29:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --switches=1

module reset
#module load devel/CMake
module load toolchain/gompi/2020a
module load numlib/FFTW/3.3.8-gompi-2020a

#module load intel
#module load numlib/FFTW

ctime=$(date "+%Y.%m.%d-%H.%M")
outdir="../data/estimate/mpi2nodes/"
iter=5

# Execute applications from 1 to 40 threads one after another
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
  echo "Executing FFT Size : $arg $arg $arg"
  outfile="${outdir}sp_${arg}_c25_${ctime}"
  echo "Writing to file : ${outfile}"

  srun ${SLURM_SUBMIT_DIR}/../build/fftw -n ${arg} -c 25 -i ${iter} -t 1 >> ${outfile}
done

