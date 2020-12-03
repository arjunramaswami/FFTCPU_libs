#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mpi_fftw
#SBATCH -p batch
#SBATCH -t 11:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32
#SBATCH --switches=1
#SBATCH --verbose
#SBATCH --recommendations=on

module load devel/CMake
module load toolchain/gompi/2020a
module load numlib/FFTW/3.3.8-gompi-2020a

ctime=$(date "+%Y.%m.%d-%H.%M")
outdir="../data/patient/mpi16nodes/"
iter=20

make PATIENT=1 -C ../

# Execute applications with number of processes that divides 3d FFT size
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
  echo "Executing FFT Size : $arg $arg $arg"
  outfile="${outdir}sp_${arg}_${ctime}"
  echo "Writing to file : ${outfile}"

  srun ../bin/fftw -m $arg -n $arg -p $arg -i ${iter} -t 1 -s >> ${outfile}
done

