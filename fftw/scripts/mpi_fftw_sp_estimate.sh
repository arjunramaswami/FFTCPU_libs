#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mpi_fftw
#SBATCH -p short
#SBATCH -t 29:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --switches=1

module load toolchain/gompi/2019b
module load numlib/FFTW/3.3.8-gompi-2019b

ctime=$(date "+%Y.%m.%d-%H.%M")
outdir="../data/estimate/mpi4nodes/"
iter=5

make VERBOSE=1 -C ../

# Execute applications from 1 to 40 threads one after another
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
  echo "Executing FFT Size : $arg $arg $arg"
  outfile="${outdir}sp_${arg}_${ctime}"
  echo "Writing to file : ${outfile}"

  srun ../bin/fftw -m $arg -n $arg -p $arg -i ${iter} -t 1 -s >> ${outfile}
done

