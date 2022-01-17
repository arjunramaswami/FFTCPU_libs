#!/bin/bash

## Find the fastest runtime between FFTW sizes $1 and $2 and store to file $3.
##   Along with the fastest runtime also bin the process that produced the perf
##   
start=$1
stop=$2
outputname=$3
ROOTDIR="../../data/patient/singlenode/all"
OUTDIR="/scratch/pc2-mitarbeiter/arjunr/fft_cpu_ref/fftw/data/$3"

echo "Searching folders $1 to $2"
echo "Output to: $OUTDIR"

echo ""
echo "Stepping into $ROOTDIR"
cd $ROOTDIR
for ((iter=$start; iter<=$stop; iter++))
do
  echo "Stepping into folder $iter"
  cd $iter || { echo "Error stepping into dir $iter"; exit 127; }
  temp="$(grep -ri "Avg Tot Runtime" . | tee /dev/tty )"
  # cut deletes ./fftw_32_t40_b1.log:Avg Tot Runtime: 0.089789 ms to 0.089789 ms
  # sed removes the ms in 0.089789 ms
  # sort by lowest
  runtime="$(echo "$temp" | cut -d ":" -f "3" | sed -e "s/ ms//g" | sort -n)"
  fastest="$(echo "$runtime" | head -n 1 | xargs | tee /dev/tty)"
  echo "$iter,$fastest" >> $OUTDIR
  #processes="$(echo "$temp" | grep -Po "(?<=t)\d+")"
  #runtime="$(echo "$temp" | cut -d ":" -f "3" | sed -e "s/ ms//g")"
  #echo $temp
  cd ..
done
