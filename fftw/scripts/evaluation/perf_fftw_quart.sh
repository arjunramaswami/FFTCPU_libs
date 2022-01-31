#!/bin/bash

## Find the median runtime between FFTW sizes $1 and $2 and store to file $3.
##   Along with that also bin the process that produced the perf
##   
## Example> ./perf_fftw.sh <starting_fft> <final_fft> <output_file>
##          ./perf_fftw.sh 16 671 perf_17.01.csv

start=$1
stop=$2
outputname=$3
ROOTDIR="../../data/patient/singlenode/wisonly"
OUTDIR="/scratch/pc2-mitarbeiter/arjunr/fft_cpu_ref/fftw/data/$3"

echo "Searching folders $1 until $2 (not including)"
echo "Output to: $OUTDIR"

echo ""
echo "Stepping into $ROOTDIR"
cd $ROOTDIR

# Associate array: (FFT Size: "Q1, Median, Q3")
declare -A quartileList

# Iterate through folders, each folder find the fastest median runtime
# Append the runtime to the QuartileList array
for ((iter=$start; iter<$stop; iter++))
do
  echo "Stepping into folder $iter"
  cd $iter || { echo "Error stepping into dir $iter"; exit 127; }

  # Iterating through all the files in a folder
  for file in *.log
  do
    # find and create a comma separated string of: Q1, Median, Q3 
    # take the first match because of several matching instances 
    temp="$(gawk '{if ($0 ~ /Q1/) pat1=$3; if ($0 ~ /Median/) pat2=$3 ; if($0 ~ /Q3  /) pat3=$3}{if (pat1 && pat2 && pat3) print pat1","pat2","pat3}' $file | head -n 1 | tee /dev/tty)"

    # create newline separated string of all the Q1, med, Q3 found within folder
    quartile="$quartile\n$temp"
  done

  # sort the runtimes found within the folder based on median runtime
  # sorting requires -k2 because first line is newline, -n for numeric
  # after sort, take the first two lines (first being newline) and then the second for the correct sorted (q1, med, q3) values
  echo "Output"
  quartileList[$iter]="$(echo -e $quartile | sort -k2 -n -t"," | head -n 2 | tail -1 | tee /dev/tty)"

  # reset quartile to empty string to not concatenate new values
  quartile=""
  cd .. 
done

# output associative array with best median runtimes to $OUTDIR
for ((iter=$start; iter<$stop; iter++))
do
  echo $iter,${quartileList[$iter]} >> $OUTDIR
done
