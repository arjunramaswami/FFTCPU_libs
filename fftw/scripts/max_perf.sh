#!/bin/bash

## Consolidates Performance Data from different input files
##  Args: ./max_perf.sh <inp_fname1> <inp_fname2> <out_fname>
##   e.g. ./max_perf.sh data/fft_16_omp.csv data/fft_32_omp.csv perf.csv
##
## Outputs 2 files for runtime and throughput performance numbers respectively.
## These files are named runtime_<out_fname> and throughput_<out_fname>

# Number of command line arguments minus last input parameter to exclude output fname
num_inp_files=$(($#-1))

# Array of the inp files
array=${@:1:num_inp_files}

# Output fname, final cmd line arg
out_fname=${@: -1} 
runtime="runtime_"
throughput="throughput_"
out_runtime=${runtime}${out_fname}
out_throughput=${throughput}${out_fname}

printf "\n"
echo "Number of input files passed : ${num_inp_files}"
echo "Files: ${array}"

printf "\n"
echo "Parsing runtime results to ${out_runtime}"
echo "Parsing throughput results to ${out_throughput}"
printf "\n"

head -1 $1 | cut -f 3,4 -d "," > ${out_runtime}
head -1 $1 | cut -f 3,5 -d "," > ${out_throughput}

for arg in ${array}
do
    cut -f 3,4 -d "," ${arg} | sort -t "," -n -k 2 | head -2 | tail -1 >> ${out_runtime}
    cut -f 3,5 -d "," ${arg} | sort -t "," -r -n -k 2 | head -1 >> ${out_throughput} 
done
