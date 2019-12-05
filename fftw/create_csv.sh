#!/bin/bash

## Takes the output file from FFTW execution and creates a csv file
## Input 1: filename to be converted to csv
## Input 2: name of the output csv file

inputname=$1
outputname=$2

echo "Parsing file $0 to $1"

cat ${inputname} | grep "FFT Size" | head -n 1 | tr -d "\t" | sed -e "s/ /,/g" > ${outputname}
cat ${inputname} | grep "fftw" | tr -d "\t" | sed -e "s/ /,/g" >> ${outputname}

echo "Completed"
