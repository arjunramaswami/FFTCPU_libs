#!/bin/bash

## Takes the output file from FFTW execution and creates a csv file
## Input 1: filename to be converted to csv
## Input 2: name of the output csv file

inputname=$1
outputname=$2

echo "Parsing file $1 to $2"

cat ${inputname} | grep "FFTSize" | head -n 1 | tr -s " " | sed -e "s/ /,/g" | sed "s/.$//" | sed -e "s/^,//g" > ${outputname}
cat ${inputname} | grep "fftw" | tr -s " " | sed -e "s/ /,/g" | sed "s/.$//" | sed -e "s/fftw:,//g" >> ${outputname}