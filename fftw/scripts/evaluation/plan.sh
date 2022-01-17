#!/bin/bash

## Takes the output file from FFTW execution and aggregate the time to plan to create a csv file
## Input 1: filename to be converted to csv
## Input 2: name of the output csv file

inputname=$1
outputname=$2

echo "Threads,PlanTime(sec)" > $2
grep -ri "Time to plan:" $1 | sed -e "s/Threads //g;s/: time to plan -//g;s/ sec//g;s/ /,/g" >> $2