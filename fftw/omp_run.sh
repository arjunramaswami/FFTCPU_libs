#!/bin/bash

./host_omp -m 32 -n 32 -p 32 -i 100 -t 1
./host_omp -m 32 -n 32 -p 32 -i 100 -t 2
./host_omp -m 32 -n 32 -p 32 -i 100 -t 5
./host_omp -m 32 -n 32 -p 32 -i 100 -t 10
./host_omp -m 32 -n 32 -p 32 -i 100 -t 20
./host_omp -m 32 -n 32 -p 32 -i 100 -t 40
