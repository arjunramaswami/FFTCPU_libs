#!/usr/bin/bash
fname=$1
gawk '
  {
    if($0 ~ /Median/)
      med=$3;
    if($0 ~ /Q1/)
      q1=$3
  }
  {
    if(med && q1)
      print $med $q1
  }' $fname