#!/bin/bash 
#source_file= "$1"
totlines="$(sed -n '$=' $1)"
lines=$((totlines - 1))
n_files=$2
#echo $n_files
#echo $3
#shuffle and save to temp file
temp_file="tempfile.csv"
tail -n +2 $1 | shuf > $3$temp_file
#increment over file and print to new file
incr=$((lines / n_files))
intsequence="$(seq 1 $incr $totlines)"
sequence=($intsequence)
#echo ${sequence[*]}
#echo ${sequence[0]}
len=${#sequence[*]}
file_postfix="split_file.csv"
#for i in $(seq 0 $((len - 2)))
#do
#    linestart=${sequence[$i]}
#    echo $linestart
#    lineend=$((${sequence[$i + 1]} - 1))
#    echo $lineend
#    nextline=$((lineend + 1))
#    sed -n "${linestart},${lineend}p;${nextline}q" $3$temp_file > $3$i$file_postfix
#done

