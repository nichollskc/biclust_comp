#!/bin/bash
set -x
RESULTS_DIR=$1
A_OUT=$2
B_OUT=$3
Z_OUT=$4
K_OUT=$5

B_TRANSPOSE="$3_transpose"

# Search for the directory that has been most recently modified
DIR=$(ls -td $RESULTS_DIR/*/ | head -n 1)
echo $DIR

# SDA outputs NxK individual score matrix as A - we call it A
#             KxL gene score matrix as X - we call it B
#             CxK context score matrix as B - we call it Z
cp $DIR/A $A_OUT
cp $DIR/X1 $B_TRANSPOSE
cp $DIR/B1 $Z_OUT

K=$(cat $DIR/X1 | wc -l)
echo $K > $K_OUT

# Transpose B so that it has dimensions LxK, to match other output
TRANPOSE_COMMAND="import sys; print('\n'.join(' '.join(c) for c in zip(*(l.split() for l in sys.stdin.readlines() if l.strip()))))"
python -c "$TRANPOSE_COMMAND" < $B_TRANSPOSE > $B_OUT
