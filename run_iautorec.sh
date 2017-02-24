#!/usr/bin/env bash
for batchSize in 512 256 128 64 32; do
    for learnRate in 0.003 0.002 0.001; do
    	echo "Train auto rec with "${batchSize}", "${learnRate}
        python IAutoRec.py ${batchSize} ${learnRate} 1 > log/i_auto_rec_${batchSize}_${learnRate}.log &
    done
done