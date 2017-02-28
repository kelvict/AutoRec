#!/usr/bin/env bash
for batchSize in 512 256; do
    for learnRate in 0.002 0.001 0.0005; do
    	echo "Train auto rec with "${batchSize}", "${learnRate}
        python IAutoRec.py ${batchSize} ${learnRate} 0 > log/i_auto_rec_${batchSize}_${learnRate}.log &
    done
done