for batchSize in 128 64 32; do
    for learnRate in 0.05 0.1 0.01 0.5; do
    	echo "Train auto rec with "${batchSize}", "${learnRate}
        python autoRec.py ${batchSize} ${learnRate} > log/a_b_${batchSize}_${learnRate}.log
    done
done