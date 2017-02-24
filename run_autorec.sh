for batchSize in 512 256 128 64 32; do
    for learnRate in 0.03 0.02 0.01; do
    	echo "Train auto rec with "${batchSize}", "${learnRate}
        python autoRec.py ${batchSize} ${learnRate} 1 > log/auto_rec_${batchSize}_${learnRate}.log &
    done
done