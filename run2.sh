for batchSize in 64 32 128; do
    for learnRate in 0.02 0.01 0.1 0.2 0.5; do
        for reLambda in  0.01 0 0.001 0.1; do
        	for seed in 0 1 2 3 4; do
            echo ${batchSize} ${learnRate} ${reLambda}
            #python mf.py ${batchSize} ${learnRate} ${reLambda} > log/mf_${batchSize}_${learnRate}_${reLambda}.log
            python biasMf.py ${batchSize} ${learnRate} ${reLambda} ${seed} > log/biasMf_${batchSize}_${learnRate}_${reLambda}_${seed}.log
            done
        done
    done
done

