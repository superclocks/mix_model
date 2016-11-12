./moe --input=../../data/WAP_xxl_PDPS000000057382.train --validation=../../data/WAP_xxl_PDPS000000057382.test --model=../../data/model.txt --k=13 --n=22000 --thread_num=4 --inter=2 --debug=0 --alpha=1.0 --beta=0.1 --l1=0.05 --l2=0.0
./moe --input=../../data/WAP_xxl_PDPS000000057382.test --model=../../data/model.txt --mode=1 --output=../../data/predict.txt
cat ../../data/predict.txt | sh ./auc.sh
