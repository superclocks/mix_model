./fm --flagfile=params.conf --method=fma --normalization=1 --train=../../data/WAP_xxl_PDPS000000057382.train --test=../../data/WAP_xxl_PDPS000000057382.test --model=../../data/fm.model
./fm --nbit=16 --mode=1 --method=fma --normalization=1 --test=../../data/WAP_xxl_PDPS000000057382.test --model=../../data/fm.model --predict=../../data/fm.pred
cat ../../data/fm.pred | sh ./auc.sh
