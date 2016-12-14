./src/encdec-gpu \
--cnn-mem 6000m \
--path_dev_src=/home/lr/kamigaito/Corpora/kftt-data-1.0/data/tok/kyoto-dev.lc.sb.ja \
--path_dev_trg=/home/lr/kamigaito/Corpora/kftt-data-1.0/data/tok/kyoto-dev.lc.sb.en \
--path_dict_src=$HOME/tmp.d_src \
--path_dict_trg=$HOME/tmp.d_trg \
--path_model=$HOME/tmp.model \
--path_test_src=/home/lr/kamigaito/Corpora/kftt-data-1.0/data/tok/kyoto-test.lc.sb.ja \
--path_test_out=$HOME/tmp.test_out \
--path_train_src=/home/lr/kamigaito/Corpora/kftt-data-1.0/data/tok/kyoto-train.cln.lc.se.ja \
--path_train_trg=/home/lr/kamigaito/Corpora/kftt-data-1.0/data/tok/kyoto-train.cln.lc.se.en \
--dim-attention 256 \
--dim-hidden 256 \
--dim-input 256 \
--src-vocab-size 20000 \
--trg-vocab-size 20000 \
--batch-size 80 \
--parallel 20 \
--builder 0 \
--trainer 5 \
--depth-layer 1 \
--length-limit 100 \
--eta 0.001 \
--train 1
