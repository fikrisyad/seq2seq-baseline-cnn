./src/encdec-cpu \
--cnn-mem 1800m \
--path_dev_src=$HOME/tmp.1000.en \
--path_dev_trg=$HOME/tmp.1000.en \
--path_dict_src=$HOME/tmp.d_src \
--path_dict_trg=$HOME/tmp.d_trg \
--path_model=$HOME/tmp.model \
--path_test_src=$HOME/tmp.1000.en \
--path_test_out=$HOME/tmp.test_out \
--path_train_src=$HOME/tmp.1000.en \
--path_train_trg=$HOME/tmp.1000.en \
--dim-attention 64 \
--dim-hidden 64 \
--dim-input 100 \
--batch-size 1 \
--builder 0 \
--depth-layer 1
