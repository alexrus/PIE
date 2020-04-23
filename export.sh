export BERT_BASE_DIR='PIE_ckpt'
path_multitoken_inserts=pickles/conll/common_multitoken_inserts.p 
path_inserts=pickles/conll/common_inserts.p 

python3 word_edit_model.py \
    --data_dir=scratch \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/pie_model.ckpt \
    --output_dir=scratch \
    --max_seq_length=128 \
    --predict_batch_size=16 \
    --do_lower_case=False \
    --path_inserts=$path_inserts \
    --path_multitoken_inserts=$path_multitoken_inserts \
    --predict_checkpoint=PIE_ckpt/pie_model.ckpt \
    --use_tpu=False \
    --do_export=true \
    --export_dir=export