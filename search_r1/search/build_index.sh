
corpus_file=/workspace/Search-R1/search_r1/search/output.jsonl # jsonl
save_dir=/workspace/Search-R1/search_r1/search
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

CUDA_VISIBLE_DEVICES=0,1 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
