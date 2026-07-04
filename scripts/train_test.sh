

export NCCL_TIMEOUT=1800   # 增大超时时间至 30 分钟
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nproc_per_node=4 ./src/vlnce_src/train.py \
--run_type train \
--collect_type TF \
--name AerialVLN \
--batchSize 4 \
--dagger_it 1 \
--epochs 500 \
--lr 0.00025 \
--trainer_gpu_device 0 \
--use_llama_tokenizer \






