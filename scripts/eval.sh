
GPU_NUM=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29572

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
DATA_PATH=path/to/train
EVAL_DATA_PATH=path/to/test

OUTPUT_PATH=./results/xxx
mkdir -p $OUTPUT_PATH

python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
    --model AIDE \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --batch_size 32 \
    --blr 5e-4 \
    --epochs 5 \
    --warmup_epochs 0 \
    --weight_decay 0 \
    --num_workers 2 \
    --reprob 0.25 \
    --smoothing 0.1 \
    --output_dir $OUTPUT_PATH \
    --resume $1 \
    --eval True \
2>&1 | tee -a $OUTPUT_PATH/log_test.txt
