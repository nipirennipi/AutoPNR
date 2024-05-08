# Data
NPRATIO=4
CTITLE_SIZE=20
HTITLE_SIZE=10
MAX_HIS_SIZE=50

# Prompt
PROMPT='<CANDIDATE> [MASK] [SEP] <USER>'

# Model
MODEL_NAME='bert-base-uncased'
PREFIX_SIZE=1
GENERATOR_HIDDEN_SIZE=512
CATE_DIM=32
GRU_NUM_LAYERS=1
TAU=1.0

# Train and test
SEED=23
VAL_RATIO=0.1
SEARCH_EPOCHS=1
TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=16
INFER_BATCH_SIZE=64
LEARNING_RATE_BERT=0.00001
LEARNING_RATE_PROMPT=0.0003
LEARNING_RATE_ANSWER=0.005
WEIGHT_DECAY=0.003

# Log tag
LOG_TAG=$(date '+%Y-%m-%d_%H:%M:%S')

programs=("search.py" "train.py" "test.py")

for program in "${programs[@]}"; do

    python "$program" \
        --seed $SEED \
        --log_tag "$LOG_TAG" \
        --npratio $NPRATIO \
        --ctitle_size $CTITLE_SIZE \
        --htitle_size $HTITLE_SIZE \
        --max_his_size $MAX_HIS_SIZE \
        --prompt "$PROMPT" \
        --model_name $MODEL_NAME \
        --prefix_size $PREFIX_SIZE \
        --cate_dim $CATE_DIM \
        --gru_num_layers $GRU_NUM_LAYERS \
        --tau $TAU \
        --val_ratio $VAL_RATIO \
        --search_epochs $SEARCH_EPOCHS \
        --train_epochs $TRAIN_EPOCHS \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --infer_batch_size $INFER_BATCH_SIZE \
        --learning_rate_bert $LEARNING_RATE_BERT \
        --learning_rate_prompt $LEARNING_RATE_PROMPT \
        --learning_rate_answer $LEARNING_RATE_ANSWER \
        --weight_decay $WEIGHT_DECAY \

    if [ $? -eq 0 ]; then
        if [ "$program" == "search.py" ]; then
            echo -e "\n \033[36m --- Search done, training! --- \033[0m \n"
        elif [ "$program" == "train.py" ]; then
            echo -e "\n \033[36m --- Training done, testing! --- \033[0m \n"
        fi
    else
        exit 1
    fi

done
