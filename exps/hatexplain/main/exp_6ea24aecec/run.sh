MODE=data \
TRAIN_LORA=1 \
TRAIN_BACKEND=deepspeed \
TRAIN_PROFILE=ds_zero2_safe \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/exps/run_one_exp.sh exps/hatexplain/main/exp_6ea24aecec