MODE=full \
TRAIN_BACKEND=deepspeed \
TRAIN_PROFILE=ds_zero2_bs4 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_LORA=1 \
bash scripts/exps/run_one_exp.sh exps/cold/baselines/exp_zero_shot_d55cfa6534