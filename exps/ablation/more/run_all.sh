# 直接跑 root 下所有 exp_*，会调用 run_one_exp.sh
MODE=infer \
TRAIN_BACKEND=deepspeed \
TRAIN_PROFILE=ds_zero3_safe \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_LORA=0 \
bash scripts/exps/run_all_exps.sh exps/ablation/more
