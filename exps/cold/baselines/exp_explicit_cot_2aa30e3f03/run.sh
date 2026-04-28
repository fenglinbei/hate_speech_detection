MODE=full \
TRAIN_BACKEND=deepspeed \
TRAIN_PROFILE=ds_zero2_bs8 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_LORA=1 \
bash scripts/exps/run_one_exp.sh exps/cold/baselines/exp_explicit_cot_2aa30e3f03
