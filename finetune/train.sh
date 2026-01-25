CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 finetune/train.py --config finetune/config/new_method/unform_stratified.json
