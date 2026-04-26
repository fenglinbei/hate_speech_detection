CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 src/finetune/train.py --config finetune/config/new_method/unform_stratified.json
