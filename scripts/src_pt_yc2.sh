#!/bin/bash
# Training
# source ~/.bashrc
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pdvc

GPU_ID=0
EXP_NAME=src_pt_yc2
args="--gpu_id ${GPU_ID} "
config_path=cfgs/${EXP_NAME}.yml
args+="--cfg_path ${config_path} "
python train.py ${args}
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.

# Evaluation
eval_json="data/yc2/captiondata/yc2_eval_wacv25.json"
eval_folder=${EXP_NAME} # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} \
                --eval_transformer_input_type queries \
                --eval_caption_file ${eval_json} \
                --gpu_id ${GPU_ID}
           