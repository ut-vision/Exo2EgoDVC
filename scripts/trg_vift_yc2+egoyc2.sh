#!/bin/bash
# Training
# source ~/.bashrc
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pdvc

GPU_ID=0
EXP_NAME=trg_vift_yc2+egoyc2
# TODO: set your own pretrain path
PRE_PATH=save/trg_ft_egoyc2/model-best.pth

args="--gpu_id ${GPU_ID} "
config_path=cfgs/${EXP_NAME}.yml
args+="--cfg_path ${config_path} --pretrain_path ${PRE_PATH} "
python train_adapt.py ${args}
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.

# Evaluation
eval_json="data/egoyc2/captiondata/egoyc2_eval_wacv25.json"
eval_folder=${EXP_NAME}* # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} \
                --eval_transformer_input_type queries \
                --eval_caption_file ${eval_json} \
                --gpu_id ${GPU_ID}
