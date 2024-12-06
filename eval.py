from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import pprint
import torch
import numpy as np
import time
import glob
from os.path import dirname, abspath

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)

from eval_utils import evaluate
from pdvc.pdvc import build
from misc.utils import create_logger
from data.video_dataset import PropSeqDataset, collate_fn
from torch.utils.data import DataLoader
from os.path import basename
import pandas as pd

def create_fake_test_caption_file(metadata_csv_path):
    out = {}
    df = pd.read_csv(metadata_csv_path)
    for i, row in df.iterrows():
        out[basename(row['filename']).split('.')[0]] = {'duration': row['video-duration'], "timestamps": [[0, 0.5]], "sentences":["None"]}
    fake_test_json = '.fake_test_json.tmp'
    json.dump(out, open(fake_test_json, 'w'))
    return fake_test_json

def main(opt):
    if opt.eval_folder.strip("./").startswith(opt.eval_save_dir):
        folder_path = opt.eval_folder
    else:
        folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    if "*" in folder_path:
        folder_path_list = glob.glob(folder_path)
    else:
        folder_path_list = [folder_path]    
    json_name = os.path.basename(opt.eval_caption_file).split(".")[0]
    save_log_filename = f"eval_{json_name}.log"
    if opt.eval_transformer_input_type is not None:
        if "gt" in opt.eval_transformer_input_type:
            save_log_filename = f"eval_{json_name}_gt_prop.log"
    for folder_path in folder_path_list:
        if not opt.re_eval and os.path.exists(os.path.join(folder_path, save_log_filename)):
            print("skip: eval file exists", folder_path)
            continue
        
        logger = create_logger(folder_path, save_log_filename)
        logger.info(f"logger created in {folder_path}/{save_log_filename}")
        if opt.eval_model_path:
            model_path = opt.eval_model_path
            infos_path = os.path.join('/'.join(opt.eval_model_path.split('/')[:-1]), 'info.json')
        else:
            model_path = os.path.join(folder_path, 'model-best.pth')
            infos_path = os.path.join(folder_path, 'info.json')

        logger.info(vars(opt))

        with open(infos_path, 'rb') as f:
            logger.info('load info from {}'.format(infos_path))
            old_opt = json.load(f)['best']['opt']

        for k, v in old_opt.items():
            if k[:4] != 'eval':
                vars(opt).update({k: v})

        if opt.eval_transformer_input_type is not None:
            opt.transformer_input_type = opt.eval_transformer_input_type

        if not torch.cuda.is_available():
            opt.nthreads = 0
        # Create the Data Loader instance

        if opt.eval_video_feature_folder is not None:
            opt.visual_feature_folder = opt.eval_video_feature_folder
        if opt.eval_visual_feature_type is not None:
            opt.visual_feature_type = opt.eval_visual_feature_type
        
        opt.gt_file_for_eval = [opt.eval_caption_file] 
        opt.gt_file_for_para_eval = [opt.eval_caption_file.replace("captiondata/", "captiondata/para/para_")]

        val_dataset = PropSeqDataset(opt.eval_caption_file, opt.visual_feature_folder,
                                    opt.dict_file, False, opt.eval_proposal_type, opt)
        loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)


        model, criterion, postprocessors = build(opt)
        model.translator = val_dataset.translator
        
        while not os.path.exists(model_path):
            raise AssertionError('File {} does not exist'.format(model_path))

        logger.debug('Loading model from {}'.format(model_path))
        loaded_pth = torch.load(model_path, map_location=opt.eval_device)
        epoch = loaded_pth['epoch']

        # loaded_pth = transfer(model, loaded_pth, model_path+'.transfer.pth')
        model.load_state_dict(loaded_pth['model'], strict=True)
        model.eval()

        model.to(opt.eval_device)

  
        date = time.strftime("%y%m%d-%H%M%S_", time.localtime())

        out_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_alpha{}.json'.format(date + str(opt.id), epoch, len(loader.dataset), opt.ec_alpha))
        if opt.eval_transformer_input_type is not None:
            if "gt" in opt.eval_transformer_input_type:
                out_json_path = os.path.join(folder_path, '{}_gt_prop_epoch{}_num{}_alpha{}.json'.format(date + str(opt.id), epoch, len(loader.dataset), opt.ec_alpha))                
        video_wise_eval_path = os.path.join(folder_path, '{}_video_wise_eval.txt'.format(date + str(opt.id)))
        caption_scores, eval_loss = evaluate(model, criterion, postprocessors, loader, out_json_path,
                        logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, 
                        device=opt.eval_device, debug=False, skip_lang_eval=False, save_feat=opt.eval_save_feat)
        avg_eval_score = {key: np.round(np.array(value).mean() * 100, 2) for key, value in caption_scores.items() if key !='tiou' and not key.startswith('para')}
        # avg_eval_score = {key: np.round(np.array(value).mean() * 100, 2) for key, value in caption_scores.items()}
        formatted_avg_eval_score = pprint.pformat(avg_eval_score)
        logger.info('\nValidation result based on all {} val videos:\n {}\n avg_score:\n{}'.format(len(loader.dataset), caption_scores.items(), formatted_avg_eval_score))

        logger.info('saving reults json to {}'.format(out_json_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_mode', type=str, default='eval', choices=['eval', 'test'])
    #parser.add_argument('--model_type', default='pdvc', choices=['pdvc', 'pdvc-view'], help='model type')
    parser.add_argument('--eval_video_feature_folder', type=str, nargs='+', default=None)
    parser.add_argument('--eval_visual_feature_type', type=str, nargs='+', default=None)
    parser.add_argument('--eval_video_meta_data_csv_path', type=str, default=None)
    parser.add_argument('--eval_folder', type=str, required=True)
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_tool_version', type=str, default='2018', choices=['2018', '2021'])
    parser.add_argument('--eval_caption_file', type=str, default='data/anet/captiondata/val_1.json')
    parser.add_argument('--eval_proposal_type', type=str, default='gt')
    parser.add_argument('--eval_transformer_input_type', type=str, default=None, choices=['gt_proposals', 'queries'])
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--re_eval', action='store_true')
    parser.add_argument('--eval_save_feat', action='store_true')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if True:
        torch.backends.cudnn.enabled = False
    main(opt)
