#!/bin/bash

model_path='exp/temp_2X200/final.pth.tar'
eval_scp_file_name='train_full_nodup_cv10/feats.scp' 
eval_scp_path='/nfs/mercury-13/u123/dbagchi/ASR_utt'
post_fname='posterior_hub5_2X200'

input_dim=13
output_dim=9096
fc_nodes=200
hidden_layers=2
#input_dim=440
#output_dim=2
#fc_nodes=1024
#fc_layers=1



/nfs/mercury-13/u123/dbagchi/anaconda3/envs/pytorch_gpu_deblin/bin/python eval.py --model_path $model_path \
 --eval_scp_file_name $eval_scp_file_name \
 --eval_scp_path $eval_scp_path \
 --post_fname $post_fname \
 --input_dim $input_dim \
 --output_dim $output_dim \
 --fc_nodes $fc_nodes \
 --hidden_layers $hidden_layers

