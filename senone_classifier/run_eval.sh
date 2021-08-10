#!/bin/bash

model_path='exp/temp_2X200/final.pth.tar'
eval_scp_file_name='train_full_nodup_cv10/feats.scp'
eval_label_scp_file_name='train_full_nodup_cv10/phone.ctm2.scp' 
eval_scp_path='/nfs/raid83/u13/caml/users/pmuthuku_ad/CSL_exps/exps/senone_classifier/sotto-voce-corpus'

input_dim=13
output_dim=9096
fc_nodes=200
hidden_layers=2



python eval.py \
 --model_path $model_path \
 --eval_scp_file_name $eval_scp_file_name \
 --eval_scp_path $eval_scp_path \
 --eval_label_scp_file_name $eval_label_scp_file_name \
 --input_dim $input_dim \
 --output_dim $output_dim \
 --fc_nodes $fc_nodes \
 --hidden_layers $hidden_layers

