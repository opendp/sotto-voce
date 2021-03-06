#!/bin/bash

train_scp_dir="/nfs/mercury-13/u123/dbagchi/ASR_utt"
train_scp_file="train_full_nodup_tr90/feats.scp"
train_label_scp_file="train_full_nodup_tr90/phone.ctm2.scp"

cv_scp_dir="/nfs/mercury-13/u123/dbagchi/ASR_utt"
cv_scp_file="train_full_nodup_cv10/feats.scp"
cv_label_scp_file="train_full_nodup_cv10/phone.ctm2.scp"

input_dim=13
output_dim=9096
fc_nodes=200
hidden_layers=2


lr=0.0001
mom=0.0
ep=200
half_lr=0
early_stop=1

save_fld='exp/temp_2X200'
ckpt=0
cont_model=0
mdl_path="final.pth.tar"

pr_fr=1000

/nfs/mercury-13/u123/dbagchi/anaconda3/envs/pytorch_gpu_deblin/bin/python train.py --train_scp_dir $train_scp_dir \
 --train_scp_file $train_scp_file \
 --train_label_scp_file $train_label_scp_file \
 --cv_scp_dir $cv_scp_dir \
 --cv_scp_file $cv_scp_file \
 --cv_label_scp_file $cv_label_scp_file \
 --input_dim $input_dim \
 --output_dim $output_dim \
 --fc_nodes $fc_nodes \
 --hidden_layers $hidden_layers \
 --learn_rate $lr \
 --momentum $mom \
 --epochs $ep \
 --save_folder $save_fld \
 --checkpoint $ckpt  --model_path $mdl_path  --print_freq $pr_fr                                                                
