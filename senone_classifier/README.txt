Step 1: Setup Pytorch-GPU environment.
Step 2: Download the training data and labels from the Github repo called sotto-voce corpus.
Step 3: Extract the downloaded data from above in the current directory.
Step 4: In the run_training.sh file, change the paths to train_scp_dir, train_scp_file, train_label_scp_file accordingly. For example, the train_scp_file is "train_full_nodup_tr90/feats.scp", the train_label_scp_file is "train_full_nodup_tr90/phone.ctm2.scp" etc. The cv_scp_file and cv_label_scp_file can be similarly found in the folder suffixed "cv10".
Step 5: Change save_fld and mdl_path to the folder and filename of your saved LSTM model.
Step 6: Change the path to python to the one in your virtual environment by doing a "which python"
