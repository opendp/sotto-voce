Step 1: Setup Pytorch-GPU environment.
Step 2: Download the training data, labels and mfcc features from the Github repo called sotto-voce corpus (https://github.com/opendifferentialprivacy/sotto-voce-corpus).
Step 3: Extract the downloaded data from above in the current directory.
Step 4: After extracting the libri_inp_data.tgz file, you will find two folders: train_full_nodup_tr90 and train_full_nodup_cv10. You need to change the path to the ark files in feats.scp and cmvn.scp (e.g. the line in feats.scp is "/d4m/ears/expts/46842-librispeech-test-10hrs/expts/gmm_data_preparation/mfcc_train/raw_fbank_pitch.ark.1". If you have extracted the mfcc features in <dir>, then replace the line to "<dir>/raw_fbank_pitch.ark.1").
Step 5: Do the same for the file "phone.ctm2.scp" in these folders as well.
Step 6: In the run_training.sh file, change the paths to train_scp_dir and cv_scp_dir accordingly.
Step 7: Change save_fld and mdl_path to the folder and filename of your saved LSTM model.
Step 8: Change the path to python to the one in your virtual environment by doing a "which python"
