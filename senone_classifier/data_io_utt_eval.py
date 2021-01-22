"""
Functions for dealing with kaldi input and output.

"""
import torch
import torch.utils.data as data
from utils import *
from model import *
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SenoneClassification(data.Dataset):
    """Features that can distinguish speech and non-speech"""
    def __init__(self, file_dir):
        self.scp_dir = file_dir['scp_dir']
        self.scp_file = file_dir['scp_file']
        
        self.utt_list = ins_utt_from_list(os.path.join(self.scp_dir, self.scp_file))
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        utt_name = self.utt_list[idx]
        utt_id, in_feats = read_kaldi_ark_from_scp(idx, os.path.join(self.scp_dir, self.scp_file))
        sample = {'name':utt_name, 'features': in_feats}
        
        return sample




if __name__ == "__main__":
    file_dir = {}
    file_dir['scp_dir'] = "/nfs/mercury-13/u123/dbagchi/SAD_utt"
    file_dir['scp_file'] = "feats_tr.scp"

    dataset =  SpeechActivityDetection(file_dir)
    dataloader = data.DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched['features'].size())


