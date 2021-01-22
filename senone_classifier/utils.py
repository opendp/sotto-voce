
import torch
import os
import struct
import numpy as np
import kaldiio

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, a=0.02, b=0.03)
        torch.nn.init.uniform_(m.bias, a=0.02, b=0.03)

def ins_utt_from_list(scp_file):
    uttl = []
    with open(scp_file) as f:
        for line in f:
            utt_id, path_pos = line.replace("\n", "").split()
            uttl.append(utt_id)
    return uttl

def stack_frames(utt_feats, window):
    start_index = -1 * window
    last_idx = utt_feats.shape[0] 
    stacked_frames_utt = []

    for i in range(last_idx):
        stacked_frames = []
        for j in range(i-window, i+window+1):
            if (j<0):
                stacked_frames.append(utt_feats[0])
            elif (j>=last_idx):
                stacked_frames.append(utt_feats[-1])
            else:
                stacked_frames.append(utt_feats[j])
        stacked_frames_arr = np.asarray(stacked_frames)
        flattened_frms = stacked_frames_arr.flatten()
        stacked_frames_utt.append(flattened_frms)
    stacked_frames_utt_arr = np.asarray(stacked_frames_utt)

    return stacked_frames_utt_arr

def match_time(feats, labels):
    feat_shape = feats.shape[0]
    label_shape = labels.shape[0]
    min_shape = min(feat_shape, label_shape)
    new_feats = feats[:min_shape]
    new_labels = labels[:min_shape]
    return new_feats, new_labels

def create_norm_label_mat(label_mat):
    new_label_mat = np.zeros_like(label_mat)
    for i in range(label_mat.shape[0]):
        if label_mat[i] != 1:
           new_label_mat[i] = 1

    return new_label_mat

def read_kaldi_labels(utt_id, label_dir, hoptime=10):
    if os.path.exists(label_dir+"/"+utt_id):
        lab_file_read = open(label_dir+"/"+utt_id)
    else:
        print(utt_id+" does not exist")
        return None
    labels = list()

    for line in lab_file_read:
        st, end, label = line.split()
        labels.append((float(st), float(end), label))

    _, utt_end, _ = labels[-1]
    label_mat = np.zeros(int(utt_end*1000/hoptime))

    labelmap = {'speech':1, 'nonspeech':0}
    for label in labels:
        st, end, seg_label = label
        label_mat[int(st*1000/hoptime):int(end*1000/hoptime)] = labelmap[seg_label] 

    return label_mat

def read_kaldi_ark_from_scp(uid, scp_fn, ark_base_dir=""):
    """
    Read a Kaldi scp file and return a Numpy matrix.
    Parameters
    ----------
    ark_base_dir : str
    The base directory for the archives to which the SCP points.
    """

    totframes = 0
    lines = 0
    utt_mat = None
    with open(scp_fn) as f:
        for line in f:
            lines = lines + 1
            if lines <= uid:
                continue
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split()
            utt_mat = kaldiio.load_mat(path_pos)
            #print (utt_id+" ")
            #print(utt_mat.shape)
            return utt_id, utt_mat
