#!/bin/bash
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

# stage=0
stage=0

ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu="0" # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# E2E model related
affix=
train_set=train_100
valid_set=dev
test_set="test_clean test_other dev_clean dev_other"
checkpoint=checkpoint_best.pt

# LM related
lm_affix=
lm_checkpoint=checkpoint_best.pt
# lm_shallow_fusion=true # no LM fusion if false
lm_shallow_fusion=true # no LM fusion if false
sentencepiece_vocabsize=1000
# sentencepiece_vocabsize=100
# sentencepiece_vocabsize=29
sentencepiece_type=unigram

# data related
# dumpdir=data-100/dump   # directory to dump full features
dumpdir="/nfs/mercury-13/u123/dbagchi/espresso/examples/asr_librispeech/data-100/dump"   # directory to dump full features
download_dir="/nfs/mercury-13/u123/dbagchi/espresso/examples/asr_librispeech/data-100" # path to where you want to put the downloaded data; need to be specified if not on CLSP grid
data_dir="/nfs/mercury-13/u123/dbagchi/espresso/examples/asr_librispeech/data-100" # path to where you want to put the downloaded data; need to be specified if not on CLSP grid
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  data=/export/a15/vpanayotov/data
fi
data_url=www.openslr.org/resources/12
kaldi_scoring=true

# feature configuration
do_delta=false
apply_specaug=true


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

exp_dir="/nfs/mercury-13/u123/dbagchi/espresso/examples/asr_librispeech/exp-100"
lmdir=$exp_dir/lm_lstm${lm_affix:+_${lm_affix}}
dir=$exp_dir/lstm${affix:+_$affix}

if [ ${stage} -le 0 ]; then
   echo "Stage 0: Data Downloading"
   for part in dev-clean test-clean dev-other test-other train-clean-100; do
     local/download_and_untar.sh  --remove-archive $data_dir $data_url $part
   done
fi

if [ ${stage} -le 1 ]; then
  echo "Stage 1: Data Preparation"
  # for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $download_dir/LibriSpeech/$part $data_dir/$(echo $part | sed s/-/_/g)
  done
fi

train_feat_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${train_feat_dir}
valid_feat_dir=${dumpdir}/${valid_set}/delta${do_delta}; mkdir -p ${valid_feat_dir}
if [ ${stage} -le 2 ]; then
  echo "Stage 2: Feature Generation"
  fbankdir=fbank
  # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
  # for dataset in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
  for dataset in dev_clean test_clean dev_other test_other train_clean_100; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
      $data_dir/$dataset $exp_dir/make_fbank/$dataset ${fbankdir}
    utils/fix_data_dir.sh $data_dir/$dataset
  done

  # utils/combine_data.sh --extra-files utt2num_frames $data_dir/${train_set} $data_dir/train_clean_100 $data_dir/train_clean_360 $data_dir/train_other_500
  utils/combine_data.sh --extra-files utt2num_frames $data_dir/${train_set} $data_dir/train_clean_100 
  # utils/combine_data.sh --extra-files utt2num_frames $data_dir/${valid_set} $data_dir/dev_clean $data_dir/dev_other
  utils/combine_data.sh --extra-files utt2num_frames $data_dir/${valid_set} $data_dir/dev_clean

  # compute global CMVN - "potential issue here - mean variance normalization"
  compute-cmvn-stats scp:$data_dir/${train_set}/feats.scp $data_dir/${train_set}/cmvn.ark

  # dump features for training
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${train_feat_dir}/storage ]; then
    utils/create_split_dir.pl \
      /export/b1{4,5,6,7}/${USER}/fairseq-$data_dir/egs/asr_librispeech/dump/${train_set}/delta${do_delta}/storage \
      ${train_feat_dir}/storage
  fi
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${valid_feat_dir}/storage ]; then
    utils/create_split_dir.pl \
      /export/b1{4,5,6,7}/${USER}/fairseq-$data_dir/egs/asr_librispeech/dump/${valid_set}/delta${do_delta}/storage \
      ${valid_feat_dir}/storage
  fi
  dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
    $data_dir/${train_set}/feats.scp $data_dir/${train_set}/cmvn.ark $exp_dir/dump_feats/train ${train_feat_dir}
  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    $data_dir/${valid_set}/feats.scp $data_dir/${train_set}/cmvn.ark $exp_dir/dump_feats/valid ${valid_feat_dir}
  for dataset in $test_set; do
    test_feat_dir=${dumpdir}/$dataset/delta${do_delta}; mkdir -p ${test_feat_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
      $data_dir/$dataset/feats.scp $data_dir/${train_set}/cmvn.ark $exp_dir/dump_feats/$dataset ${test_feat_dir}
  done
fi

dict=$data_dir/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}_units.txt
sentencepiece_model=$data_dir/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}
lmdatadir=$data_dir/lm_text
if [ ${stage} -le 3 ]; then
  echo "Stage 3: Dictionary Preparation and Text Tokenization"
  mkdir -p $data_dir/lang
  cut -f 2- -d" " $data_dir/${train_set}/text > $data_dir/lang/input # Option 1: character level, Option 2: share data, create mixed model, Option 3: create 3 sentencepiece models
  echo "$0: training sentencepiece model..." # works on text side, takes transcripts and breaks into segments. segments are the outermost layer of model
  python3 ../../scripts/spm_train.py --bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2 --input=$data_dir/lang/input \
    --vocab_size=$((sentencepiece_vocabsize+3)) --character_coverage=1.0 \
    --model_type=$sentencepiece_type --model_prefix=$sentencepiece_model \
    --input_sentence_size=10000000
  echo "$0: making a dictionary and tokenizing text for train/valid/test set..."
  for dataset in $train_set $valid_set $test_set; do
    text=$data_dir/$dataset/text
    token_text=$data_dir/$dataset/token_text
    cut -f 2- -d" " $text | \
      python3 ../../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
      paste -d" " <(cut -f 1 -d" " $text) - > $token_text
    if [ "$dataset" == "$train_set" ]; then
      cut -f 2- -d" " $token_text | tr ' ' '\n' | sort | uniq -c | \
        awk '{print $2,$1}' | sort > $dict
      wc -l $dict
    fi
  done

 if $lm_shallow_fusion; then
  echo "$0: preparing text for subword LM..."
  mkdir -p $lmdatadir
  for dataset in $train_set $valid_set $test_set; do
    token_text=$data_dir/$dataset/token_text
    cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
  done
  if [ ! -e $lmdatadir/librispeech-lm-norm.txt.gz ]; then # considered public, AMI is private data
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $lmdatadir
  fi
  echo "$0: preparing extra corpus for subword LM training..."
  zcat $lmdatadir/librispeech-lm-norm.txt.gz | \
    python3 ../../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
    cat $lmdatadir/$train_set.tokens - > $lmdatadir/train.tokens
 fi
fi

lmdict=$dict
if $lm_shallow_fusion; then
 if [ ${stage} -le 4 ]; then
  echo "Stage 4: Text Binarization for subword LM Training"
  mkdir -p $lmdatadir/log #  unknown
  for dataset in $test_set; do test_paths="$test_paths $lmdatadir/$dataset.tokens"; done
  test_paths=$(echo $test_paths | awk '{$1=$1;print}' | tr ' ' ',')
  ${decode_cmd} $lmdatadir/log/preprocess.log \
    python3 ../../fairseq_cli/preprocess.py --task language_modeling_for_asr \
      --workers 50 --srcdict $lmdict --only-source \
      --trainpref $lmdatadir/train.tokens \
      --validpref $lmdatadir/$valid_set.tokens \
      --testpref $test_paths \
      --destdir $lmdatadir
 fi
fi

# Pause and review
# exit 1

[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
  echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;

if $lm_shallow_fusion; then
  if [ ${stage} -le 5 ]; then
    echo "Stage 5: subword LM Training"
    valid_subset=valid
    mkdir -p $lmdir/log
    log_file=$lmdir/log/train.log
    [ -f $lmdir/checkpoint_last.pt ] && log_file="-a $log_file"
    CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../fairseq_cli/train.py $lmdatadir --seed 1 \
      --task language_modeling_for_asr --dict $lmdict \
      --log-interval $((16000/ngpus)) --log-format simple \
      --num-workers 0 --max-tokens 32000 --batch-size 1024 --curriculum 1 \
      --valid-subset $valid_subset --batch-size-valid 1536 \
      --distributed-world-size $ngpus --distributed-port $(if [ $ngpus -gt 1 ]; then echo 100; else echo -1; fi) \
      --max-epoch 30 --optimizer adam --lr 0.001 --clip-norm 1.0 \
      --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
      --save-dir $lmdir --restore-file checkpoint_last.pt --save-interval-updates $((16000/ngpus)) \
      --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 \
      --arch lstm_lm_librispeech --criterion cross_entropy --sample-break-mode eos 2>&1 | tee $log_file
  fi
fi 


if $lm_shallow_fusion; then
  if [ ${stage} -le 6 ]; then
    echo "Stage 6: subword LM Evaluation"
    gen_set_array=(test)
    num=$(echo $test_set | awk '{print NF-1}')
    for i in $(seq $num); do gen_set_array[$i]="test$i"; done
    test_set_array=($test_set)
    for i in $(seq 0 $num); do
      log_file=$lmdir/log/evaluation_${test_set_array[$i]}.log
      python3 ../../fairseq_cli/eval_lm.py $lmdatadir --cpu \
        --task language_modeling_for_asr --dict $lmdict --gen-subset ${gen_set_array[$i]} \
        --max-tokens 40960 --max-sentences 1536 --sample-break-mode eos \
        --path $lmdir/$lm_checkpoint 2>&1 | tee $log_file
    done
  fi
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Dump Json Files"
  train_feat=$train_feat_dir/feats.scp
  train_token_text=$data_dir/$train_set/token_text
  train_utt2num_frames=$data_dir/$train_set/utt2num_frames
  valid_feat=$valid_feat_dir/feats.scp
  valid_token_text=$data_dir/$valid_set/token_text
  valid_utt2num_frames=$data_dir/$valid_set/utt2num_frames
  asr_prep_json.py --feat-files $train_feat --token-text-files $train_token_text --utt2num-frames-files $train_utt2num_frames --output $data_dir/train.json
  asr_prep_json.py --feat-files $valid_feat --token-text-files $valid_token_text --utt2num-frames-files $valid_utt2num_frames --output $data_dir/valid.json
  for dataset in $test_set; do
    feat=${dumpdir}/$dataset/delta${do_delta}/feats.scp
    token_text=$data_dir/$dataset/token_text
    utt2num_frames=$data_dir/$dataset/utt2num_frames
    asr_prep_json.py --feat-files $feat --token-text-files $token_text --utt2num-frames-files $utt2num_frames --output $data_dir/$dataset.json
  done
fi

if [ ${stage} -le 8 ]; then
  echo "Stage 8: Model Training"
  valid_subset=valid
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  opts=""
  update_freq=$(((2+ngpus-1)/ngpus))
  opts="$opts --arch speech_conv_lstm_librispeech"
  if $apply_specaug; then
    opts="$opts --max-epoch 95 --lr-scheduler tri_stage --warmup-steps $((2000/ngpus/update_freq)) --hold-steps $((600000/ngpus/update_freq)) --decay-steps $((1040000/ngpus/update_freq))"
    opts="$opts --encoder-rnn-layers 5"
    specaug_config="{'W': 80, 'F': 27, 'T': 100, 'num_freq_masks': 2, 'num_time_masks': 2, 'p': 1.0}"
  else
    opts="$opts --max-epoch 30 --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 --start-reduce-lr-epoch 10"
  fi
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py data-100 --task speech_recognition_espresso --seed 1 \
    --log-interval $((8000/ngpus/update_freq)) --log-format simple --print-training-sample-interval $((4000/ngpus/update_freq)) \
    --num-workers 0 --data-buffer-size 0 --max-tokens 26000 --batch-size 24 --curriculum 1 --empty-cache-freq 50 \
    --valid-subset $valid_subset --batch-size-valid 48 --ddp-backend no_c10d --decoder-embed-dim 320 --update-freq $update_freq \
    --distributed-world-size $ngpus --distributed-port $(if [ $ngpus -gt 1 ]; then echo 100; else echo -1; fi) \
    --optimizer adam --lr 0.001 --weight-decay 0.0 --clip-norm 2.0 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((6000/ngpus/update_freq)) \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 --best-checkpoint-metric wer \
    --criterion label_smoothed_cross_entropy_v2 --label-smoothing 0.1 --smoothing-type uniform \
    --scheduled-sampling-probs 1.0 --start-scheduled-sampling-epoch 1 \
    --dict $dict --bpe sentencepiece --sentencepiece-model ${sentencepiece_model}.model \
    --max-source-positions 9999 --max-target-positions 999 \
    $opts --specaugment-config "$specaug_config" 2>&1 | tee $log_file
fi


if [ ${stage} -le 9 ]; then
  echo "Stage 9: Decoding"
  opts=""
  path=$dir/$checkpoint
  decode_affix=
  if $lm_shallow_fusion; then
    opts="$opts --lm-path $lmdir/$lm_checkpoint"
    opts="$opts --lm-weight 0.47 --eos-factor 1.5"
    if $apply_specaug; then
      # overwrite the existing opts
      opts="$opts --lm-weight 0.4"
    fi
    decode_affix=shallow_fusion
  fi
  for dataset in $test_set; do
    decode_dir=$dir/decode_$dataset${decode_affix:+_${decode_affix}}
    CUDA_VISIBLE_DEVICES=$(echo $free_gpu | sed 's/,/ /g' | awk '{print $1}') speech_recognize.py data-100 \
      --task speech_recognition_espresso --user-dir espresso --max-tokens 15000 --batch-size 24 \
      --num-shards 1 --shard-id 0 --dict $dict --bpe sentencepiece --sentencepiece-model ${sentencepiece_model}.model \
      --gen-subset $dataset --max-source-positions 9999 --max-target-positions 999 \
      --path $path --beam 60 --max-len-a 0.08 --max-len-b 0 --lenpen 1.0 \
      --results-path $decode_dir $opts

    echo "log saved in ${decode_dir}/decode.log"
    if $kaldi_scoring; then
      echo "verify WER by scoring with Kaldi..."
      local/score_e2e.sh $data/$dataset $decode_dir
      cat ${decode_dir}/scoring_kaldi/wer
    fi
  done
fi

