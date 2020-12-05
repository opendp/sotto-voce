MAIN_ROOT=/data
export KALDI_ROOT=$MAIN_ROOT/kaldi

# BEGIN from kaldi path.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# END

export PATH=/home/azureuser/anaconda3/bin:$PATH
export PATH=$MAIN_ROOT:$MAIN_ROOT/espresso.v4/espresso:$MAIN_ROOT/espresso.v4/espresso/espresso/tools/:$PATH
export PYTHONPATH=$MAIN_ROOT:$MAIN_ROOT/espresso:$MAIN_ROOT/espresso/tools:/home/azureuser/anaconda3/envs/espresso-v4/bin/python:$PYTHONPATH
export PYTHONUNBUFFERED=1
