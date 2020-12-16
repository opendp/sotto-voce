# Note: this requires the external drive /dev/sda mounted at /data
# (https://docs.microsoft.com/en-us/azure/virtual-machines/linux/attach-disk-portal)
# For Azure, I did the following:
#  1. sudo parted /dev/sda
#  2. mklabel gpt
#  3. mkpart xfspart
#  4. xfspart xfs
#  5. 0%
#  6. 100%
#  7. sudo mkfs.xfs /dev/sda1
#  8. sudo partprobe /dev/sda1
#  9. sudo mkdir /data
# 10. sudo chown azureuser:azureuser /data
# 11. sudo mount /dev/sda1 /data


echo "==========================="
echo "| Installing dependencies |"
echo "==========================="

# Install ZLib
if [ ! -d /usr/local/zlib ]; then
    git clone https://github.com/madler/zlib.git
    cd zlib/
    ./configure --prefix=/usr/local/zlib
    make test
    sudo make install
fi

# sudo apt update
sudo apt install -y make automake autoconf unzip g++ gfortran libtool subversion sox flac

compile_kaldi=1
if [ $compile_kaldi -eq 1 ]; then
    echo "===================="
    echo "| Installing Kaldi |"
    echo "===================="
    git clone https://github.com/kaldi-asr/kaldi.git
    KALDI=/data/kaldi
    cd $KALDI
    ./tools/extras/install_mkl.sh

    echo "==============================="
    echo "| Checking Kaldi Dependencies |"
    echo "==============================="
    ./tools/extras/check_dependencies.sh
    
    echo "=================="
    echo "| Building Kaldi |"
    echo "=================="
    cd $KALDI/tools
    make -j 6
    cd $KALDI/src
    make -j 6
    ./configure --shared
    make depend -j 6
    make -j 6
fi


echo "==============="
echo "| Conda setup |"
echo "==============="
conda create -y -n espresso-v4 python=3.7
conda activate espresso-v4
conda install -y pytorch=1.4.0 cudatoolkit=10.0 gcc_linux-64 gxx_linux-64 -c pytorch


echo "=================="
echo "| Espresso setup |"
echo "=================="
cd /data
mkdir espresso.v4
cd espresso.v4/
git clone https://github.com/freewym/espresso
cd espresso/
pip install --editable .
pip install sentencepiece
cd espresso/tools
make
pip install pychain
cd /data/espresso.v4/espresso/examples/asr_librispeech/
rm steps
ln -s /data/espresso.v4/espresso/espresso/tools/kaldi/egs/wsj/s5/steps
rm utils
ln -s /data/espresso.v4/espresso/espresso/tools/kaldi/egs/wsj/s5/utils/
# cd  ~/espresso.v4/espresso/examples/asr_librispeech/local/
cd /data/sotto-voce/local
rm data_prep.sh
ln -s /data/espresso.v4/espresso/espresso/tools/kaldi/egs/librispeech/s5/local/data_prep.sh
rm download_and_untar.sh
ln -s /data/espresso.v4/espresso/espresso/tools/kaldi/egs/librispeech/s5/local/download_and_untar.sh
if [ ! -d /data/espresso.v4/espresso/examples/asr_librispeech/data-100 ]; then
    mkdir /data/espresso.v4/espresso/examples/asr_librispeech/data-100
    mkdir /data/espresso.v4/espresso/examples/asr_librispeech/data-100/dev_clean
fi
