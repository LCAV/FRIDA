#!/bin/bash

# 2016, Robin Scheibler
# This script was used to setup the environment for FRIDA computations
# on a Ubuntu Xenial machine

# Install anaconda locally
wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
bash Anaconda2-4.1.1-Linux-x86_64.sh -b -p /opt/anaconda2 -f
echo 'export PATH=/opt/anaconda2/bin:$PATH' >> .bashrc 
. .bashrc

# update conda
conda update -y conda

conda upgrade -y numpy scipy pandas
conda install -y ipython ipyparallel
conda install -y mkl accelerate iopro

apt-get install libsndfile1 libsndfile1-dev libsamplerate0 libsamplerate0-dev git tmux
pip install scikits.samplerate
pip install scikits.audiolab
pip install seaborn
pip install zmq
pip install joblib

pip install git+https://github.com/LCAV/pyroomacoustics
