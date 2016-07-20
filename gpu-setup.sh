#!/bin/bash

export PATH_TO_CNN=$PWD/cnn
export PATH_TO_EIGEN=$PWD/eigen
export PATH_TO_CUDA=/usr/local/cuda-7.5

mkdir $PATH_TO_CNN/build-cuda
cd $PATH_TO_CNN/build-cuda
make clean
cmake .. -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DCUDA_TOOLKIT_ROOT_DIR=$PATH_TO_CUDA
make -j 4

