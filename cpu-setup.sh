#!/bin/bash

export PATH_TO_CNN=$PWD/cnn
export PATH_TO_EIGEN=$PWD/eigen
export PATH_TO_CUDA=/usr/local/cuda-7.5

mkdir $PATH_TO_CNN/build
cd $PATH_TO_CNN/build
make clean
cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN
make -j 4

