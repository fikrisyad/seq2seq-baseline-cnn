#!/bin/bash

export PATH_TO_CNN=$PWD/cnn
export PATH_TO_EIGEN=$PWD/cnn/eigen
export PATH_TO_CUDA=/usr/local/cuda-7.5

cd $PWD/src

g++ \
-g \
-o \
encdec main.cc \
-I${PATH_TO_CNN} \
-I${PATH_TO_EIGEN} \
-I${PATH_TO_CUDA}/include \
-I${BOOST_ROOT}/include \
-std=c++11 \
-L/usr/lib \
-L${BOOST_ROOT}/lib \
-lboost_program_options \
-lboost_serialization \
-lboost_system \
-lboost_filesystem \
-L${PATH_TO_CNN}/build-cuda/cnn \
-lcnn \
-lcnncuda \
-DHAVE_CUDA \
-L${PATH_TO_CUDA}/targets/x86_64-linux/lib \
-lcudart \
-lcublas

