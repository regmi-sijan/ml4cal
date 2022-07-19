#!/bin/bash

# This is an interim utility script before we commit to a proper Makefile

export CPLUS_INCLUDE_PATH=./onnxruntime-linux-x64-1.11.1/include/
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.11.1/lib/

g++ onnxtest.C -L$LD_LIBRARY_PATH -lonnxruntime

