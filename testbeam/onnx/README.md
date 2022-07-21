# Application of ONNX to the EMCal signal frature extraction

## ONNX runtime binaries

To make things easier, there is a copy of prefab binaries,
complete with the original license which allows for this to happen,
in this folder.

## Compiling the code

```bash
# Example of the include and library path definitions
export CPLUS_INCLUDE_PATH=/home/maxim/onnxruntime-linux-x64-1.11.1/include/
export LD_LIBRARY_PATH=/home/maxim/onnxruntime-linux-x64-1.11.1/lib/

# If ROOT is used
ROOTSYS=/usr/local/root/
export LD_LIBRARY_PATH=/home/maxim/onnxruntime-linux-x64-1.11.1/lib/:$ROOTSYS/lib
export CPLUS_INCLUDE_PATH=/home/maxim/onnxruntime-linux-x64-1.11.1/include/:$ROOTSYS/include
# 

g++ onnxtest.C -L$LD_LIBRARY_PATH -lonnxruntime
```

## Prerequisites
For ROOT applications, it may be necessary to build ROOT.
Instructions can be found at https://root.cern/install/build_from_source/.

There may be extra requirements e.g. miscellaneous X11 elements, such as

```bash
sudo apt-get install libx11-dev libxpm-dev libxft-dev libxext-dev mesa-common-dev
cmake -DCMAKE_INSTALL_PREFIX=../install ../root_src
```

# ONNX Conversion

```bash
# Install the module (needs tensorflow installed, too)
pip install tf2onnx

# Convert an existing model
python -m tf2onnx.convert --saved-model ./16_ch27 --output tfmodel.onnx
```
