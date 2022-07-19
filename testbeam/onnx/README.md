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

# 
g++ onnxtest.C -L$LD_LIBRARY_PATH -lonnxruntime
```
