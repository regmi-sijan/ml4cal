# Application of ONNX to the EMCal signal frature extraction

## ONNX runtime binaries

To make things easier, there is a copy of prefab binaries,
complete with the original license which allows for this to happen,
in this folder.

## Compiling the code

```bash
g++ onnxtest.C -L$LD_LIBRARY_PATH -lonnxruntime
```
