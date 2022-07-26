# Application of ONNX to the EMCal signal frature extraction


## Setting up ROOT (if necessary)

### ROOT Prerequisites and build
For ROOT applications, it may be necessary to build ROOT.
Instructions can be found at https://root.cern/install/build_from_source/.

```bash
# There may be extra requirements e.g. miscellaneous X11 elements, such as
sudo apt-get install libx11-dev libxpm-dev libxft-dev libxext-dev mesa-common-dev
cmake -DCMAKE_INSTALL_PREFIX=../install ../root_src

# The actual build: -j 4 or something like that will work with "make" as the option
cmake --build . --target install [-- <options to the native tool>]
```

### Building ROOT-based applications

Building the standalone (not macro) ROOT app:

```bash
ROOTSYS=/usr/local/root/
g++ roottest.C $(root-config --glibs --cflags --libs) -o roottest

# Example of the command line option
./roottest -r myROOTfile.root

```

## ONNX

### ONNX runtime binaries

To make things easier, there is a copy of prefab binaries,
complete with the original license which allows for this to happen,
in this folder. The basic test of ONNX C++ runtime --  build procedure:

```bash
# Example of the include and library path definitions
export CPLUS_INCLUDE_PATH=./onnxruntime-linux-x64-1.11.1/include
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.11.1/lib
g++ onnxtest.C -L$LD_LIBRARY_PATH -lonnxruntime

# Building a helper library
g++ onnxlib.C -L$LD_LIBRARY_PATH -lonnxruntime -fPIC -shared -o onnxlib.so
```

### Running the test app

```bash
# One example:
 ./onnxtest -v -r  ~/data/evaluationtrees/8gev_2101.root -m ../ch27.onnx -N 50
```

### ONNX Conversion

```bash
# Install the module (needs tensorflow installed, too)
pip install tf2onnx

# Convert an existing model
python -m tf2onnx.convert --saved-model ./16_ch27 --output tfmodel.onnx
```
