# Application of ONNX to the EMCal signal frature extraction

## ONNX

### Inspiration for the inference code

https://github.com/leimao/ONNX-Runtime-Inference/tree/main/src

The code by Lei Mao is evolving, an older version was used
as a template to create the starting point for the signal
extraction inference test.


### ONNX runtime binaries

#### Local build

To make things easier, there is a copy of prefab binaries,
complete with the original license which makes this procedure
proper, placed in this folder.

```bash
# Example of the include and library path definitions
export CPLUS_INCLUDE_PATH=./onnxruntime-linux-x64-1.11.1/include
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.11.1/lib
```

There is also a script `setup.sh` which can be sourced to set
this easily.

#### Build on SDCC machines

Please see `Makefile-sdcc`.

### Test applications

#### The code
The files `Makefile` and `Makefile-sdcc` can be used for building the test applications.
There are three:

* `onnxtest`: a simple "Hello World" which does single-input inference, w/o batching
* `onnxtestN`: evolution of the above, with batch inference for (much) better performance
* `onnxDriver`: a fully refactored version, with all of ONNX funcitonality wrapped in
a library (`onnxlib`). The library can be used to easily include ONNX into any application.

The `onnxDriver` can be used as an example of how to integrate ONNX into any C++
application in a simple manner.

An example of the `onnxDriver` run, in which data is read from a ROOT file in the "evaluation"
format, the model is read from the file `8_ch27.onxx` located in the local "models" folder -- can
be anywhere -- and the number of entries to run inference on is defined as 50:

```bash
./onnxDriver -r ~/data/evaluationtrees/8gev_2101.root -m models/8_ch27.onnx -N 50 -v -o
```

All of the applications are using the `lyra` library to parse the command line, making
the applications self-documented. For example, one can use the `help` option to
get information about the CLI:

```bash
onnxDriver --help
```

### Model Conversion from TensorFlow/Keras to the ONNX format

Note that ONNX Python packages are only compatible with Python versions 3.9
and lower.

```bash
# Install the module (needs tensorflow installed, too)
pip install tf2onnx

# Convert an existing model
python -m tf2onnx.convert --saved-model ./16_ch27 --output tfmodel.onnx
```

There are multiple CLI options that affect the behavior of the conversion module,
these may need to be studied carefully. Caveat -- the names of the NN layers do matter,
and they may be affected during the conversion process, so some debugging may be
in order.


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
