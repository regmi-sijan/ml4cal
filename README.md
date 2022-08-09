# ML4CAL -- Machine Learning for sPHENIX calorimetry

## About

This repository contains code and other materials
related to applying _Machine Learning_ techniques
to signal feature extraction, for the sPHENIX calorimeters.

At present, we are using the following platforms:
* _Keras_ (which runs on top of TensorFlow) and work in the Python environment
* ONNX: https://onnx.ai/

In the latter case, a Keras model is converted into the `ONNX` format and used
with a variety of runtimes libraries, such as both C++ and Python versions of
the ONNX runtime, which is characterized by high performance. Work is under
way to develop this into a lightweight, standalone library which is easy
to integrate into sPHENIX and other software.

In addition, Keras inference was tested in service-type deployments
based on _nginx_ and _gunicorn_.

Training for signal feature extraction was performed with two
different methods
* Approximated version of the Landau function
* "Template fit" which is using a parametrized average shape of the pulse

## Data

The first stage of the study used simulated signals
with the shape approximated by the Landau function.
Then, the test beam data taken with the EMCAL prototype
in 2018 was used as input for the ML process.

## Content

Most of ML code is contained in the folder `testbeam` (for data analysis and model training),
and its subfolder `onnx`. Dependencies for the ONNX-based code are contained therein, so
it can be built locally.


