# ml4cal

## About

This repository contains code and other materials
related to applying _Machine Learning_ techniques
to signal feature extraction, for the sPHENIX calorimeters.

At present, we are using _Keras_ (which runs on top of TensorFlow)
and work entirely with Python.


Functionality has been tested with service-type deployments
based on _nginx_ and _gunicorn_.

## Data

The first stage of the study used simulated signals
with the shape approximated by the Landau function.
Then, the test beam data taken with the EMCAL prototype
in 2018 was used as input for the ML process.

