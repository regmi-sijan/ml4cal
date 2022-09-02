# Models

## The baseline model to study performance

`8_ch27.onnx` -- the 'canonical' model produced with data
from channel 27, 8GeV test beam data electron sample sample.

## The "global" model

`gl.onnx` -- this model was train on a merged data sample, in order to
cover a wider range of energies. The following files were merged to form
the training sample:

```
2gev_2059.root, 6gev_2088.root, 12gev_2121.root, 20gev_2148.root, 28gev_2161.root
```

Obviously, the weights here are completely ad hoc i.e. all equal 1 for each energy bin.

## The 20GeV model

`20_ch27.onnx` -- the model produced with data from channel 27, 20GeV test beam data
electron sample sample. Expected to provide a much wider dynamic range for training
than 8GeV data. Was demonstrated to be an improvement over the original 8GeV model.

 ## The high-energy mix model

`high.onnx` -- channel 27 again, based on files:
 
 ```
 16gev_2129.root, 20gev_2145.root, 28gev_2161.root
 ```

