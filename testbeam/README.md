# Contents

## Keras

This folder contains a number of Python scripts to process test beam data.
Interface with data stored in ROOT files is implemented using the ```uproot``
package. The following functionality is implemented:

* `root2numpyV3`: read the data, perform a fit, store the result in a numpy-formatted array;
the fit is currently based on the "template" method but any type of fit can be easily added,
it was Landau in the prior version; the template data is read from a CSV-formatted file
* `modelV3`: train a Keras model based on data from the previous stage, optionally
save the model in a file
* `validatorV3`: perform inference and do regression with respect to a control sample,
typically different from the training sample


All scripts are instrumented with extensive sets of command line options, which can
be examines using the `--help` option.

## ONNX

ONNX software is maintained in the subfolder `onnx` and documented in the README file
within.

# Misc

## Vector length

In a few places in the code the length of the input array (the waveform) is hardcoded
to 31 for simplicity, reflecting the layout of the test beam data.
This is easy to spot and is configurable in most cases.


## TF Debug Level

Keras/TF will complain about missing CUDA libraries, to avoid
this one can set the debug level to 3 (or alternatively install
CUDA if available)

```bash
# Change TF log level to remove CUDA and other warnings:
export TF_CPP_MIN_LOG_LEVEL=3
```


## R2 map and channel map

Below is the "R2" map the test beam data, i.e. statistics
for the fits that pass R2 selection. Basically just showing
that the "good data" are all in the center, as expected.

```
[[  0   0   0   0   0   0   0   0]
 [  0   1   0   7   3   1   0   0]
 [  0   4  41 264  87   1   0   1]
 [  2  12 290 773 418  12   0   1]
 [  1  16 253 554 305  10   1   0]
 [  0   5  14  31  29   1   0   0]
 [  1   0   2   1   0   0   1   0]
 [  0   0   1   0   0   0   0   0]]
```

Map of the serial channel numbers to their
position on the face of the calorimeter:

```
[[ 0.  1.  2.  3.  4.  5.  6.  7.]
 [ 8.  9. 10. 11. 12. 13. 14. 15.]
 [16. 17. 18. 19. 20. 21. 22. 23.]
 [24. 25. 26. 27. 28. 29. 30. 31.]
 [32. 33. 34. 35. 36. 37. 38. 39.]
 [40. 41. 42. 43. 44. 45. 46. 47.]
 [48. 49. 50. 51. 52. 53. 54. 55.]
 [56. 57. 58. 59. 60. 61. 62. 63.]]
```

The towers that are better populated are:

```
[[18, 19, 20]
 [26, 27, 28]
 [34, 35, 36]]
```
