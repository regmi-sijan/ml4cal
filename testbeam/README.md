
# Caution

In the present version the length of the array is hardcoded to 31 for simplicity. __FIXME__.
Also, the converted/fit samples have 4 parameters attached (amplitude, time, pedestal, buzz),
but only 3 are used in the current state of the ML study.

# R2 map and channel map

```
[[  0   0   0   0   0   0   0   0]
 [  0   1   0   7   3   1   0   0]
 [  0   4  41 264  87   1   0   1]
 [  2  12 290 773 418  12   0   1]
 [  1  16 253 554 305  10   1   0]
 [  0   5  14  31  29   1   0   0]
 [  1   0   2   1   0   0   1   0]
 [  0   0   1   0   0   0   0   0]]


[[ 0.  1.  2.  3.  4.  5.  6.  7.]
 [ 8.  9. 10. 11. 12. 13. 14. 15.]
 [16. 17. 18. 19. 20. 21. 22. 23.]
 [24. 25. 26. 27. 28. 29. 30. 31.]
 [32. 33. 34. 35. 36. 37. 38. 39.]
 [40. 41. 42. 43. 44. 45. 46. 47.]
 [48. 49. 50. 51. 52. 53. 54. 55.]
 [56. 57. 58. 59. 60. 61. 62. 63.]]

```

# TF Debug Level

```bash
# Change TF log level to remove CUDA and other warnings:
export TF_CPP_MIN_LOG_LEVEL=3
```

# Provisional: C++ build quick tips

```bash
export CPLUS_INCLUDE_PATH=~/onnxruntime-linux-x64-1.11.1/include
export LD_LIBRARY_PATH=/home/maxim/onnxruntime-linux-x64-1.11.1/lib
g++ onnxtest.C -L$LD_LIBRARY_PATH -lonnxruntime
```
