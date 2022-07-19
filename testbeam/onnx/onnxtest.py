#!/usr/bin/env python

import onnx
import onnxruntime as rt
print(f'''ONNX version: {onnx.__version__}''')
print(f'''ONNX runtime version: {rt.__version__}''')

sess = rt.InferenceSession("tfmodel.onnx")
wave=[[1554, 1558, 1555, 1564, 1558, 1555, 1556, 1554, 1750, 2284, 2424, 2116, 1838, 1713, 1649, 1613, 1601, 1589, 1583, 1578, 1572, 1574, 1573, 1569, 1567, 1562, 1563, 1560, 1561, 1557, 1557]]

z = sess.run(["dense_1"], {"dense_input": wave})

print(f'''Benchmark output: {z}''')
