#!/usr/bin/env python

import onnx
import onnxruntime as rt
import uproot3
import argparse
import numpy as np

treename    = 'trainingtree'
branchname  = 'waveform'

print(f'''ONNX version: {onnx.__version__}''')
print(f'''ONNX runtime version: {rt.__version__}''')


#####
parser  = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",       type=str,               help="Input ROOT file",     default='')
parser.add_argument("-m", "--modelfile",    type=str,               help="Filename to load the model from", required=True)
parser.add_argument("-c", "--channel",  type=int,   help="Channel",             default=27)
parser.add_argument("-N", "--entries",  type=int,   help="Number of entries",   default=0)
parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
#####
args        = parser.parse_args()

infile      = args.infile
modelfile   = args.modelfile
entries     = args.entries
verbose     = args.verbose
channel     = args.channel
#####
print(f'''Input ROOT file: {infile}, model file: {modelfile}''')
if(infile==''):
    print('Please specify a valid input file name')
    exit(-1)

file    = uproot3.open(infile)

if verbose: print(f'''Opened file "{infile}", will use tree "{treename}"''')

dir     = file[treename]
branch  = dir[branchname]
Nentries = branch.numentries

N=Nentries if entries==0 else min(entries,Nentries)

X = branch.array()
dims = X.shape

if verbose:
    print(f'''Read an array: {dims}''')
    print(f'''Will process {N} entries out of total {Nentries}''')


sess = None

try: 
    sess = rt.InferenceSession(modelfile)
except:
    print(f'''Could not initiazlize a model from file {modelfile}, will exit now...''')
    exit(-1)



inputs = sess.get_inputs()
for x in inputs:
    print(x)


outputs = sess.get_outputs()
for x in outputs:
    print(x)

#wave=[[1554, 1558, 1555, 1564, 1558, 1555, 1556, 1554, 1750, 2284, 2424, 2116, 1838, 1713, 1649, 1613, 1601, 1589, 1583, 1578, 1572, 1574, 1573, 1569, 1567, 1562, 1563, 1560, 1561, 1557, 1557]]
#z = sess.run(["Identity:0"], {"dense_input:0": wave})
#print(f'''Benchmark output: {z}''')



for i in range(N): # loop over the data sample
    if (verbose and (i %100)==0 and i!=0): print(f'''Processed: {i}''')
    frame = X[i]
    wave = [np.array(frame[channel][0:31], dtype=np.float32)]

    # z = sess.run(["dense_1"], {"dense_input": wave})
    # print(wave)
    # wave=[[1554, 1558, 1555, 1564, 1558, 1555, 1556, 1554, 1750, 2284, 2424, 2116, 1838, 1713, 1649, 1613, 1601, 1589, 1583, 1578, 1572, 1574, 1573, 1569, 1567, 1562, 1563, 1560, 1561, 1557, 1557]]

    print(wave)

    z = sess.run(["Identity:0"], {"dense_input:0": wave})
    # z = sess.run(None, {"dense_input:0": wave.astype('float32')})
    print(f'''Benchmark output: {z}''')

exit(0)






z = sess.run(["dense_1"], {"dense_input": wave})

print(f'''Benchmark output: {z}''')




# wave=[[1554, 1558, 1555, 1564, 1558, 1555, 1556, 1554, 1750, 2284, 2424, 2116, 1838, 1713, 1649, 1613, 1601, 1589, 1583, 1578, 1572, 1574, 1573, 1569, 1567, 1562, 1563, 1560, 1561, 1557, 1557]]
