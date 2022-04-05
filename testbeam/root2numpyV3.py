#!/usr/bin/env python
'''
Root to numpy converter (from file to file)

Read the most recent version of ROOT files with EMCal testbeam.

The first index in the array is the 64 channel numbers
and the second is the time steps of the waveform.
Branch name is 'electron_adc_counts'.

The 32nd time bin of the waveform always contains -999 and is useless,
so by default we exclude it

Experimenting with uproot3:

import uproot3
import numpy as np

f = uproot3.recreate("moo.root")
f['test']=uproot3.newtree({'branch': "int32"})
f['test'].extend({'branch': np.array([1,2,3])})

'''
###################################
import uproot3
import numpy as np
import matplotlib.pyplot as plt
import argparse


###################################

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="Input ROOT file",     default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output numpy file",   default='')

parser.add_argument("-N", "--entries",  type=int,   help="Number of samples",   default=0)

parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
parser.add_argument("-z", "--zip",      action='store_true',    help="Store compressed")

parser.add_argument("-x", "--experimental",      action='store_true',    help="Experimental Mode")

###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile

entries     = args.entries
verbose     = args.verbose

xpr         = args.experimental


# Options for names in the tree:

treename    = 'T;1'
branchname  = 'electron_adc_counts'

if xpr:
    treename    = 'trainingtree'
    branchname  = 'waveform'

if(infile=='' or outfile==''):
    print('Please specify valid input and output file names')
    exit(-1)

file    = uproot3.open(infile)

if verbose: print(f'''Opened file {infile}, will use tree {treename}''')

dir     = file[treename]
branch  = dir[branchname]

Nentries = branch.numentries


N=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N} entries''')

X = branch.array()


if verbose : print(f'''Created an array: {X.shape}''')

dir.extend({'branch': np.array([1,2,3])})
exit(0)

f = uproot.recreate(outfile)

f['test']=uproot.newtree({'branch': np.array([1,2,3])})

# dir.extend({'branch': np.array([1,2,3])})

exit(0)

with open(outfile, 'wb') as f:
    if(args.zip):
        if verbose : print(f'''Saving to compressed file {outfile} ''')
        np.savez_compressed(f, X=X)
    else:
        if verbose : print(f'''Saving to uncompressed file {outfile} ''')
        np.save(f, X)

f.close()

exit(0)
