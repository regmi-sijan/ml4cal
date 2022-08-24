#!/usr/bin/env python
'''
Root to numpy converter (from file to file).

Read the most recent version of ROOT files with EMCal testbeam,
produced with "evaluation trees).

NB. The 32nd time bin of the waveform always contains -999 and is useless

Future work: experimenting with uproot3 (extending branches):

import uproot3
import numpy as np

f = uproot3.recreate("moo.root")
f['test']=uproot3.newtree({'branch': "int32"})
f['test'].extend({'branch': np.array([1,2,3])})

'''

t_offset    = 6.17742

template    = None
vec         = None


###
def tempfit(x, *par):
    w = x - par[1]
    return par[0]*np.interp(w, vec, template[:,1], left=0.0, right=0.0) + par[2]

###################################
import uproot3
import numpy as np
from   numpy import loadtxt

import scipy
from   scipy.optimize import curve_fit

import argparse
###################################
# Input normalization
norm    = np.array([4000, 16, 2000])

parser  = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="List of input ROOT files (comma separated)",     default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output numpy file",   default='')
parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
parser.add_argument("-z", "--zip",      action='store_true',    help="Store compressed")


###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile
verbose     = args.verbose

treename    = 'trainingtree'
branchname  = 'waveform'

#####################################

np.set_printoptions(precision=3, linewidth=80)

if(infile==''):
    print('Please specify a valid list of input files')
    exit(-1)

# Using shell to produce the list of comma-separated inputs:
# ls -m ~/data/evaluationtrees/8gev_2* | tr -d ' ' | tr -d '\n'

file_list = infile.split(',')
Nfiles = len(file_list)

if verbose:
    print("*** Verbose mode selected ***")
    print(f'''*** List of root files to open, total {Nfiles} ***''')
    for file_in_list in file_list: print(file_in_list)


arrays      = [None] * Nfiles
i           = 0
total       = 0

for file_in_list in file_list:
    file = uproot3.open(file_in_list)

    dir         = file[treename]
    branch      = dir[branchname]
    Nentries    = branch.numentries

    total+=Nentries
    arrays[i] = branch.array()
    i+=1

    if verbose : print(f'''*** Elements in array: {Nentries} ***''') 

    #dims = X.shape
    #if verbose : print(f'''Read an array: {dims}''')

    file.close()

if verbose: print(f'''*** Total {total} ***''')

all_data = np.concatenate(arrays, axis=0)
dims = all_data.shape
if verbose : print(f'''Combined array dimensions: {dims}''') 


if(outfile == ''): exit(0)

with open(outfile, 'wb') as f:
    if(args.zip):
        if verbose : print(f'''Saving to compressed file {outfile} ''')
        np.savez_compressed(f, all_data)
    else:
        if verbose : print(f'''Saving to uncompressed file {outfile} ''')
        np.save(f, all_data)

f.close()

exit(0)


