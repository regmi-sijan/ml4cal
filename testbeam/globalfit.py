#!/usr/bin/env python
##########################################################
# Read an argitrary (potenially large) array of testbeam
# data previously converted to numpy, fit it, augment and
# save in a new file
##########################################################

import argparse
from concurrent.futures.process import _chain_from_iterable_of_lists
import numpy as np
from   numpy import loadtxt

import scipy
from   scipy.optimize import curve_fit

import argparse
#################################################

t_offset    = 6.17742

template    = None
vec         = None


###
def tempfit(x, *par):
    w = x - par[1]
    return par[0]*np.interp(w, vec, template[:,1], left=0.0, right=0.0) + par[2]

###

parser  = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="Input file",     default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output file",   default='')

parser.add_argument("-c", "--channels", type=str,   help="List of channels", default='')

parser.add_argument("-T", "--tmplfile", type=str,   help="Fit template file",   default='template.csv')
parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile
verbose     = args.verbose

channels    = args.channels.split(',')

print(infile, outfile, channels)

if verbose: print(f'''Will attempt to use the template file "{args.tmplfile}".''')

try:
    template = loadtxt(args.tmplfile, delimiter=',')
except:
    print("Problem with reading template file, exiting")
    exit(-1)

with open(infile, 'rb') as f: dataset = np.load(f)
if verbose: print(f'''Read the input array: {dataset.shape}''')

N = dataset.shape[0]

cut_dataset = np.delete(dataset, 31, 2)
if verbose: print(f'''Truncated input array: {cut_dataset.shape}''')

fit_array = np.zeros((N, 3))

if verbose: print(f'''Created the output array: {fit_array.shape}''')

x = np.linspace(0, 31, 31, endpoint=False)
cnt_bad, cnt_out, cnt_small, first, output_array = 0, 0, 0, True, None

# channels = [26,27]

for i in range(5): # loop over the data sample
    if (verbose and (i %100)==0 and i!=0): print(f'''Processed: {i}  Percentage bad: {float(cnt_bad)/float(i)}''')

    frame   = cut_dataset[i]                  # select a row
    for channel in channels:
        wave    = frame[int(channel)][0:31]  # select waveform, 31 bin
        print(wave)