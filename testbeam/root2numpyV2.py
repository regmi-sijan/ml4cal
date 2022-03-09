#!/usr/bin/env python
'''
Root to numpy converter (from file to file)

Read the most recent version of ROOT files with EMCal testbeam.

The first index in the array is the 64 channel numbers
and the second is the time steps of the waveform.
Branch name is 'electron_adc_counts'.

The 32nd time bin of the waveform always contains -999 and is useless,
so by default we exclude it


'''
###################################
import uproot
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


###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile

entries     = args.entries
verbose     = args.verbose

if(infile=='' or outfile==''):
    print('Please specify valid input and output file names')
    exit(-1)

file    = uproot.open(infile)
dir     = file['T;1']

Nentries = dir['electron_adc_counts'].num_entries
N=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N} entries''')

branch = dir['electron_adc_counts']
X = branch.array(library='np', entry_stop=N)

if verbose : print(f'''Created an array: {X.shape}''')

with open(outfile, 'wb') as f:
    if(args.zip):
        if verbose : print(f'''Saving to compressed file {outfile} ''')
        np.savez_compressed(f, X=X)
    else:
        if verbose : print(f'''Saving to uncompressed file {outfile} ''')
        np.save(f, X)

f.close()

exit(0)
