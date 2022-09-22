#!/usr/bin/env python
'''

Interface to the MC data (photon/gamma)

'''


###################################
import uproot3
import numpy as np
from   numpy import loadtxt

import scipy
from   scipy.optimize import curve_fit

import argparse
###################################

parser  = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="Input ROOT file",     default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output numpy file",   default='')

parser.add_argument("-T", "--tmplfile", type=str,   help="Fit template file",   default='template.csv')

parser.add_argument("-N", "--entries",  type=int,   help="Number of entries",   default=0)
parser.add_argument("-c", "--channel",  type=int,   help="Channel",             default=0)


parser.add_argument("-t", "--threshold",type=float, help="threshold",           default=0.0)
parser.add_argument("-r", "--r2",       type=float, help="R2 threshold",        default=0.0)

parser.add_argument("-f", "--normfactor", type=float, help="Normalization factor",default=1.0)

parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
parser.add_argument("-z", "--zip",      action='store_true',    help="Store compressed")
parser.add_argument("-s", "--short",    action='store_true',    help="Shorten the waveform (downsample)")
parser.add_argument("-w", "--window",   action='store_true',    help="Narrow window the waveform")
parser.add_argument("-n", "--normalize",action='store_true',    help="Normalize input")
parser.add_argument("-p", "--peaktime", action='store_true',    help="Strict cut on peak time")

parser.add_argument("-d", "--debug",    action='store_true',    help="Debug mode")

###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile

tmplfile    = args.tmplfile

entries     = args.entries
verbose     = args.verbose

channel     = args.channel

normalize   = args.normalize
nrm         = args.normfactor

#####################################

np.set_printoptions(precision=3, linewidth=80)

if(infile==''):
    print('Please specify a valid input file name')
    exit(-1)


file    = uproot3.open(infile)


if verbose: print(f'''Opened the file "{infile}".''')



dir         = file['ttree']
p_branch    = dir['p']
Nentries    = p_branch.numentries

N=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N} entries out of total {Nentries}''')


N_branch        = dir["N"]
nlive_branch    = dir["nlive"]
p_branch        = dir["p"]
energy_branch   = dir["energy"]

N       = N_branch.array()
nlive   = nlive_branch.array()
p       = p_branch.array()
energy  = energy_branch.array()

dims = N.shape
if verbose : print(f'''Read an array: {dims}''')

print(energy.shape)

for i in range(20): # loop over the data sample
    ntowers = nlive[i]
    e = energy[i]

    en = 0.0
    for nt in range(ntowers):
        en+=e[nt]

    print(en)


exit(0)


####################################################
### -- attic --
# Well populated channels:
# selected = (18, 19, 20, 26, 27, 28, 34, 35, 36)
#
# print(np.reshape(output_array, (-1,34)))

# Some previous experimentation:
# f = uproot.recreate(outfile)
# f['test']=uproot.newtree({'branch': np.array([1,2,3])})
# dir.extend({'branch': np.array([1,2,3])})