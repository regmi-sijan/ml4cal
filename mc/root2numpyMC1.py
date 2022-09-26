#!/usr/bin/env python
'''

Interface to the MC data (photon/gamma)

'''


###################################
import uproot3
import numpy as np
from   numpy import loadtxt

import math

import argparse
###################################

Neta    = 96
Nphi    = 256
shape   = (Neta, Nphi)

Ntotal = shape[0]*shape[1]

tower_map = np.zeros((Ntotal,2), dtype=int)


for n in range(Ntotal):
    eta = n // Nphi
    phi = n %  Nphi
    tower_map[n] =  [eta, phi]

###
def pseudorap(p):
    pTot = math.sqrt(p[0]**2+p[1]**2+p[2]**2)
    return math.atanh(p[2]/pTot)

###



parser  = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="Input ROOT file",     default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output numpy file",   default='')

parser.add_argument("-N", "--entries",  type=int,   help="Number of entries",   default=0)

parser.add_argument("-L", "--eta_lo",   type=float, help="low eta cut",         default=0.024)
parser.add_argument("-H", "--eta_hi",   type=float, help="high eta cut",        default=1.1)

parser.add_argument("-t", "--truth",    action='store_true',    help="Toggle 1/0 for MC truth")

parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
parser.add_argument("-z", "--zip",      action='store_true',    help="Store compressed")

parser.add_argument("-d", "--debug",    action='store_true',    help="Debug mode")

###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile

entries     = args.entries
verbose     = args.verbose
truth       = args.truth

eta_lo      = args.eta_lo
eta_hi      = args.eta_hi

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

N2do=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N2do} entries out of {Nentries}, eta cut: {eta_lo}, {eta_hi}. Total EMCal channels: {Ntotal}.''')


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

(cnt, cnt_sq, square) = (0, 0, 5)

output  = None
first   = True

for i in range(N2do): # loop over the data sample
    etarap = abs(pseudorap(p[i]))
    if etarap < eta_lo or etarap > eta_hi: continue

    barrel = np.zeros(shape, dtype=float)
    ntowers = nlive[i]

    for nt in range(ntowers):
        tower  = N[i][nt]    # print(myTower, e[nt])

        my_eta = tower_map[tower][0]
        my_phi = tower_map[tower][1]
        barrel[my_eta][my_phi] = energy[i][nt]

    maxval      = barrel.max()

    indices     = np.where(barrel == maxval)

    eta, phi = (indices[0][0], indices[1][0])

    if (eta+1)<square or (eta+1)>(Neta-square) or (phi+1)<square or (phi+1)>(Nphi-square):
        cnt_sq+=1
        continue

    AOI = barrel[eta-2:eta+3, phi-2:phi+3]
    
    if first:
        output = [AOI.flatten()]
        first = False
    else:
        output = np.vstack((output, [AOI.flatten()]))
    
    cnt+=1


# print(output, output.shape)
# print(cnt, cnt_sq)

# np.random.shuffle(output)

if truth:
    truth_info = np.ones((cnt,1))
else:
    truth_info = np.zeros((cnt,1))

output  = np.hstack((output,truth_info))

if outfile == '': exit(0)

with open(outfile, 'wb') as f:
    if(args.zip):
        if verbose : print(f'''Saving to compressed file {outfile} ''')
        np.savez_compressed(f, X=output)
    else:
        if verbose : print(f'''Saving to uncompressed file {outfile} ''')
        np.save(f, output)

if verbose : print(f'''Save the array: {output.shape}''')

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