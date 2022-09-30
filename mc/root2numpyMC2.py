#!/usr/bin/env python
'''

Interface to the MC data (photon/gamma)

"old format"

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
def angle_index(angle, ref_array):
    for i in range(256):
        if math.isclose(angle, ref_array[i], rel_tol=1e-6): return i
    
    return -1

########################################################################
unique_etas = [-1.12303412,-1.1017679 ,-1.0800885 ,-1.05861473,-1.03672588,-1.01504612,
 -0.99294972,-0.97106558,-0.94876307,-0.92667603,-0.9041689 ,-0.88188028,
 -0.8591699 ,-0.83668089,-0.81376833,-0.79107976,-0.76796585,-0.74507833,
 -0.72176349,-0.69867712,-0.6751613 ,-0.65187591,-0.62815851,-0.60467309,
 -0.58075309,-0.55706614,-0.53294152,-0.50905073,-0.48471886,-0.46062115,
 -0.43607837,-0.41176963,-0.38701132,-0.36248645,-0.33750686,-0.31275961,
 -0.28755182,-0.26257476,-0.23713052,-0.21191494,-0.18622468,-0.16076051,
 -0.13569817,-0.11177617,-0.08742137,-0.06338444,-0.03894061,-0.01484409,
  0.01484409, 0.03894061, 0.06338444, 0.08742137, 0.11177617, 0.13569817,
  0.16076051, 0.18622468, 0.21191494, 0.23713052, 0.26257476, 0.28755182,
  0.31275961, 0.33750686, 0.36248645, 0.38701132, 0.41176963, 0.43607837,
  0.46062115, 0.48471886, 0.50905073, 0.53294152, 0.55706614, 0.58075309,
  0.60467309, 0.62815851, 0.65187591, 0.6751613 , 0.69867712, 0.72176349,
  0.74507833, 0.76796585, 0.79107976, 0.81376833, 0.83668089, 0.8591699 ,
  0.88188028, 0.9041689 , 0.92667603, 0.94876307, 0.97106558, 0.99294972,
  1.01504612, 1.03672588, 1.05861473, 1.0800885 , 1.1017679 , 1.12303412]

unique_phis = [-3.12932086,-3.1047771 ,-3.08023334,-3.05568981,-3.03114605,-3.00660229,
 -2.98205876,-2.957515  ,-2.93297124,-2.90842748,-2.88388395,-2.85934019,
 -2.83479643,-2.8102529 ,-2.78570914,-2.76116538,-2.73662162,-2.71207809,
 -2.68753433,-2.66299057,-2.63844705,-2.61390328,-2.58935952,-2.56481576,
 -2.54027224,-2.51572847,-2.49118471,-2.46664119,-2.44209743,-2.41755366,
 -2.39301014,-2.36846638,-2.34392262,-2.31937885,-2.29483533,-2.27029157,
 -2.2457478 ,-2.22120428,-2.19666052,-2.17211676,-2.14757299,-2.12302947,
 -2.09848571,-2.07394195,-2.04939842,-2.02485466,-2.0003109 ,-1.97576725,
 -1.95122361,-1.92667985,-1.90213621,-1.87759244,-1.8530488 ,-1.82850516,
 -1.8039614 ,-1.77941775,-1.75487399,-1.73033035,-1.70578659,-1.68124294,
 -1.6566993 ,-1.63215554,-1.60761189,-1.58306813,-1.55852449,-1.53398085,
 -1.50943708,-1.48489344,-1.46034968,-1.43580604,-1.41126227,-1.38671863,
 -1.36217499,-1.33763123,-1.31308758,-1.28854382,-1.26400018,-1.23945653,
 -1.21491277,-1.19036913,-1.16582537,-1.14128172,-1.11673796,-1.09219432,
 -1.06765068,-1.04310691,-1.01856327,-0.99401957,-0.96947587,-0.94493216,
 -0.92038846,-0.89584476,-0.87130111,-0.84675741,-0.82221371,-0.79767001,
 -0.7731263 ,-0.7485826 ,-0.72403896,-0.69949526,-0.67495155,-0.65040785,
 -0.62586415,-0.60132045,-0.5767768 ,-0.5522331 ,-0.5276894 ,-0.50314569,
 -0.47860199,-0.45405832,-0.42951462,-0.40497091,-0.38042724,-0.35588354,
 -0.33133984,-0.30679616,-0.28225246,-0.25770876,-0.23316509,-0.20862138,
 -0.1840777 ,-0.15953401,-0.1349903 ,-0.11044662,-0.08590292,-0.06135923,
 -0.03681554,-0.01227185, 0.01227185, 0.03681554, 0.06135923, 0.08590292,
  0.11044662, 0.1349903 , 0.15953401, 0.1840777 , 0.20862138, 0.23316509,
  0.25770876, 0.28225246, 0.30679616, 0.33133984, 0.35588354, 0.38042724,
  0.40497091, 0.42951462, 0.45405832, 0.47860199, 0.50314569, 0.5276894 ,
  0.5522331 , 0.5767768 , 0.60132045, 0.62586415, 0.65040785, 0.67495155,
  0.69949526, 0.72403896, 0.7485826 , 0.7731263 , 0.79767001, 0.82221371,
  0.84675741, 0.87130111, 0.89584476, 0.92038846, 0.94493216, 0.96947587,
  0.99401957, 1.01856327, 1.04310691, 1.06765068, 1.09219432, 1.11673796,
  1.14128172, 1.16582537, 1.19036913, 1.21491277, 1.23945653, 1.26400018,
  1.28854382, 1.31308758, 1.33763123, 1.36217499, 1.38671863, 1.41126227,
  1.43580604, 1.46034968, 1.48489344, 1.50943708, 1.53398085, 1.55852449,
  1.58306813, 1.60761189, 1.63215554, 1.6566993 , 1.68124294, 1.70578659,
  1.73033035, 1.75487399, 1.77941775, 1.8039614 , 1.82850516, 1.8530488 ,
  1.87759244, 1.90213621, 1.92667985, 1.95122361, 1.97576725, 2.0003109 ,
  2.02485466, 2.04939842, 2.07394195, 2.09848571, 2.12302947, 2.14757299,
  2.17211676, 2.19666052, 2.22120428, 2.2457478 , 2.27029157, 2.29483533,
  2.31937885, 2.34392262, 2.36846638, 2.39301014, 2.41755366, 2.44209743,
  2.46664119, 2.49118471, 2.51572847, 2.54027224, 2.56481576, 2.58935952,
  2.61390328, 2.63844705, 2.66299057, 2.68753433, 2.71207809, 2.73662162,
  2.76116538, 2.78570914, 2.8102529 , 2.83479643, 2.85934019, 2.88388395,
  2.90842748, 2.93297124, 2.957515  , 2.98205876, 3.00660229, 3.03114605,
  3.05568981, 3.08023334, 3.1047771 , 3.12932086]

#########################################################################
parser  = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="Input ROOT file",     default='~/data/mc/gamma_10000_evts.root')
parser.add_argument("-o", "--outfile",  type=str,   help="Output numpy file",   default='')

parser.add_argument("-N", "--entries",  type=int,   help="Number of entries",   default=0)

parser.add_argument("-L", "--eta_lo",   type=float, help="low eta cut",         default=0.1) # original -- 0.024
parser.add_argument("-H", "--eta_hi",   type=float, help="high eta cut",        default=1.0) # original -- 1.1

parser.add_argument("-t", "--truth",    action='store_true',    help="Toggle 1/0 for MC truth")
parser.add_argument("-p", "--part",     action='store_true',    help="1st (true) or 2nd (false) part of data")

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

part       = args.part

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
Nentries    = 10000 # --FIXME-- p_branch.numentries

N2do=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N2do} entries out of {Nentries}, eta cut: {eta_lo}, {eta_hi}. Total EMCal channels: {Ntotal}.''')


px = dir['truthpar_px'].array()
py = dir['truthpar_py'].array()
pz = dir['truthpar_pz'].array()

e_tow   = dir['emcalen'].array()
eta_tow = dir['emcalet'].array()
phi_tow = dir['emcalph'].array()


(cnt, cnt_sq, square) = (0, 0, 5)

output  = None
first   = True

for i in range(N2do): # loop over the data sample
    etarap  = pseudorap([px[i], py[i], pz[i]])
    if abs(etarap) < eta_lo or abs(etarap) > eta_hi: continue

    barrel  = np.zeros(shape, dtype=float)

    Nlive   = e_tow[i].size

    ### Populate the barrel:
    for nl in range(Nlive):
        # Test print to check separation of seed and other leading towers: if(e_tow[i][nl]<1.0): continue
        eta = angle_index(eta_tow[i][nl], unique_etas)
        phi = angle_index(phi_tow[i][nl], unique_phis)
        barrel[eta][phi] = e_tow[i][nl]

    ###
    maxval      = barrel.max()
    indices     = np.where(barrel == maxval)
    eta, phi    = (indices[0][0], indices[1][0])

    if (eta+1)<square or (eta+1)>(Neta-square) or (phi+1)<square or (phi+1)>(Nphi-square):
        cnt_sq+=1
        continue

    AOI = barrel[eta-2:eta+3, phi-2:phi+3] # print(AOI)

    if first:
        output = [AOI.flatten()]
        first = False
    else:
        output = np.vstack((output, [AOI.flatten()]))
    
    cnt+=1

### Done with the data, now augment and save output

if verbose : print(f'''Events passed processing: {cnt} ''')

if truth:
    truth_info = np.ones((cnt,1))
else:
    truth_info = np.zeros((cnt,1))

output  = np.hstack((output,truth_info))


to_keep = int(output.shape[0]/2)
if part:
    output = output[0:to_keep,:]
else:
    output = output[to_keep:2*to_keep,:]

if verbose : print(f'''Output array: {output.shape}, first half: {part} ''')

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