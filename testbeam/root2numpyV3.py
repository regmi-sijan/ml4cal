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

template=None

t_offset = 6.17742

###
def tempfit(x, *par):
    return par[0]*np.interp((x - par[1] + t_offset), template[:,0], template[:,1]) + par[2]

###################################
import uproot3
import numpy as np
from   numpy import loadtxt

import scipy
from   scipy.optimize import curve_fit

import argparse
###################################

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",   type=str,   help="Input ROOT file",     default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output numpy file",   default='')

parser.add_argument("-t", "--tmplfile", type=str,   help="Fit template file",   default='template.csv')

parser.add_argument("-N", "--entries",  type=int,   help="Number of samples",   default=0)
parser.add_argument("-c", "--channel",  type=int,   help="Channel",             default=0)


parser.add_argument("-g", "--gain",     type=float, help="gain (guess)",        default=500.0)
parser.add_argument("-r", "--r2",       type=float, help="R2 threshold",       default=0.0)

parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
parser.add_argument("-z", "--zip",      action='store_true',    help="Store compressed")

parser.add_argument("-d", "--debug",    action='store_true',    help="Debug mode")

###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile

tmplfile    = args.tmplfile

entries     = args.entries
verbose     = args.verbose

treename    = 'trainingtree'
branchname  = 'waveform'

channel     = args.channel
gain        = args.gain


#####################################
np.set_printoptions(precision=3, linewidth=80)

if(infile==''):
    print('Please specify a valid input file name')
    exit(-1)
file    = uproot3.open(infile)


if verbose: print(f'''Will attempt to use the template file "{tmplfile}".''')

try:
    template = loadtxt(tmplfile, delimiter=',')
except:
    print("Problem with reading template file, exiting")
    exit(-1)

if verbose:
    print(f'''Template dimensions: {template.shape}''')
    print(f'''Opened file "{infile}", will use tree "{treename}"''')


dir     = file[treename]
branch  = dir[branchname]

Nentries = branch.numentries

N=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N} entries out of total {Nentries}''')

X = branch.array()

dims = X.shape
if verbose : print(f'''Read an array: {dims}''')

x  = np.linspace(0, 31, 31, endpoint=False)

first           = True
output_array    = None

cnt_bad         = 0

param_bounds=([0.0, 0.0, 1200.0],[10000.0, 30.0, 2000.0])

for i in range(N): # loop over the data sample
    if (verbose and (i %100)==0): print(i)
    frame = X[i]
    wave = frame[channel][0:31]  # print(wave)

    maxindex    =   np.argmax(wave)
    maxval      =   wave[maxindex]

    try:
        popt, _ = scipy.optimize.curve_fit(tempfit, x, wave, p0=[float(maxval-1580), float(maxindex), 1580.0], bounds = param_bounds)
    except:
        cnt_bad+=1
        continue

    fit  = tempfit(x, *popt)

    # residual sum of squares
    ss_res = np.sum((wave - fit) ** 2)

    # total sum of squares
    ss_tot = np.sum((wave - np.mean(wave)) ** 2)

    # r-squared
    r2 = 1 - (ss_res / ss_tot)
    if args.debug: print(str(popt[0])+', '+str(r2))

    if r2<args.r2:
        cnt_bad+=1
        continue

    # print(maxindex, maxval, popt, r2)

    buzz = np.std(wave-fit)

    # adding the "Y" vector: origin, peak value, pedestal, buzz
    result      = np.array(popt)
    extra       = np.array([buzz, r2])
    appended    = np.append(wave, np.append(result, extra))

    # print(appended)

    if first:
        output_array = np.array([appended])
        first = False
    else:
        output_array = np.append(output_array,[appended], axis=0)

if verbose: print(f'''Created an array: {output_array.shape}, bad fits: {cnt_bad}''')

# print(np.reshape(output_array, (-1,34)))

if(outfile == ''): exit(0)

with open(outfile, 'wb') as f:
    if(args.zip):
        if verbose : print(f'''Saving to compressed file {outfile} ''')
        np.savez_compressed(f, output_array)
    else:
        if verbose : print(f'''Saving to uncompressed file {outfile} ''')
        np.save(f, output_array)

f.close()

exit(0)

# selected = (18, 19, 20, 26, 27, 28, 34, 35, 36)

### -- attic --
# Some previous experimentation:
# f = uproot.recreate(outfile)
# f['test']=uproot.newtree({'branch': np.array([1,2,3])})
# dir.extend({'branch': np.array([1,2,3])})