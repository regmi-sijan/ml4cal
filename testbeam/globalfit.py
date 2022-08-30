#!/usr/bin/env python
##########################################################
# Read an argitrary (potenially large) array of testbeam
# data previously converted to numpy, fit it, augment and
# save in a new file
##########################################################

import argparse
from concurrent.futures.process import _chain_from_iterable_of_lists
from locale import normalize
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

parser.add_argument("-i", "--infile",   type=str,   help="Input file",          default='')
parser.add_argument("-o", "--outfile",  type=str,   help="Output file",         default='')
parser.add_argument("-c", "--channels", type=str,   help="List of channels",    default='')
parser.add_argument("-T", "--tmplfile", type=str,   help="Fit template file",   default='template.csv')

parser.add_argument("-v", "--verbose",  action='store_true',    help="Verbose mode")
parser.add_argument("-p", "--peaktime", action='store_true',    help="Strict cut on peak time")

parser.add_argument("-n", "--normalization",type=float, help="threshold",       default=1.0)
parser.add_argument("-t", "--threshold",    type=float, help="threshold",       default=0.0)
parser.add_argument("-r", "--r2",           type=float, help="R2 threshold",    default=0.95)

###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile
verbose     = args.verbose
nrm         = args.normalization
threshold   = args.threshold

channels    = [int(numeric_string) for numeric_string in args.channels.split(',')]

print(infile, outfile, channels)

if verbose:
    print(f'''R2 threshold is set to {args.r2}''')
    print(f'''Will attempt to use the template file "{args.tmplfile}".''')

try:
    template = loadtxt(args.tmplfile, delimiter=',')
except:
    print("Problem with reading template file, exiting")
    exit(-1)

# Translate the template "x" axis
vec         = template[:,0] - t_offset

with open(infile, 'rb') as f: dataset = np.load(f)
if verbose: print(f'''Read the input array: {dataset.shape}''')

N = dataset.shape[0]

cut_dataset = np.delete(dataset, 31, 2)
if verbose: print(f'''Truncated input array: {cut_dataset.shape}''')

fit_array   = np.zeros((N, 3))
filter      = np.ones(N, dtype=bool)

if verbose: print(f'''Created the output array: {fit_array.shape}''')

x = np.linspace(0, 31, 31, endpoint=False)
cnt_bad, cnt_out, cnt_small, first, output_array = 0, 0, 0, True, None

# channels = [26,27]


#if nrm!=1.0:
# amplitude, time, pedestal
param_bounds=([0.01, 3.0, 0.3],[20.0, 19.0, 2.5])

# else:
#    param_bounds=([20.0, 3.0, 1100.0],[14000.0, 25.0, 2400.0])

for i in range(N): # loop over the data sample
    if (verbose and (i %1000)==0 and i!=0): print(f'''Processed: {i}  Percentage bad: {float(cnt_bad)/float(i)}''')

    frame   = cut_dataset[i]                  # select a row
    for channel in channels:
        wave        = frame[channel][0:31]  # select waveform, 31 bin
        ped_guess   = np.average(wave[0:5])  # ped_guess = 1580 NB. Good guess for channel 27

        maxindex    =   np.argmax(wave)
        maxval      =   wave[maxindex]

        if args.peaktime: # strict timing selection
            if maxindex>15 or maxindex<9: # filter out outliers
                cnt_out+=1
                continue
        

        wave        = wave/nrm
        maxval      = maxval/nrm
        ped_guess   = ped_guess/nrm

        amp = float(maxval-ped_guess)


        if amp<threshold:
            cnt_small+=1
            continue # reject small signals

        try:
            popt, _ = scipy.optimize.curve_fit(tempfit, x, wave, p0=[amp, float(maxindex), ped_guess], bounds = param_bounds)
        except:
            cnt_bad+=1
            continue

        fit     = tempfit(x, *popt)
        ss_res  = np.sum((wave - fit) ** 2)              # residual sum of squares
        ss_tot  = np.sum((wave - np.mean(wave)) ** 2)    # total sum of squares
        r2      = 1 - (ss_res / ss_tot)                  # r-squared

        # if args.debug: print(str(popt[0])+', '+str(r2))

        if r2<args.r2:
            cnt_bad+=1
            continue

################################
if verbose:
    print(f'''Bad fits counter: {cnt_bad}''')
    print(f'''Below threshold counter: {cnt_small}''')
    if args.peaktime: print(f'''Peak out of bounds counter: {cnt_out} ''')