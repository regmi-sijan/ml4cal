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

treename    = 'trainingtree'
branchname  = 'waveform'

channel     = args.channel
threshold   = args.threshold

normalize   = args.normalize
nrm         = args.normfactor

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


# Translate the template "x" axis
vec = template[:,0] - t_offset

dir     = file[treename]
branch  = dir[branchname]

Nentries = branch.numentries

N=Nentries if entries==0 else min(entries,Nentries)

if verbose: print(f'''Will process {N} entries out of total {Nentries}''')

X = branch.array()

dims = X.shape
if verbose : print(f'''Read an array: {dims}''')

x  = np.linspace(0, 31, 31, endpoint=False)

cnt_bad, cnt_out, cnt_small, first, output_array = 0, 0, 0, True, None

if normalize:
    param_bounds=([0.01, 3.0, 0.3],[10.0, 19.0, 2.5])
else:
    param_bounds=([20.0, 3.0, 1100.0],[14000.0, 25.0, 2400.0])



indices = range(3, 31, 3)

for i in range(N): # loop over the data sample
    x  = np.linspace(0, 31, 31, endpoint=False) # Keep it here!
    if (verbose and (i %100)==0 and i!=0): print(f'''Processed: {i}  Percentage bad: {float(cnt_bad)/float(i)}''')
    frame = X[i]
    wave = frame[channel][0:31]

    if args.short:
        wave    = np.take(wave, indices)
        x       = np.arange(3, 31, 3)


    maxindex    =   np.argmax(wave)
    maxval      =   wave[maxindex]
    
    if args.short: maxindex = x[maxindex]

    if args.window:
        if maxindex>25 or maxindex<5: # filter out outliers
            cnt_out+=1
            continue

        selection   = np.arange(maxindex-4, maxindex+6)
        maxindex    = x[maxindex]
        wave        = np.take(wave, selection)
        x           = selection


    if args.peaktime: # strict timing selection
        if maxindex>15 or maxindex<9: # filter out outliers
            cnt_out+=1
            continue
    

    # ped_guess = 1580
    ped_guess = np.average(wave[0:5])

 
    # Core fit:
    if normalize:
        wave        = wave/nrm
        maxval      = maxval/nrm
        ped_guess   = ped_guess/nrm
    
    amp = float(maxval-ped_guess)

    if amp<threshold:
        cnt_small+=1
        continue
    try:
        popt, _ = scipy.optimize.curve_fit(tempfit, x, wave, p0=[amp, float(maxindex), ped_guess], bounds = param_bounds)
    except:
        cnt_bad+=1
        continue

    fit = tempfit(x, *popt)

    ss_res = np.sum((wave - fit) ** 2)              # residual sum of squares
    ss_tot = np.sum((wave - np.mean(wave)) ** 2)    # total sum of squares
    r2 = 1 - (ss_res / ss_tot)                      # r-squared

    # if args.debug: print(str(popt[0])+', '+str(r2))

    if r2<args.r2:
        cnt_bad+=1
        continue


    if args.debug: print(popt[1]-maxindex)
    
    # Keep this as an option, for later... buzz = np.std(wave-fit)

    # adding the "Y" vector: origin, peak value, pedestal, r2
    # result      = np.array(popt)   # For two or more extra elements, use this to append: extra = np.array([buzz, r2])

    if args.window: popt[1]-=float(maxindex-4) # special case -- offset the wave since it was truncated on the left

    # Experimental:
    # if normalize:   popt[1] = popt[1]/2.0

    result      = np.array(popt)
    appended    = np.append(wave, result)

    if first:
        output_array = np.array([appended])
        first = False
    else:
        output_array = np.append(output_array,[appended], axis=0)

if verbose:
    print(f'''Bad fits counter: {cnt_bad}''')
    print(f'''Below threshold counter: {cnt_small}''')
    if args.window or args.peaktime: print(f'''Peak out of bounds counter: {cnt_out} ''')
    try:
        print(f'''Created an output array: {output_array.shape}''')
    except:
        print("There is a problem with the output array")

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