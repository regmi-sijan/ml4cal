#!/usr/bin/env python

'''
Function:
* Read an arbitrary (potenially large) array of testbeam data previously converted to numpy,
  fit it, augment with the fit results and save in a new file.
* This is intended to serve as input to "modelV3"


Notes:
* Pay attention to the "window" function, since it affects the fitting etc

'''

import  argparse

import  numpy as np
import  scipy
from    numpy import loadtxt

#################################################

# Empirical offset in the template; the template is shifted wrt data
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
parser.add_argument("-n", "--normalize",action='store_true',    help="Normalize input")

parser.add_argument("-f", "--factor",       type=float, help="Norm. factor",    default=1.0)
parser.add_argument("-t", "--threshold",    type=float, help="Threshold",       default=0.0)
parser.add_argument("-r", "--r2",           type=float, help="R2 lower limit",  default=0.95)

parser.add_argument("-N", "--entries",  type=int,   help="Number of entries to process",    default=0)
parser.add_argument("-L", "--left",     type=int,   help="If present, left window limit",   default=0)
parser.add_argument("-R", "--right",    type=int,   help="If present, right window limit",  default=0)

###################################
args        = parser.parse_args()

infile      = args.infile
outfile     = args.outfile

verbose     = args.verbose
normalize   = args.normalize

nrm         = args.factor
threshold   = args.threshold

channels    = [int(numeric_string) for numeric_string in args.channels.split(',')]

entries     = args.entries
left        = args.left
right       = args.right

window =  (left!=0 and right!=0)

print(infile, outfile, channels)

if verbose:
    print(f'''R2 threshold is set to {args.r2}\nWill use the template file "{args.tmplfile}".''')
    if window:
        print(f'''Window mode selected with limits {left}:{right}''')

try:
    template = loadtxt(args.tmplfile, delimiter=',')
except:
    print("Problem with reading template file, exiting")
    exit(-1)

# Translate the template "x" axis
if verbose: print(f'''Timing correction {t_offset}''')
vec             = template[:,0] - t_offset

dataset = None
try:
    with open(infile, 'rb') as f: dataset = np.load(f)
except:
    print(f'''Problems reading file {infile}, exiting''')
    exit(-1)

# Check the optional limit on the number of processed events, set it:
N = dataset.shape[0] if entries==0 else min(entries, dataset.shape[0])
if verbose: print(f'''Read the input array: {dataset.shape}, will process {N} events''')

fit_array   = np.zeros((N, 3))
if verbose: print(f'''Created the output array: {fit_array.shape}''')

waves       = np.zeros((N, 31)) # recall that the 32nd bin contains a constant, need to exclude it
filter      = np.full(N, True, dtype=bool) # Was: filter=np.ones(N, dtype=bool)


x = np.linspace(0, 31, 31, endpoint=False) # the "original" vector of input bin numbers
cnt_bad, cnt_out, cnt_small, first, output_array = 0, 0, 0, True, None # misc init

# Auto correction for the window start
if window:
    (t_limit_left, t_limit_right) = (2.0, 10.0)
else:
    (t_limit_left, t_limit_right) = (5.0, 18.0)



# Recall fit parameters: amplitude, time, pedestal

# -FIXME- factor 1000 implied here
if normalize:
    param_bounds=([0.02, t_limit_left, 1.0],   [15.0,    t_limit_right, 2.5])
else:
    param_bounds=([20.0, t_limit_left, 1000.0],[15000.0, t_limit_right, 2500.0])

if verbose: print(f'''Parameter bounds: {param_bounds}''')


for i in range(N): # loop over the data sample
    if (verbose and (i %1000)==0 and i!=0): print(f'''Processed: {i}  Percentage bad: {float(cnt_bad)/float(i)}''')
    frame   = dataset[i]
    for channel in channels:
        wave        = frame[channel][0:31]  # select waveform, 31 bin
        ped_guess   = np.average(wave[0:5])  # ped_guess = 1580 NB. Good guess for channel 27

        maxindex    =   np.argmax(wave)
        maxval      =   wave[maxindex]

        if args.peaktime: # strict timing selection
            if maxindex>15 or maxindex<9: # filter out outliers
                cnt_out+=1
                filter[i] = False
                continue

        if window:
            selection   = np.arange(maxindex-left, maxindex+right)
            shortwave   = np.take(wave, selection)
            print(len(shortwave))
            continue

        wave        = wave/nrm
        maxval      = maxval/nrm
        ped_guess   = ped_guess/nrm

        amp = float(maxval-ped_guess)

        if amp<threshold:
            cnt_small+=1
            filter[i] = False
            continue # reject small signals

        try:
            popt, _ = scipy.optimize.curve_fit(tempfit, x, wave, p0=[amp, float(maxindex), ped_guess], bounds = param_bounds)
        except:
            cnt_bad+=1
            filter[i] = False
            continue

        fit     = tempfit(x, *popt)
        ss_res  = np.sum((wave - fit) ** 2)              # residual sum of squares
        ss_tot  = np.sum((wave - np.mean(wave)) ** 2)    # total sum of squares
        r2      = 1 - (ss_res / ss_tot)                  # r-squared

        # if args.debug: print(str(popt[0])+', '+str(r2))

        if r2<args.r2:
            cnt_bad+=1
            filter[i] = False
            continue

        fit_array[i]    = popt
        waves[i]        = wave


waves       = waves[filter]
fit_array   = fit_array[filter]

output      = np.append(waves, fit_array, axis=1)

################################
if verbose:
    print(f'''Bad fits counter: {cnt_bad}''')
    print(f'''Below threshold counter: {cnt_small}''')
    if args.peaktime: print(f'''Peak out of bounds counter: {cnt_out} ''')
    print(f'''Output array: {output.shape}''')


if outfile == '': exit(0)
with open(outfile, 'wb') as f:
    if verbose : print(f'''Saving to uncompressed file {outfile} ''')
    np.save(f, output)

f.close()

exit(0)