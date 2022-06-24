#!/usr/bin/env python
##########################################################
# Load a model, a date file and compare the model's
# results with the data


###
def tempfit(x, *par):
    return par[0]*np.interp((x - par[1]), template[:,0], template[:,1]) + par[2]
###

import argparse

import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt
from matplotlib import colors

from keras.models import Sequential, load_model
from keras.models import load_model

import scipy
from   scipy.optimize import curve_fit

import time

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datafile",     type=str,               help="Name of the data file",           required=True)
parser.add_argument("-m", "--modelfile",    type=str,               help="Filename to load the model from", required=True)
parser.add_argument("-G", "--graphicfile",  type=str,               help="Optional filename to save graphic to (requires the graphic option)", default='')

parser.add_argument("-g", "--graphic",      action='store_true',	help="Display comparison graphic and exit")
parser.add_argument("-v", "--verbose",      action='store_true',	help="Verbose mode")
parser.add_argument("-s", "--stats",        action='store_true',	help="Calculate stats")

parser.add_argument("-b", "--batch",        type=int,               help="Batch size for inference",        default=32)
# parser.add_argument("-T", "--timing",       action='store_true',	help="Time the model")

################################################################
args    = parser.parse_args()

datafile    = args.datafile
modelfile   = args.modelfile
verb        = args.verbose
stats       = args.stats

batch       = args.batch
verbose     = args.verbose
################################################################


template = loadtxt('template.csv', delimiter=',')

if verb: print(f'''Template dimensions: {template.shape}''')

if verb: print(('Will read data from file %s, will read the model from file %s') % (datafile, modelfile))

if datafile == '':
    print('Please specify the input data file name')
    exit(-1)

start = time.time()
model   = load_model(modelfile)
end = time.time()

if verb: print("Model load - elapsed time:", end-start)

start = time.time()

with open(datafile, 'rb') as f: dataset = np.load(f)
if verbose: print(f'''Read an array: {dataset.shape}''')
L = 31 # len(dataset[0]) - 3 # the "y" vector: origin, peak value, pedestal

# Split into input (X) and output (y) variables
X = dataset[:,0:L]
y = dataset[:,(L):(L+3)]
end = time.time()

#for i in range(0,4): print(X[i], '!', y[i])
#exit(0)

if verb: print("Data prep - elapsed time:", end-start)

start = time.time()

if batch==32:
    answer = model.predict(X)
else:
    answer = model.predict(X, batch_size=batch)

end = time.time()


#for i in range(0,4): print(y[i], answer[i])
#exit(0)


if verb: print("Inference - elapsed time:", end-start)


x       = np.linspace(0, 31, 31, endpoint=False)
N       = dataset.shape[0]
fits    = [None] * N

for i in range(N): # loop over the data sample
    frame = X[i]
    wave = frame[0:31]  # print(wave)
    popt, _ = scipy.optimize.curve_fit(tempfit, x, wave, p0=[500.0, 7.0, 1500.0])
    fits[i]=popt
    # fit  = tempfit(x, *popt)

result = np.array(fits)

print(answer.shape)
print(result.shape)


if stats:
    labels = ['amplitude', 'time', 'pedestal',]
    diff = answer - y

    print('Average and Standard deviation values (ML)')
    for i in range(0,3): print(labels[i], np.average(diff[:,i]), np.std(diff[:,i]))

    diff = answer - result
    print('Average and Standard deviation values (Fit)')
    for i in range(0,3): print(labels[i], np.average(diff[:,i]), np.std(diff[:,i]))


exit(0)


if args.graphic:
    diff = answer - y

    plt.style.use('seaborn-whitegrid')

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
    
    # Parameters: origin, peak value, pedestal

    fig.set_size_inches(18.0, 12.0)

    fig.suptitle("Model vs reference")

    _ = ax1.hist(diff[:,0], bins=50, range=(-20,20))
    ax1.set_title('Residuals: Timing')

    _ = ax2.hist(diff[:,1], color='red', bins=50, range=(-100,100))
    ax2.set_title('Residuals: Peak')

    _ = ax3.hist(diff[:,2], color='magenta', bins=50, range=(-15,15))
    ax3.set_title('Residuals: Pedestals')

    _ = ax4.hist2d(y[:,1]-y[:,2], diff[:,0], bins=(50,50), range=[[0,1000], [-10,10]], norm=colors.LogNorm(0.1), cmap='Blues')
    ax4.set_title('Timing Residual vs Peak-Pedestal (log)')
    ax4.grid()


    _ = ax5.hist2d(y[:,1]-y[:,2], diff[:,1], bins=(50,50), norm=colors.LogNorm(0.1), cmap='autumn')
    ax5.set_title('Peak Residual vs Peak-Pedestal (log)')
    ax5.grid()


    _ = ax6.hist2d(diff[:,1], diff[:,2], bins=(50,50), range=[[-50,50], [-15,15]], norm=colors.LogNorm(0.1), cmap='BuPu')
    ax6.set_title('Pedestal Residual vs Peak residual (log)')
    ax6.grid()



    plt.show()

if (args.graphicfile!='' and args.graphic):
    fig.savefig(args.graphicfile, dpi=100)
    exit(0)

exit(0)

#diff0, diff1 = [], []

#for q, truth in zip(X, y):
#    answer = model.predict(np.array([q]))
#    # print(answer, truth)
#    diff = answer-truth
#    diff0.append(diff[0][0])
#    diff1.append(diff[0][1])
