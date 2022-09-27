#!/usr/bin/env python

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

# load the datasets

gamma_file  = 'gamma.npy'
pi0_file    = 'pi0.npy'

verbose = True

### 
with open(gamma_file, 'rb') as f_gamma: gamma = np.load(f_gamma)
if verbose: print(f'''Read an array: {gamma.shape} from file {gamma_file}''')

with open(pi0_file, 'rb') as f_pi0: pi0 = np.load(f_pi0)
if verbose: print(f'''Read an array: {pi0.shape} from file {pi0_file}''')

dataset = np.vstack((gamma, pi0))

print(dataset.shape)
np.random.shuffle(dataset)

# split into input (X) and output (y) variables
X = dataset[:,0:25]
y = dataset[:,25]

# for i in range(10):     print(X[i], y[i])



# define the keras model
model = Sequential()
model.add(Dense(25, input_dim=25, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

exit(0)
