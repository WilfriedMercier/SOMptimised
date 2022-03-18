#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Run the SOM on the iris dataset.

.. The MIT License (MIT)

    Copyright © 2022 <copyright holders>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import matplotlib.pyplot   as     plt
from   matplotlib.colors   import TwoSlopeNorm
from   matplotlib.gridspec import GridSpec
from   matplotlib          import rc
import matplotlib          as     mpl
import pandas
from   SOMptimised         import SOM

# Extract data
table      = pandas.read_csv('iris_dataset.csv').sample(frac=1)
target     = table['target']
table      = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]

data       = table.to_numpy()

data_train = data[:-10]
data_test  = data[-10:]

# Fit SOM
nf  = data_train.shape[1] # Number of features
som = SOM(m=1, n=3, dim=nf, lr=1, sigma=1, max_iter=1e4, random_state=None)
som.fit(data_train, epochs=1, shuffle=True)

pred_train = som.train_bmus_
pred_test  = som.predict(data_test)

# PLotting
norm = TwoSlopeNorm(1, vmin=0, vmax=2)

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'

f   = plt.figure(figsize=(10, 4.5))
gs  = GridSpec(1, 2, wspace=0)
ax1 = f.add_subplot(gs[0])
ax2 = f.add_subplot(gs[1])

for ax in [ax1, ax2]:
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='x', which='both', direction='in', labelsize=13, length=3)
    ax.tick_params(axis='y', which='both', direction='in', labelsize=13, length=3)
    ax.set_xlabel('Petal width (cm)', size=16)
    
ax1.scatter(data_train[:, 1], data_train[:, 0], c=pred_train, cmap='bwr', ec='k', norm=norm, marker='o', s=30)
ax1.scatter(data_test[:, 1],  data_test[:, 0],  c=pred_test,  cmap='bwr', ec='k', marker='o', norm=norm, s=60)
ax1.set_ylabel('Petal length (cm)', size=16)

target = target.to_numpy()
target[target == 'Iris-setosa'] = 0
target[target == 'Iris-versicolor'] = 1
target[target == 'Iris-virginica'] = 2

ax2.scatter(data_train[:, 1], data_train[:, 0], c=target[:-10], cmap='bwr', ec='k', norm=norm, marker='o', s=30)
ax2.scatter(data_test[:, 1],  data_test[:, 0],  c=target[-10:],  cmap='bwr', ec='k', marker='o', norm=norm, s=60)
ax2.set_yticks([0.2])
ax2.set_yticklabels([])

ax1.set_title('SOM clustering', size=18)
ax2.set_title('IRIS dataset', size=18)

plt.show()
