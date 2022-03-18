#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Predict the sepal width of the test set using the SOM.

.. The MIT License (MIT)

    Copyright © 2022 <copyright holders>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import pandas
from   SOMptimised import SOM
import numpy       as np

# Extract data
table        = pandas.read_csv('iris_dataset.csv').sample(frac=1)
swidth       = table['sepal width (cm)'].to_numpy()
data         = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']].to_numpy()

data_train   = data[:-10]
data_test    = data[-10:]
swidth_train = swidth[:-10]
swidth_test  = swidth[-10:]

# Fit SOM
m   = 5
n   = 5
nf  = data_train.shape[1] # Number of features
som = SOM(m=m, n=n, dim=nf, lr=1, sigma=1, max_iter=1e4, random_state=None)
som.fit(data_train, epochs=1, shuffle=True)

pred_train = som.train_bmus_
pred_test  = som.predict(data_test)

# Compute median sepal width and uncertainty
swidth_med = []
swidth_std = []

for i in range(m*n):
    tmp    = swidth_train[pred_train == i]
    
    swidth_med.append(np.nanmedian(tmp))
    swidth_std.append(np.nanstd(tmp))
    
# Predict sepal width for test set
swidth_test_pred     = np.array(swidth_med)[pred_test]
swidth_test_pred_std = np.array(swidth_std)[pred_test]


print('Predicted     Real')
for pred, err, true in zip(swidth_test_pred, swidth_test_pred_std, swidth_test):
    print(f'{pred:.1f} +- {err:.1f}    {true:.1f}')