#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

An optimised Self Organising Map which can write and read its values into and from an external file.

Most of the code comes from **Riley Smith** implementation found in `sklearn-som <https://pypi.org/project/sklearn-som/>`_ python library. Original code from Riley Smith is always marked with :python:`'.. codeauthor:: Riley Smith'`.

.. The MIT License (MIT)

    Copyright © 2022 <copyright holders>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pickle
import numpy  as     np
from   typing import Optional

class SOM():
    r"""
    .. codeauthor:: Riley Smith
    
    The 2-D, rectangular grid self-organizing map class using Numpy.
    
    :param m: (**Optional**) shape along dimension 0 (vertical) of the SOM
    :type m: :python:`int`
    :param n: (**Optional**) shape along dimesnion 1 (horizontal) of the SOM
    :type n: :python:`int`
    :param dim: (**Optional**) dimensionality (number of features) of the input space
    :type dim: :python:`int`
    :param lr: (**Optional**) initial step size for updating the SOM weights.
    :type lr: :python:`float`
    :param sigma: (**Optional**) magnitude of change to each weight. Does not update over training (as does learning rate). Higher values mean more aggressive updates to weights.
    :type sigma: :python:`float`
    :param max_iter: (**Optional**) parameter to stop training if you reach this many interation.
    :type max_iter: :python:`int`
    :param random_state: (**Optional**) integer seed to the random number generator for weight initialization. This will be used to create a new instance of Numpy's default random number generator (it will not call np.random.seed()). Specify an integer for deterministic results.
    :type random_state: :python:`int`
    """
    
    def __init__(self, m: int = 3, n: int = 3, dim: int = 3, lr: float = 1, sigma: float = 1, max_iter: int = 3000, random_state: Optional[int] = None) -> None:
        r"""
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Init method.
        """
        
        # Initialize descriptive features of SOM
        self.m            = m
        self.n            = n
        self.dim          = dim
        self.shape        = (m, n)
        self.initial_lr   = lr
        self.lr           = lr
        self.sigma        = sigma
        self.sigma2       = sigma*sigma
        self.max_iter     = max_iter
        
        # Physical parameters associated to each cell in the SOM
        self.phys         = {}

        # Initialize weights
        self.random_state = random_state
        rng               = np.random.default_rng(random_state)
        self.weights      = rng.normal(size=(m * n, dim))
        self._locations   = self._get_locations(m, n)

        # Set after fitting
        self._inertia     = None
        self._train_bmus  = None
        self._n_iter_     = None
        self._trained     = False

    def _get_locations(self, m: int, n: int) -> np.ndarray:
        r"""
        .. codeauthor:: Riley Smith
        
        Return the indices of an m by n array. Indices are returned as float to save time.
        
        :param m: shape along dimension 0 (vertical) of the SOM
        :type m: :python:`int`
        :param n: shape along dimension 1 (horizontal) of the SOM
        :type n: :python:`int`
        
        :returns: indices of the array
        :rtype: `ndarray`_ [:python:`float`]
        """
        
        return np.argwhere(np.ones(shape=(m, n))).astype(np.float64)

    def _find_bmu(self, x: np.ndarray) -> int:
        r"""
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Find the index of the best matching unit for the input vector x.
        
        :param x: input vector (1D)
        :type x: `ndarray`_
        
        :returns: index of the best matching unit
        :rtype: :python:`int`
        """
        
        diff     = self.weights-x
        distance = np.sum(diff*diff, axis=1)
        
        return np.argmin(distance)

    def step(self, x: np.ndarray) -> None:
        r"""
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Do one step of training on the given input vector.
        
        :param x: input vector (1D)
        :type x: `ndarray`_
        """

        # Get index of best matching unit
        bmu_index        = self._find_bmu(x)

        # Find location of best matching unit
        bmu_location     = self._locations[bmu_index, :]

        # Find square distance from each weight to the BMU
        diff             = self._locations - bmu_location
        bmu_distance     = np.sum(diff*diff, axis=1)
        
        # Compute update on neighborhood
        neighborhood     = np.exp(-bmu_distance / (self.sigma2))
        local_step       = self.lr * neighborhood

        # Stack local step to be proper shape for update
        local_multiplier = np.array([local_step]).T

        # Update weights
        self.weights    += local_multiplier * (x - self.weights)
        
        return

    def _compute_point_inertia(self, x: np.ndarray) -> float:
        """
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Compute the inertia of a single point. Inertia defined as squared distance from point to closest cluster center (BMU)
        
        :param x: input vector (1D)
        :type x: `ndarray`_
        
        :returns: inertia for the point
        :rtype: :python:`float`
        """
        
        # Find BMU
        bmu_index = self._find_bmu(x)
        bmu       = self.weights[bmu_index]
        
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        diff      = x - bmu
        
        return np.sum(diff*diff)
    
    def _compute_points_inertia(self, X: np.ndarray, bmus_indices: Optional = None) -> np.ndarray:
        """
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Compute the inertia for a set of points. Inertia defined as squared distance from point to closest cluster center (BMU)
        
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :param bmus_indices: (**Optional**) indices of the best matching units for all the points. If :python:`None`, the bmus are computed.
        
        :returns: inertia for all the points
        :rtype: `ndarray`_ [:python:`float`]
        """
        
        if bmus_indices is None:
            bmus_indices = self._find_bmus_byweight(X)
            
        diff    = X - self.weights[bmus_indices]
        
        return  np.sum(diff*diff, axis=1)

    def fit(self, X: np.ndarray, epochs: int = 1, shuffle: bool = True) -> None:
        """
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Take data (a tensor of type `float64`_) as input and fit the SOM to that data for the specified number of epochs.

        :param X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :type X: `ndarray`_
        
        :param epochs: (**Optional**) number of times to loop through the training data when fitting
        :type epochs: :python:`int`
        :param shuffle: (**Optional**) whether or not to randomize the order of train data when fitting. Can be seeded with np.random.seed() prior to calling fit.
        :type shuffle: :python:`bool`
        """
        
        # Count total number of iterations
        global_iter_counter          = 0
        n_samples                    = X.shape[0]
        total_iterations             = np.minimum(epochs * n_samples, self.max_iter)

        for epoch in range(epochs):
            
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                rng                  = np.random.default_rng(self.random_state)
                indices              = rng.permutation(n_samples)
            else:
                indices              = np.arange(n_samples)

            # Train
            for idx in indices:
                
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                
                # Do one step of training
                inp                  = X[idx]
                self.step(inp)
                
                # Update learning rate
                global_iter_counter += 1
                self.lr              = (1 - (global_iter_counter / total_iterations)) * self.initial_lr

        # Store bmus of train set
        self._train_bmus             = self._find_bmus_byweight(X)
        
        # Compute total inertia
        inertia                      = self._compute_points_inertia(X, bmus_indices=self._train_bmus)
        
        self._inertia_               = np.sum(inertia)

        # Set n_iter_ attribute
        self._n_iter_                = global_iter_counter

        # Set trained flag
        self._trained                = True

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Predict cluster for each element in X.

        :param X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :type X: `ndarray`_

        :returns: an ndarray of shape (n,). The predicted cluster index for each item in X.
        :rtype: `ndarray`_ [:python:`int`]
        
        :raises NotImplmentedError: if :py:meth:`~.SOM.fit` method has not been called already
        :raises ValueError:
            
        * if **X** is not a 2-dimensional array
        * if the second dimension of **X** has not a length equal to self.dim
        """
        
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        if len(X.shape) != 2:
            raise ValueError(f'X should have two dimensions, not {len(X.shape)}.')
            
        if  X.shape[1] != self.dim:
            raise ValueError(f'This SOM has dimension {self.dim}. Received input with dimension {X.shape[1]}.')
        
        #labels = self._find_bmus_bydata(X)
        labels = self._find_bmus_byweight(X)
            
        return labels
    
    def _find_bmus_bydata(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Find the indices of the best matching unit for the input matrix X by looping through the data.
        
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :returns: indices of the best matching units
        :rtype: `ndarray`_ [:python:`int`]
        """
        
        return np.array([self._find_bmu(x) for x in X])
    
    def _find_bmus_byweight(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Find the indices of the best matching unit for the input matrix X by looping through the weights.
        
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :returns: indices of the best matching units
        :rtype: `ndarray`_ [:python:`int`]
        """
        
        # Output indices set to 0 by default
        indices           = np.zeros(len(X), dtype=int)
        
        # Compute first distance
        diff              = X - self.weights[0]
        dist              = np.sum(diff*diff, axis=1)
        
        # Only update weight position if distance is less than the previous one
        for pos, weight in enumerate(self.weights[1:]):
            
            diff          = X - weight
            tmp           = np.sum(diff*diff, axis=1)
            mask          = tmp < dist
            indices[mask] = pos+1
            dist[mask]    = tmp[mask]
            
        return indices
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Transform the data X into cluster distance space.

        :param X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :type X: `ndarray`_

        :returns: tansformed data of shape (n, self.n*self.m). The Euclidean distance from each item in X to each cluster center.
        :rtype: `ndarray`_ [:python:`float`]
        """
        
        # Stack data and cluster centers
        X_stack       = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff          = X_stack - cluster_stack

        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Convenience method for calling fit(X) followed by predict(X).

        :param X: data of shape (n, self.dim). The data to fit and then predict.
        :type X: `ndarray`_
        
        :param \**kwargs: optional keyword arguments for the :py:meth:`~.SOM.fit` method

        :returns: ndarray of shape (n,). The index of the predicted cluster for each item in X (after fitting the SOM to the data in X).
        :rtype: `ndarray`_ [:python:`float`]
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Convenience method for calling fit(X) followed by transform(X). Unlike in sklearn, this is not implemented more efficiently (the efficiency is the same as calling fit(X) directly followed by transform(X)).

        :param X: data of shape (n, self.dim) where n is the number of samples
        :type X: `ndarray`_
        
        :param \**kwargs: optional keyword arguments for the :py:meth:`~.SOM.fit` method

        :returns: ndarray of shape (n, self.m*self.n). The Euclidean distance from each item in **X** to each cluster center.
        :rtype: ndarray[:python:`float`]
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return points in cluster distance space
        return self.transform(X)
    
    ######################################
    #             IO methods             #
    ######################################
    
    def write(self, fname: str, *args, **kwargs) -> None:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Write the result of the SOM into a binary file.
        
        :param fname: output filename
        :type fname: :python:`str`
        
        :raises TypeError: if **fname** is not of type :python:`str`
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'fname is of type {type(fname)} but it must be of type str.')
        
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
            
        return
    
    @staticmethod
    def read(fname: str, *args, **kwargs):
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Read the result of a SOM written into a binary file with the :py:meth:`~.SOM.write` method.
        
        :param fname: input file
        :type fname: :python:`str`
        
        :returns: the loaded SOM object
        :rtype: :py:class:`~.SOM`
        
        :raises TypeError: if **fname** is not of type :python:`str`
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'fname is of type {type(fname)} but it must be of type str.')
            
        with open(fname, 'rb') as data:
            return pickle.load(data)
    
    #################################################
    #          Physical parameters methods          #
    #################################################
    
    def get(self, param: str) -> np.ndarray:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Return the given physical parameters if it exists.
        
        :param param: parameter to return
        :type param: :python:`str`
        
        :returns: array of physical parameter value associated to each node
        :rtype: `ndarray`_
        
        :raises KeyError: if **param** is not found
        '''
        
        if param not in self.phys:
            raise KeyError(f'physical parameter {param} not found.')
            
        return self.phys[param]
    
    def set(self, param: str, value: np.ndarray) -> None:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Set the given physical parameter. Must be an array of shape (self.n*self.m,)
        
        :param param: parameter to set
        :type param: :python:`str`
        :param value: array with the values of the physical parameter to store
        :type value: `ndarray`_
    
        :raises ValueError: if **value** is not a 1-dimensional array of length self.m*self.n
        '''
        
        if np.shape(value) != (self.m*self.n,):
            raise ValueError(f'value has shape {np.shape(value)} but it must have a shape ({self.m*self.n}).')
            
        self.phys[param] = value
        
        return
    
    ##########################################
    #               Properties               #
    ##########################################

    @property
    def cluster_centers_(self) -> np.ndarray:
        '''
        .. codeauthor:: Riley Smith
        
        Give the coordinates of each cluster centre as an array of shape (m, n, dim).
        
        :returns: cluster centres
        :rtype: `ndarray`_ [:python:`int`]
        '''
        
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self) -> np.ndarray:
        '''
        .. codeauthor:: Riley Smith
        
        Inertia.
        
        :returns: computed inertia
        :rtype: `ndarray`_ [:python:`float`]
        
        :raises AttributeError: if the SOM does not have the inertia already computed
        '''
        
        if self._inertia_ is None:
            raise AttributeError('SOM does not have inertia until after calling fit().')
            
        return self._inertia_

    @property
    def n_iter_(self) -> int:
        '''
        .. codeauthor:: Riley Smith
        
        Number of iterations.
        
        :returns: number of iterations
        :rtype: :python:`int`
        
        :rtype AttributeError: if the number of iterations is not initialised yet
        '''
        
        if self._n_iter_ is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit().')
            
        return self._n_iter_
    
    @property
    def train_bmus_(self) -> np.ndarray:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Best matching units indices for the train set.
        
        :returns: BMUs indices for the train set
        :rtype: `ndarray`_ [:python:`int`]
        
        :rtype AttributeError: if the number of iterations is not initialised yet
        '''
        
        if self._train_bmus is None:
            raise AttributeError('SOM does have train_bmus_ attribute until after calling fit().')
            
        return self._train_bmus

