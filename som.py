#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

An optimised Self Organising Map which can write and read its values into and from an external file.

Most of the code comes from **Riley Smith** implementation found in `sklearn-som <https://pypi.org/project/sklearn-som/>`_ python library. Original code from Riley Smith is always marked with :python:`'.. codeauthor:: Riley Smith'`.

.. The MIT License (MIT)

    Copyright © 2022 <Wilfried Mercier>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import pickle
import joblib
import colorama
import multiprocessing
import numpy          as     np
from   typing         import Optional, Union
from   .learning_rate import LearningStrategy, LinearLearningStrategy
from   .neighbourhood import NeighbourhoodStrategy, ConstantRadiusStrategy
from   . metric       import euclidianMetric

# Automatically reset any color used with colorama when printing
colorama.init(autoreset=True)

# Maximum number of threads on the computer
N_JOBS_MAX = multiprocessing.cpu_count()

class SOM():
    r'''
    .. codeauthor:: Riley Smith
    
    Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
    
    The 2-D, rectangular grid self-organizing map class using Numpy.
    
    :param m: (**Optional**) shape along dimension 0 (vertical) of the SOM
    :type m: :python:`int`
    :param n: (**Optional**) shape along dimesnion 1 (horizontal) of the SOM
    :type n: :python:`int`
    :param dim: (**Optional**) dimensionality (number of features) of the input space
    :type dim: :python:`int`
    :param lr: (**Optional**) learning strategy used to update the SOM weights
    :type lr: :py:class:`~.LearningStrategy`
    :param sigma: (**Optional**) neighbourhood strategy used to compute the step applied to each weight.
    :type sigma: :py:class:`~.NeighbourhoodStrategy`
    :param max_iter: (**Optional**) parameter to stop training if you reach this many interations
    :type max_iter: :python:`int` or :python:`float`
    :param metric: (**Optional**) metric used to compute the distance between the train data and the neurons, and between the neurons and the test data
    :type metric: :python:`callable`
    :param random_state: (**Optional**) integer seed to the random number generator for weight initialization. This will be used to create a new instance of Numpy's default random number generator (it will not call np.random.seed()). Specify an integer for deterministic results.
    :type random_state: :python:`int`
    '''
    
    def __init__(self, 
                 m: int                       = 3, 
                 n: int                       = 3, 
                 dim: int                     = 3, 
                 lr: LearningStrategy         = LinearLearningStrategy(lr=1), 
                 sigma: NeighbourhoodStrategy = ConstantRadiusStrategy(sigma=1), 
                 metric: callable             = euclidianMetric,
                 max_iter: Union[int, float]  = 3000,
                 random_state: Optional[int]  = None) -> None:
        r'''
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Init method.
        '''
        
        # Check types
        if not isinstance(lr, LearningStrategy):
           raise TypeError(f'learning strategy lr has type {type(lr)} but it must be a LearningStrategy object.')
           
        if not isinstance(sigma, NeighbourhoodStrategy):
           raise TypeError(f'neighbourhood radius sigma has type {type(sigma)} but it must be a NeighbourhoodStrategy object.')
           
        for var, name in zip((m, n, dim), ('m', 'n', 'dim')):
           if not isinstance(var, int):
              raise TypeError(f'parameter {name} has type {type(var)} but it must be an int.')
              
        if not isinstance(max_iter, (int, float)):
           raise TypeError(f'max_iter has type {type(max_iter)} but it must be an int or float.')
              
        if random_state is not None and not isinstance(random_state, int):
           raise TypeError(f'parameter random_state has type {type(random_state)} but it must be an int.')
        
        # Initialize descriptive features of SOM
        self.m            = m
        self.n            = n
        self.dim          = dim
        self.lr           = lr
        self.sigma        = sigma
        self.metric       = metric
        self.max_iter     = int(max_iter)
        
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
        r'''
        .. codeauthor:: Riley Smith
        
        Return the indices of an m by n array. Indices are returned as float to save time.
        
        :param m: shape along dimension 0 (vertical) of the SOM
        :type m: :python:`int`
        :param n: shape along dimension 1 (horizontal) of the SOM
        :type n: :python:`int`
        
        :returns: indices of the array
        :rtype: `ndarray`_ [:python:`float`]
        '''
        
        return np.argwhere(np.ones(shape=(m, n))).astype(np.float64)

    def _find_bmu(self, x: np.ndarray, *args, metric: Optional[callable] = None, **kwargs) -> int:
        r'''
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Find the index of the best matching unit for the input vector x.
        
        :param x: input vector (1D)
        :type x: `ndarray`_
        
        :param metric: (**Optional**) metric to use. If None, the metric provided at init is used.
        :type metric: :python:`callable`
        
        :param \*args: additional arguments to pass to the metric. This must be a tuple or list of 1D `ndarray`_ with the same shape as **x**. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        
        :returns: index of the best matching unit
        :rtype: :python:`int`
        '''
        
        metric   = metric if metric is not None else self.metric
        distance = metric(x, self.weights, *args, squared=True, axis=1, **kwargs)
        
        return np.argmin(distance)

    def step(self, x: np.ndarray, counter: int, *args, **kwargs) -> None:
        r'''
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Do one step of training on the given input vector.
        
        :param x: input vector (1D)
        :type x: `ndarray`_
        :param counter: global counter used to compute the neighbourhood radius and the learning rate
        :type counter: :python:`int`
           
        :param \*args: additional arguments to pass to the metric. This must be a tuple or list of 1D `ndarray`_ with the same shape as **x**. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        '''

        # Get index of best matching unit (we set metric to None to use the one provided in init)
        bmu_index        = self._find_bmu(x, *args, metric=None, **kwargs)

        # Find location of best matching unit
        bmu_location     = self._locations[bmu_index, :]

        # Find square distance from each weight to the BMU
        diff             = self._locations - bmu_location
        bmu_distance     = np.sum(diff*diff, axis=1)
        
        # Compute learning rate
        lr               = self.lr(counter)
        
        # Compute neighbourhood radius squared
        sigma2           = self.sigma(counter, squared=True)
        
        # Compute update on neighborhood
        neighbourhood    = np.exp(-bmu_distance / sigma2)
        
        # Compute local step
        local_step       = lr * neighbourhood

        # Stack local step to be proper shape for update
        local_multiplier = np.array([local_step]).T

        # Update weights
        self.weights    += local_multiplier * (x - self.weights)
        
        return
    
    def _compute_points_inertia(self, X: np.ndarray, *args,
                                bmus_indices: Optional[Union[int, list, np.ndarray]] = None,
                                metric: Optional[callable]                           = None,
                                n_jobs: int                                          = 1,
                                **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Compute the inertia for a set of points. Inertia defined as squared distance from point to closest cluster center (BMU).
        
        .. note::
            
            ***args** and **\**kwargs** are additional arguments and keyword arguments which can be passed depending on the metric used. In this implementation:
                
            - **args** must always be a collection of `ndarray`_ with shapes similar to that of **X**
            - **\**kwargs** are keyword arguments which have no constraints on their type or shape
             
            See the metric specific implementation for more details.
            
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :param bmus_indices: (**Optional**) indices of the best matching units for all the points. If :python:`None`, the bmus are computed.
        :type bmus_indices: :python:`int`, :python:`list` [:python:`int`] or `ndarray`_ [:python:`int`]
        :param metric: (**Optional**) metric to use. If None, the metric provided at init is used.
        :type metric: :python:`callable`
        
        
        :param \*args: additional arguments to pass to the metric. These arguments are looped similarly to **X**, so they should be a collection of `ndarray`_ with the same shape. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        
        :returns: inertia for all the points
        :rtype: `ndarray`_ [:python:`float`]
        '''
        
        metric       = metric if metric is not None else self.metric
        bmus_indices = bmus_indices if bmus_indices is not None else self._find_bmus(X, *args, metric=metric, n_jobs=n_jobs, **kwargs)
        
        return metric(X, self.weights[bmus_indices], *args, squared=True, axis=1, **kwargs)

    def fit(self, X: np.ndarray, *args, epochs: int = 1, shuffle: bool = True, n_jobs: int = 1, unnormalise_weights: bool = False, **kwargs) -> None:
        r'''
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Take data (a tensor of type `float64`_) as input and fit the SOM to that data for the specified number of epochs.
        
        .. note::
            
            ***args** and **\**kwargs** are additional arguments and keyword arguments which can be passed depending on the metric used. In this implementation:
                  
            - ***args** must always be a collection of `ndarray`_ with shapes similar to that of **X**
            - **\**kwargs** are keyword arguments which have no constraints on their type or shape
             
            See the metric specific implementation for more details.
        
        :param X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :type X: `ndarray`_
        
        :param epochs: (**Optional**) number of times to loop through the training data when fitting
        :type epochs: :python:`int`
        :param shuffle: (**Optional**) whether or not to randomize the order of train data when fitting. Can be seeded with np.random.seed() prior to calling :py:meth:`~.SOM.fit` method.
        :type shuffle: :python:`bool`
        :param n_jobs: (**Optional**) number of threads used to find the BMUs at the end of the loop. This parameter is only used when using :py:meth:`~.SOM._find_bmus_bydata` method.
        :type n_jobs: :python:`int`
        :param unnormalise_weights: whether to unnormalise weights or not
        :type unnormalise_weights: :python:`bool`
        
        :param \*args: additional arguments to pass to the metric. These arguments are looped similarly to **X**, so they should be a collection of `ndarray`_ with the same shape. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        '''
        
        # Count total number of iterations
        global_iter_counter          = 0
        n_samples                    = X.shape[0]
        total_iterations             = np.minimum(epochs * n_samples, self.max_iter)
        
        # Unnormalise the weights
        if unnormalise_weights:
            self.weights             = self.weights*np.nanstd(X, axis=0) + np.nanmean(X, axis=0)
        
        # Set the _ntot attribute if the learning rate strategy requires it
        if '_ntot' in self.lr.__dict__:
           self.lr.ntot              = int(total_iterations)

        for epoch in range(epochs):
            
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                rng                  = np.random.default_rng(self.random_state)
                indices              = rng.permutation(n_samples)
                   
            else:
               indices               = np.arange(n_samples)

            ##########################
            #        Training        #
            ##########################
            
            for idx in indices:
                
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
               
                # Data for the step idx
                inp                  = X[idx]
                
                # Additional arguments for the step idx
                step_args            = (i[idx] for i in args)
                
                self.step(inp, global_iter_counter, *step_args, **kwargs)
                  
                # Update learning rate
                global_iter_counter += 1

        # Store bmus of train set
        self._train_bmus             = self._find_bmus(X, *args, n_jobs=n_jobs, **kwargs)
        
        # Compute total inertia (metric set to None because we use the one provided in init)
        inertia                      = self._compute_points_inertia(X, *args, bmus_indices=self._train_bmus, metric=None, **kwargs)
        
        self._inertia_               = np.sum(inertia)

        # Set n_iter_ attribute
        self._n_iter_                = global_iter_counter

        # Set trained flag
        self._trained                = True

        return

    def predict(self, X: np.ndarray, *args, metric: Optional[callable] = None, n_jobs: int = 1, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Riley Smith
        
        Modified by Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>.
        
        Predict cluster for each element in X.
        
        .. note::
            
            ***args** and **\**kwargs** are additional arguments and keyword arguments which can be passed depending on the metric used. In this implementation:
                  
            - ***args** must always be a collection of `ndarray`_ with shapes similar to that of **X**
            - **\**kwargs** are keyword arguments which have no constraints on their type or shape
             
            See the metric specific implementation for more details.

        :param X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :type X: `ndarray`_
        
        :param metric: (**Optional**) metric to use. If None, the metric provided at init is used.
        :type metric: :python:`callable`
        :param n_jobs: (**Optional**) number of threads used to find the BMUs. This parameter is only used when using :py:meth:`~.SOM._find_bmus_bydata` method.
        :type n_jobs: :python:`int`
        
        :param \*args: additional arguments to pass to the metric. These arguments are looped similarly to **X**, so they should be a collection of `ndarray`_ with the same shape. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.

        :returns: an ndarray of shape (n,). The predicted cluster index for each item in X.
        :rtype: `ndarray`_ [:python:`int`]
        
        :raises NotImplmentedError: if :py:meth:`~.SOM.fit` method has not been called already
        :raises ValueError:
            
        * if **X** is not a 2-dimensional array
        * if the second dimension of **X** has not a length equal to self.dim
        '''
        
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        if len(X.shape) != 2:
            raise ValueError(f'X should have two dimensions, not {len(X.shape)}.')
            
        if X.shape[1] != self.dim:
            raise ValueError(f'This SOM has dimension {self.dim}. Received input with dimension {X.shape[1]}.')
        
        return self._find_bmus(X, *args, metric=None, n_jobs=n_jobs, **kwargs)
     
    def _find_bmus(self, X: np.ndarray, *args, metric: Optional[callable] = None, n_jobs: int = 1, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Find the indices of the best matching unit for the input matrix X.
        
        .. note::
            
            ***args** and **\**kwargs** are additional arguments and keyword arguments which can be passed depending on the metric used. In this implementation:
                  
            - ***args** must always be a collection of `ndarray`_ with shapes similar to that of **X**
            - **\**kwargs** are keyword arguments which have no constraints on their type or shape
             
            See the metric specific implementation for more details.
        
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :param metric: (**Optional**) metric to use. If None, the metric provided at init is used.
        :type metric: :python:`callable`
        :param n_jobs: (**Optional**) number of threads used to find the BMUs. This parameter is only used when using :py:meth:`~.SOM._find_bmus_bydata` method.
        :type n_jobs: :python:`int`
        
        :param \*args: additional arguments to pass to the metric. These arguments are looped similarly to **X**, so they should be a collection of `ndarray`_ with the same shape. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        
        :returns: indices of the best matching units
        :rtype: `ndarray`_ [:python:`int`]
        '''
        
        if len(X) > (self.m*self.n)*n_jobs:
           labels = self._find_bmus_byweight(X, *args, metric=metric, **kwargs)
        else:
           labels = self._find_bmus_bydata(X, *args, metric=metric, n_jobs=n_jobs, **kwargs)
           
        return labels
    
    def _find_bmus_bydata(self, X: np.ndarray, *args, metric: Optional[callable] = None, n_jobs: int = 1, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Find the indices of the best matching unit for the input matrix X by looping through the data.
        
        .. note::
            
            ***args** and **\**kwargs** are additional arguments and keyword arguments which can be passed depending on the metric used. In this implementation:
                  
            - ***args** must always be a collection of `ndarray`_ with shapes similar to that of **X**
            - **\**kwargs** are keyword arguments which have no constraints on their type or shape
             
            See the metric specific implementation for more details.
            
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :param metric: (**Optional**) metric to use. If None, the metric provided at init is used.
        :type metric: :python:`callable`
        :param n_jobs: (**Optional**) number of threads used to find the BMUs
        :type n_jobs: :python:`int`
        
        :param \*args: additional arguments to pass to the metric. These arguments are looped similarly to **X**, so they should be a collection of `ndarray`_ with the same shape. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        
        :returns: indices of the best matching units
        :rtype: `ndarray`_ [:python:`int`]
        '''
        
        if not isinstance(n_jobs, int):
            raise TypeError(f'n_jobs parameter has type {type(n_jobs)} but it must be an int.')
          
        if n_jobs < 1 or n_jobs > N_JOBS_MAX:
            print(f'{colorama.Fore.ORANGE}Warning:{colorama.Style.RESET_ALL} n_jobs must be between 1 and {N_JOBS_MAX} on your computer. Setting to default value equal to 1...')
            n_jobs  = 1
         
        return joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self._find_bmu)(X[idx], *[arg[idx] for arg in args], **kwargs) for idx in range(len(X)))
    
    def _find_bmus_byweight(self, X: np.ndarray, *args, metric: Optional[callable] = None, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Find the indices of the best matching unit for the input matrix X by looping through the weights.
        
        .. note::
            
            ***args** and **\**kwargs** are additional arguments and keyword arguments which can be passed depending on the metric used. In this implementation:
                  
            - ***args** must always be a collection of `ndarray`_ with shapes similar to that of **X**
            - **\**kwargs** are keyword arguments which have no constraints on their type or shape
             
            See the metric specific implementation for more details.
            
        :param X: input matrix (2D)
        :type X: `ndarray`_
        
        :param metric: (**Optional**) metric to use. If None, the metric provided at init is used.
        :type metric: :python:`callable`
        
        :param \*args: additional arguments to pass to the metric. These arguments are looped similarly to **X**, so they should be a collection of `ndarray`_ with the same shape. See the metric specific signature to know which parameters to pass.
        :param \**kwargs: additional keyword arguments to pass to the metric. See the metric specific signature to know which parameters to pass.
        
        :returns: indices of the best matching units
        :rtype: `ndarray`_ [:python:`int`]
        '''
        
        metric            = metric if metric is not None else self.metric
        
        # Output indices set to 0 by default
        indices           = np.zeros(len(X), dtype=int)
        
        # First, compute distance
        dist              = metric(X, self.weights[0], *args, squared=True, axis=1, **kwargs)
        
        # Only update weight position if distance is less than the previous one
        for pos, weight in enumerate(self.weights[1:]):
            
            tmp           = metric(X, weight, *args, squared=True, axis=1, **kwargs)
            mask          = tmp < dist
            indices[mask] = pos+1
            dist[mask]    = tmp[mask]
            
        return indices
    
    def transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Riley Smith
        
        Transform the data X into cluster distance space.
        
        .. warning::
           
           This method has not been updated accordingly with other updates. It may not work as expected.

        :param X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :type X: `ndarray`_

        :returns: tansformed data of shape (n, self.n*self.m). The Euclidean distance from each item in X to each cluster center.
        :rtype: `ndarray`_ [:python:`float`]
        '''
        
        # Stack data and cluster centers
        X_stack       = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff          = X_stack - cluster_stack

        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Riley Smith
        
        Convenience method for calling :py:meth:`~.SOM.fit` followed by :py:meth:`~.SOM.predict`.
        
        .. warning::
           
           This method has not been updated accordingly with other updates. It may not work as expected.

        :param X: data of shape (n, self.dim). The data to fit and then predict.
        :type X: `ndarray`_
        
        :param \*args: optional arguments for the :py:meth:`~.SOM.fit` method
        :param \**kwargs: optional keyword arguments for the :py:meth:`~.SOM.fit` method

        :returns: ndarray of shape (n,). The index of the predicted cluster for each item in X (after fitting the SOM to the data in X).
        :rtype: `ndarray`_ [:python:`float`]
        '''
        
        # Fit to data
        self.fit(X, *args, **kwargs)

        # Return predictions
        return self.predict(X, *args, **kwargs)

    def fit_transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        r'''
        .. codeauthor:: Riley Smith
        
        Convenience method for calling :py:meth:`~.SOM.fit` followed by :py:meth:`~.SOM.transform`. Unlike in sklearn, this is not implemented more efficiently (the efficiency is the same as calling :py:meth:`~.SOM.fit` directly followed by :py:meth:`~.SOM.transform`).

        .. warning::
           
           This method has not been updated accordingly with other updates. It may not work as expected.
           
        :param X: data of shape (n, self.dim) where n is the number of samples
        :type X: `ndarray`_
        
        :param \*args: optional arguments for the :py:meth:`~.SOM.fit` method
        :param \**kwargs: optional keyword arguments for the :py:meth:`~.SOM.fit` method

        :returns: ndarray of shape (n, self.m*self.n). The Euclidean distance from each item in **X** to each cluster center.
        :rtype: ndarray[:python:`float`]
        '''
        
        # Fit to data
        self.fit(X, *args, **kwargs)

        # Return points in cluster distance space
        return self.transform(X, *args, **kwargs)
    
    ######################################
    #             IO methods             #
    ######################################
    
    def write(self, fname: str, *args, **kwargs) -> None:
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Write the result of the SOM into a binary file.
        
        :param fname: output filename
        :type fname: :python:`str`
        
        :param \*args: other arguments passed to `pickle.dump`_
        :parma \**kwargs: other keyword arguments passed to `pickle.dump`_
        
        :raises TypeError: if **fname** is not of type :python:`str`
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'fname is of type {type(fname)} but it must be of type str.')
        
        with open(fname, 'wb') as f:
            pickle.dump(self, f, *args, **kwargs)
            
        return
    
    @staticmethod
    def read(fname: str, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Read the result of a SOM written into a binary file with the :py:meth:`~.SOM.write` method.
        
        :param fname: input file
        :type fname: :python:`str`
        
        :param \*args: other arguments passed to `pickle.load`_
        :parma \**kwargs: other keyword arguments passed to `pickle.load`_
        
        :returns: the loaded SOM object
        :rtype: :py:class:`~.SOM`
        
        :raises TypeError: if **fname** is not of type :python:`str`
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'fname is of type {type(fname)} but it must be of type str.')
            
        with open(fname, 'rb') as data:
            return pickle.load(data, *args, **kwargs)
    
    #################################################
    #          Physical parameters methods          #
    #################################################
    
    def get(self, param: str) -> np.ndarray:
        r'''
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
        r'''
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
        r'''
        .. codeauthor:: Riley Smith
        
        Give the coordinates of each cluster centre as an array of shape (m, n, dim).
        
        :returns: cluster centres
        :rtype: `ndarray`_ [:python:`int`]
        '''
        
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self) -> np.ndarray:
        r'''
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
        r'''
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
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Best matching units indices for the train set.
        
        :returns: BMUs indices for the train set
        :rtype: `ndarray`_ [:python:`int`]
        
        :rtype AttributeError: if the number of iterations is not initialised yet
        '''
        
        if self._train_bmus is None:
            raise AttributeError('SOM does have train_bmus_ attribute until after calling fit().')
            
        return self._train_bmus

