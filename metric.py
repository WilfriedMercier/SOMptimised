#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r'''
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Functions defining the metric used to find the BMU.

How to include a custom metric ?
--------------------------------

To write a **custom metric**, the following signature must be used:
   
.. code::
   
   def custom_metric(coord1: np.ndarray, coord2: np.ndarray, *args, squared: bool = False, axis: int = 0, **kwargs) -> float:
      
Note that in the above signature, ***args** should always correspond to iterable arguments with the same dimension as **coord1** (e.g. uncertainties). On the other hand, **\**kwargs** correspond to additional arguments which do not have to have the same shape as **coord1**.

Additionally, the function must be able to return either the distance (if :python:`squared = False`) or its squared value (if :python:`squared = True`).

For instance, assume we want to define a new metric which takes into account errors which can be scaled by a given factor. Such a metric could be written as

.. code::
    
    def new_metric(coord1: np.ndarray, coord2: np.ndarray, errors: np.ndarray, *args, squared: bool = False, axis: int = 0, factor: float = 1.0, **kwargs):
        
        diff = (coord1 - coord2)/(error*factor)
        
        if squared:
            return np.sum(diff*diff, axis=axis)
        else:
            return np.sqrt(np.sum(diff*diff, axis=axis))
        
.. note::
    
    Even if not used, it is better to keep the ***args** and **\**kwargs** parameters in the metric declaration.
    
A note on normalisation
-----------------------

Depending on the metric used, the data may need to be normalised beforehand, and the SOM weight vectors may need to be un-normalised. 

The following metrics need normalised train and test data to give an optimal result:

- :py:func:`~.euclidianMetric`
- :py:func:`~.chi2Metric`

The following metrics need normalised train and test data **and un-normalised SOM initial weight vectors**:
   
- :py:func:`~.chi2CigaleMetric`
   
To un-normalise the initial values of the SOM weight vectors, the :python:`unormalise_weights=True` argument can be passed to the :py:meth:`~.SOM.fit` method of the SOM, for instance doing:
   
.. code:: python

   som = SOM(m, n, dim, lr=lr, sigma=sigma, metric=chi2CigaleMetric, max_iter=max_iter)
   som.fit(X, error, epochs=1, shuffle=True, n_jobs=1, unnormalise_weights=False)

API
---

.. The MIT License (MIT)

    Copyright © 2022 <Wilfried Mercier>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from   typing import Union
import numpy  as     np
  
def euclidianMetric(coord1: np.ndarray, coord2: Union[int, float, np.ndarray], *args,
                    squared: bool = False, 
                    axis: int     = 1, 
                    **kwargs) -> float:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Provide the euclidian distance estimated between **coord1** and **coord2**. The euclidian distance :math:`d` between coordinates :math:`a = (a_i)` and :math:`b = (b_i)` is given by
    
    .. math::
    
       d = \sqrt{\sum_i (a_i - b_i)^2}
    
    :param coord1: first array of coordinates
    :type coord1: `ndarray`_
    :param coord2: second array of coordinates
    :type coord2: :python:`int`, :python:`float` or `ndarray`_ [:python:`float`]
    
    :param axis: (**Optional**) axis onto which to compute the sum
    :type axis: :python:`int`
    
    :returns: euclidian distance estimated between the two sets of coordinates
    :rtype: :python:`float`
    
    :raises TypeError: if
    
    * :python:`not isinstance(coord1, np.ndarray)`
    * :python:`not isinstance(coord2, np.ndarray)`
    * :python:`not isinstance(squared, bool)`
    * :python:`not isinstance(axis, int)`
    '''
    
    for param, name, typ in zip([coord1, coord2, squared, axis], 
                                ['coord1', 'coord2', 'squared', 'axis'], 
                                [np.ndarray, (int, float, np.ndarray), bool, int]
                               ):
        
        if not isinstance(param, typ):
            raise TypeError(f'{name} has type {type(param)} but it must be {typ}.')
    
    diff = coord1 - coord2
    
    return np.sum(diff*diff, axis=axis) if squared else np.sqrt(np.sum(diff*diff, axis=axis))

def chi2Metric(coord1: np.ndarray, coord2: Union[int, float, np.ndarray], error: np.ndarray, *args,
               squared: bool  = False, 
               axis: int      = 1,
               no_error: bool = False,
               **kwargs) -> float:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Provide a :math:`\chi^2` distance estimated between **coord1** and **coord2** using an uncertainty given by **error**. The :math:`\chi^2` distance :math:`d` between coordinates :math:`a = (a_i)` and :math:`b = (b_i)` with error :math:`e = (e_i)` is given by
    
    .. math::
    
       d = \sqrt{\sum_i \left ( \frac{a_i - b_i}{e_i} \right )^2}
       
    .. note::
       
       If one of the coordinates in **error** is 0, the :math:`\chi^2` distance will diverge.
    
    :param coord1: first array of coordinates
    :type coord1: `ndarray`_
    :param coord2: second array of coordinates
    :type coord2: :python:`int`, :python:`float` or `ndarray`_ [:python:`float`]
    :param error: array of uncertainties. Must have the same shape as **coord1**. To provide no error, set **no_error** to :python:`True`.
    :type error: `ndarray`_ [:python:`float`]
    
    :param squared: (**Optional**) whether to return the square of the metric or not
    :type squared: :python:`bool`
    :param axis: (**Optional**) axis onto which to compute the sum
    :type axis: :python:`int`
    :param no_error: (**Optional**) whether to use no error in the computation (i.e. Euclidian distance or not)
    :type no_error: :python:`bool`
    
    :returns: euclidian distance estimated between the two sets of coordinates
    :rtype: :python:`float`
    
    :raises TypeError: if
    
    * :python:`not isinstance(no_error, bool)`
    * :python:`not isinstance(coord1, np.ndarray)`
    * :python:`not isinstance(coord2, np.ndarray)`
    * :python:`not isinstance(error, (np.ndarray, int, float))`
    * :python:`not isinstance(squared, bool)`
    * :python:`not isinstance(axis, int)`
    '''
    
    if not isinstance(no_error, bool):
       raise TypeError(f'no_error parameter has type {type(no_error)} but it must be a bool.')
    
    # If no error is provided, we set it to 1.0 (similar to computing the Euclidian distance)
    if no_error:
        error = np.full(coord1.shape, 1.0)
    
    # Check type for all parameters including the additional arguments
    for param, name, typ in zip([coord1, coord2, error, squared, axis], 
                                ['coord1', 'coord2', 'error', 'squared', 'axis'], 
                                [np.ndarray, (int, float, np.ndarray), (np.ndarray, float, int, ), bool, int]
                               ):
       
        if not isinstance(param, typ):
            raise TypeError(f'{name} has type {type(param)} but it must be {typ}.')
        
    if coord1.shape != error.shape:
        raise ValueError(f'coord1 parameter has shape {coord1.shape} and error parameter has shape {error.shape} when they should be similar.')
          
    chi2_all = (coord1 - coord2)/error
    
    return np.sum(chi2_all*chi2_all, axis=axis) if squared else np.sqrt(np.sum(chi2_all*chi2_all, axis=axis))

def chi2CigaleMetric(coord1: np.ndarray, coord2: Union[int, float, np.ndarray], error: np.ndarray, *args,
               squared: bool  = False, 
               axis: int      = 1,
               no_error: bool = False,
               **kwargs) -> float:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Provide a :math:`\chi^2` distance defined in `Cigale`_ estimated between **coord1** and **coord2** using an uncertainty given by **error**. The :math:`\chi^2` distance :math:`d` between coordinates :math:`a = (a_i)` and :math:`b = (b_i)` with error :math:`e = (e_i)` is given by
    
    .. math::
    
       d = \sqrt{\sum_i \left ( \frac{a_i - \alpha b_i}{e_i} \right )^2}
       
    where :math:`\alpha` is a scale factor which is computed by the function as
    
    .. math::
        
        \alpha = \frac{\sum_i a_i b_i / e_i^2}{\sum_i b_i^2 / e_i^2}
       
    .. note::
       
       If one of the coordinates in **error** is 0, the :math:`\chi^2` distance will diverge.
    
    :param coord1: first array of coordinates
    :type coord1: `ndarray`_
    :param coord2: second array of coordinates
    :type coord2: :python:`int`, :python:`float` or `ndarray`_ [:python:`float`]
    :param error: array of uncertainties. Must have the same shape as **coord1**. To provide no error, set **no_error** to :python:`True`.
    :type error: `ndarray`_ [:python:`float`]
    
    :param squared: (**Optional**) whether to return the square of the metric or not
    :type squared: :python:`bool`
    :param axis: (**Optional**) axis onto which to compute the sum
    :type axis: :python:`int`
    :param no_error: (**Optional**) whether to use no error in the computation (i.e. Euclidian distance or not)
    :type no_error: :python:`bool`
    
    :returns: euclidian distance estimated between the two sets of coordinates
    :rtype: :python:`float`
    
    :raises TypeError: if
    
    * :python:`not isinstance(no_error, bool)`
    * :python:`not isinstance(coord1, np.ndarray)`
    * :python:`not isinstance(coord2, np.ndarray)`
    * :python:`not isinstance(error, (np.ndarray, int, float))`
    * :python:`not isinstance(squared, bool)`
    * :python:`not isinstance(axis, int)`
    '''
    
    if not isinstance(no_error, bool):
       raise TypeError(f'no_error parameter has type {type(no_error)} but it must be a bool.')
    
    # If no error is provided, we set it to 1.0 (similar to computing the Euclidian distance)
    if no_error:
        error = np.full(coord1.shape, 1.0)
    
    # Check type for all parameters including the additional arguments
    for param, name, typ in zip([coord1, coord2, error, squared, axis], 
                                ['coord1', 'coord2', 'error', 'squared', 'axis'], 
                                [np.ndarray, (int, float, np.ndarray), (np.ndarray, float, int, ), bool, int]
                               ):
       
        if not isinstance(param, typ):
            raise TypeError(f'{name} has type {type(param)} but it must be {typ}.')
        
    if coord1.shape != error.shape:
        raise ValueError(f'coord1 parameter has shape {coord1.shape} and error parameter has shape {error.shape} when they should be similar.')
        
    error2   = error*error
    alpha    = np.array([np.sum(coord1*coord2/(error2), axis=1) / np.sum(coord2*coord2/error2, axis=1)]*error.shape[-1]).T  
    
    return chi2Metric(coord1, coord2*alpha, error, *args, squared=squared, axis=axis, no_error=no_error, **kwargs)