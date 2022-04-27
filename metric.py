#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r'''
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Functions defining the metric used to find the BMU.

To write a custom metric, the following signature must be used:
   
.. code::
   
   def metric(coord1: np.ndarray, coord2: np.ndarray, *args, squared: bool = False, axis: int = 0, **kwargs) -> float:
      
Note that in the above signature, ** *args ** should always correspond to iterable arguments with the same dimension as **coord1** (e.g. uncertainties). On the other hand, ** **kwargs ** correspond to additional arguments which do not have to have the same shape as **coord1**.

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
    
       d = \sqrt{\Sum_i (a_i - b_i)^2}
    
    :param coord1: first array of coordinates
    :type coord1: `ndarray`_
    :param coord2: second array of coordinates
    :type coord2: python:`int`, python:`float` or `ndarray`_ [python:`float`]
    
    :param squared: (**Optional**) whether to return the square of the metric or not
    :type squared: :python:`bool`
    :param axis: (**Optional**) axis onto which to compute the sum
    :type axis: python:`int`
    
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
    
    Provide a chi2 distance estimated between **coord1** and **coord2** using an uncertainty given by **error**. The chi2 distance :math:`d` between coordinates :math:`a = (a_i)` and :math:`b = (b_i)` with error :math:`e = (e_i)` is given by
    
    .. math::
    
       d = \sqrt{\Sum_i \left ( \frac{a_i - b_i}{e_i} \right )^2}
       
    .. note::
       
       If one of the coordinates in **error** is 0, the chi2 distance will diverge.
    
    :param coord1: first array of coordinates
    :type coord1: `ndarray`_
    :param coord2: second array of coordinates
    :type coord2: python:`int`, python:`float` or `ndarray`_ [python:`float`]
    :param error: array of uncertainties. Must have the same shape as **coord1**. To provide no error, set **no_error** to python:`True`. Note that using :python:`error=1` is similar to using the Euclidian distance.
    :type error: `ndarray`_ [python:`float`]
    
    :param squared: (**Optional**) whether to return the square of the metric or not
    :type squared: :python:`bool`
    :param axis: (**Optional**) axis onto which to compute the sum
    :type axis: python:`int`
    :param no_error: (**Optional**) whether to use no error in the computation (i.e. Euclidian distance or not)
    :type no_error: python:`bool`
    
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