#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Classes defining the behaviour of the BMU neighbourhood.

How to implement a custom neighbourhood class ?
-----------------------------------------------

To implement a custom neighbourhood strategy, please inherit from :py:class:`~.NeighbourhoodStrategy` and implement your own :python:`NeighbourhoodStrategy.__init__` and :python:`NeighbourhoodStrategy.__call__` methods as follows

.. code::
    
    class NewNeighbourhoodStrategy(NeighbourhoodStrategy):
        
        def __init__(self, sigma: Union[int, float] = 1, **kwargs) -> None:
            
            super().__init__(sigma=sigma)
            
            ...
        
        def __call__(step, *args, **kwargs) -> float:
            
            ...
            
            return sigma
        
The :python:`NeighbourhoodStrategy.__call__` method must always have **step** as its first argument.

.. note::

    Additional arguments with ***args** and **\**kwargs** in :python:`NeighbourhoodStrategy.__call__` method can be present but should not be used.

API
---

.. The MIT License (MIT)

    Copyright © 2022 <Wilfried Mercier>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from   abc    import ABC, abstractmethod
from   typing import Union, Any
import numpy  as     np

class NeighbourhoodStrategy(ABC):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Abstract class for strategies implementing neighbourhood radii strategies.
   
   :param sigma: (**Optional**) neighbourhood radius
   :type sigma: :python:`int` or :python:`float`
   '''
   
   def __init__(self, sigma: Union[int, float] = 1, **kwargs) -> None:
      r'''Init method.'''
       
      self._check_sigma(sigma, name='sigma')
      
      # Initial sigma value
      self._sigma = sigma
   
   @abstractmethod
   def __call__(self, step: int, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the neighbourhood radius at the given step.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :returns: neighbourhood radius at the given time step
      :rtype: :python:`float`
      '''
      
      return

   #########################################
   #          Setters and getters          #
   #########################################
      
   @property
   def sigma(self, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of sigma.
      '''
      
      return self._sigma
   
   @sigma.setter
   def sigma(self, value: Union[int, float], *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of sigma.
         
      :param sigma: initial neighbourhood radius
      :type sigma: :python:`int` or :python:`float`
      '''
      
      self._check_sigma(value, name='sigma')
      self._sigma = value
      return
   
   #############################
   #       Check methods       #
   #############################
  
   @staticmethod
   def _check_int_float(param: Any, name: str, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a parameter is int or float.
      
      :param param: value to check
      :param name: name of the parameter
      :type: :python:`str`
      
      :raises TypeError: if :python:`not isinstance(param, (int, float))`
      '''
      
      if not isinstance(param, (int, float)):
         raise TypeError(f'{name} has type {type(param)} but it must be an int or a float.')
   
      return
   
   @staticmethod
   def _check_negative(param: Any, name: str, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a parameter is negative.
      
      :param param: value to check
      :param name: name of the parameter
      :type: :python:`str`
      
      :raises ValueError: if :python:`lr < 0`
      '''
      
      if param < 0:
         raise ValueError('{name} must be positive.')
         
      return
   
   def _check_sigma(self, value: Any, name: str = 'sigma', **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a value is acceptable for the neighbourhood radius.
      
      :param value: value to check
      
      :raises TypeError: if :python:`not isinstance(value, (int, float))`
      :raises ValueError: if :python:`value < 0`
      '''
      
      self._check_int_float(value, name, **kwargs)
      self._check_negative(value, name, **kwargs)
      return
      
class ConstantRadiusStrategy(NeighbourhoodStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing a constant neighbourhood radius strategy.
   
   :param sigma: (**Optional**) neighbourhood radius
   :type sigma: :python:`int` or :python:`float`
   '''
   
   def __init__(self, sigma: Union[int, float] = 1, **kwargs) -> None:
      r'''Init method.'''
      
      super().__init__(sigma=sigma)
      
   def __call__(self, step: int, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the neighbourhood radius at the given time step.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :returns: neighbourhood radius
      :rtype: :python:`float`
      '''
      
      return self.sigma
   
   
class ExponentialRadiusStrategy(NeighbourhoodStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing an exponential neighbourhood radius strategy.
   
   :param sigma: (**Optional**) neighbourhood radius
   :type sigma: :python:`int` or :python:`float`
   :param tau: (**Optional**) decay time scale
   :type tau: :python:`int` or :python:`float`
   '''
   
   def __init__(self, sigma: Union[int, float] = 1, tau: Union[int, float] = 1, **kwargs):
      r'''Init method.'''

      super().__init__(sigma=sigma)
      
      self.tau = tau
      
   def __call__(self, step: int, *args, **kwargs):
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the neighbourhood radius at the given time step.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :returns: neighbourhood radius
      :rtype: :python:`float`
      '''
      
      if step < 0:
         raise ValueError('step must be larger than 0.')
      
      return np.exp(-step/self._tau) * self._sigma
   
   #########################################
   #          Setters and getters          #
   #########################################
      
   @property
   def tau(self, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of tau.
      '''
      
      return self._tau
   
   @tau.setter
   def tau(self, value: Union[int, float], *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of tau.
         
      :param sigma: decay time scale
      :type sigma: :python:`int` or :python:`float`
      '''
      
      self._check_sigma(value, name='tau', **kwargs)
      self._tau = value
      return
      