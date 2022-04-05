#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Classes defining the behaviour of the BMU neighbourhood.

.. The MIT License (MIT)

    Copyright © 2022 <copyright holders>

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
   
   Protocol class for strategies implementing neighbourhood radii strategies.
   '''
   
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
      
class ConstantRadiusStrategy(NeighbourhoodStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing a constant neighbourhood radius strategy.
   
   :param sigma: (**Optional**) neighbourhood radius
   :type sigma: :python:`int` or :python:`float`
   '''
   
   def __init__(self, sigma: Union[int, float]=1, **kwargs) -> None:
      r'''Init method.'''
      
      self._check_sigma(sigma)
      
      super().__init__(**kwargs)
      
      # It is convenient to have the squared version as well because this is what the SOM really uses
      self._sigma  = sigma
      self._sigma2 = sigma*sigma
      
   def __call__(self, step: int, squared: bool = False, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the neighbourhood radius at the given time step.
      
      .. note::
         
         Since this strategy provides the same value at each iteration step, the step parameter is not mandatory.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :param squared: (**Optional**) whether to provide the raduis squared or not
      :type squared: :python:`bool`
      
      :returns: neighbourhood radius
      :rtype: :python:`float`
      '''
      
      return self._sigma2 if squared else self._sigma
      
   ####################################
   #          Check methods           #
   ####################################
   
   def _check_sigma(self, sigma: Any, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a value is acceptable for the neighbourhood radius.
      
      :param sigma: value to check
      
      :raises TypeError: if **sigma** is neither :python:`int`, nor :python:`float`
      :raises ValueError: if :python:`sigma < 0`
      '''
      
      if not isinstance(sigma, (int, float)):
         raise TypeError(f'sigma has type {type(sigma)} but it must be an int or a float.')
         
      if sigma < 0:
         raise ValueError('sigma must be positive.')
         
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
      
      self._check_sigma(value)
      self._sigma  = value
      self._sigma2 = value*value
      return
   
   
class ExponentialRadiusStrategy(NeighbourhoodStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing an exponential neighbourhood radius strategy.
   
   :param sigma: (**Optional**) neighbourhood radius
   :type sigma: :python:`int` or :python:`float`
   :param tau: (**Optional**) decay time scale
   :type tau: :python:`int` or :python:`float`
   '''
   
   def __init__(self, sigma: Union[int, float]=1, tau: Union[int, float]=1, **kwargs):
      r'''Init method.'''
      
      self._check_sigma_tau(sigma)
      self._check_sigma_tau(tau)
      
      super().__init__(**kwargs)
      
      # It is convenient to have the squared version as well because this is what the SOM really uses
      self._sigma  = sigma
      self._sigma2 = sigma*sigma
      self._tau    = tau
      
   def __call__(self, step: int, squared: bool = False, *args, **kwargs):
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the neighbourhood radius at the given time step.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :param squared: (**Optional**) whether to provide the raduis squared or not
      :type squared: :python:`bool`
      
      :returns: neighbourhood radius
      :rtype: :python:`float`
      '''
      
      if step < 0:
         raise ValueError('step must be larger than 0.')
      
      return np.exp(-2*step/self._tau) * self._sigma2 if squared else np.exp(-step/self._tau) * self._sigma
      
   ####################################
   #          Check methods           #
   ####################################
   
   def _check_sigma_tau(self, value: Any, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a value is acceptable for the neighbourhood radius and decay time scale.
      
      :param value: value to check
      
      :raises TypeError: if **value** is neither :python:`int`, nor :python:`float`
      :raises ValueError: if :python:`value < 0`
      '''
      
      if not isinstance(value, (int, float)):
         raise TypeError('sigma and tau must be int or float.')
         
      if value < 0:
         raise ValueError('sigma and tau must be positive.')
         
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
      
      self._check_sigma_tau(value)
      self._sigma  = value
      self._sigma2 = value*value
      return
      
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
      
      self._check_sigma_tau(value)
      self._tau = value
      return
      