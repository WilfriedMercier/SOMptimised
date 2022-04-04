#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Strategies for the learning rate evolution which can be used in the SOM.

.. The MIT License (MIT)

    Copyright © 2022 <copyright holders>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from   abc    import ABC, abstractmethod
from   typing import Union, Any
import numpy  as     np

class LearningStrategy(ABC):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Abstract class for learning rate strategies.
   '''
   
   @abstractmethod
   def __call__(self, step: int, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the learning rate at the given step.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :returns: learning rate at the given time step
      :rtype: :python:`float`
      '''
      
      return
      
class LinearLearningStrategy(LearningStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing a linear learning rate strategy given by
   
   .. math::
      
      \eta \times \left ( 1 - t / t_\rm{tot} \right ),
   
   where :math:`\eta` is the initial learning rate, :math:`t` is the time step and :math:`t_'{\rm{tot}}' is the total number of iterations during the learning process.
   
   :param lr: (**Optional**) intial learning rate
   :type lr: :python:`int` or :python:`float`
   :param ntot: (**Optional**) maximum number of iterations
   :type ntot: :python:`int`
   '''
   
   def __init__(self, lr: Union[int, float]=1, **kwargs) -> None:
      r'''Init method.'''
      
      self._check_lr(lr)
      
      super().__init__(**kwargs)
         
      self._initial_lr = lr
      self._ntot       = None
      
   def __call__(self, step: int) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the learning rate at the given step.
      
      :param step: iteration step
      :type step: :python:`int`
      
      :returns: learning rate at the given step
      :rtype: :python:`float`
      '''
      
      if self._ntot is None:
         raise ValueError('ntot must be set before the learning rate can be computed.')
      
      if step < 0 or step > self.ntot:
         raise ValueError('step must be between 0 and ntot.')
      
      return (1 - step/self._ntot) * self._initial_lr
   
   ####################################
   #          Check methods           #
   ####################################
   
   def _check_lr(self, lr: Any, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a value is acceptable for the learning rate.
      
      :param lr: value to check
      
      :raises TypeError: if **lr** is neither :python:`int`, nor :python:`float`
      :raises ValueError: if :python:`lr < 0`
      '''
      
      if not isinstance(lr, (int, float)):
         raise TypeError(f'lr has type {type(lr)} but it must be an int or a float.')
         
      if lr < 0:
         raise ValueError('learning rate must be positive.')
         
      return
   
   #########################################
   #          Setters and getters          #
   #########################################
   
   @property
   def initial_lr(self, *args, **kwargs) -> Union[int, float]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the initial learning rate.
      
      :returns: initial learning rate
      :rtype: :python:`int` or :python:`float`
      '''
      
      return self._initial_lr
   
   @initial_lr.setter
   def initial_lr(self, value: Union[int, float], *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of the initial learning rate.
      
      :param value: new initial learning rate
      :type value: :python:`int` or :python:`float`
      '''
      
      self._check_lr(value)
      self._initial_lr = value
      return
   
   @property
   def ntot(self, *args, **kwargs) -> int:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the total number of iterations.
      
      :returns: total number of iterations
      :rtype: :python:`int`
      '''
      
      return self._ntot
   
   @ntot.setter
   def ntot(self, value: int, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of the total number of iterations.
      
      :param value: new total number of iterations
      :type value: :python:`int`
      '''
      
      self._check_ntot(value)
      self._ntot = value
      return
      
class ExponentialLearningStrategy(LearningStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing an exponential learning rate strategy given by
   
   .. math::
      
      \eta \times \exp \left \lbrace - t / \tau \right \rbrace,
   
   where :math:`\eta` is the initial learning rate, :math:`t` is the time step and :math:`\tau' is the decay time scale.
   
   :param lr: (**Optional**) intial learning rate
   :type lr: :python:`float`
   :param tau: (**Optional**) decay time scale
   :type tau: :python:`float`
   '''
   
   def __init__(self, lr: Union[int, float]=1, tau: Union[int, float]=1, **kwargs) -> None:
      r'''Init method.'''
      
      self._check_lr_tau(lr)
      self._check_lr_tau(tau)
      
      super().__init__(**kwargs)
         
      self._initial_lr = lr
      self._tau        = tau
      
   def __call__(self, step: int, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the learning rate at the given step.
      
      :param step: iteration step
      :type step: :python:`int`
      '''
      
      if step < 0 or step > self.ntot:
         raise ValueError('step must be between 0 and ntot.')
      
      return np.exp(-step/self.tau) * self.initial_lr
      
   ####################################
   #          Check methods           #
   ####################################
   
   def _check_lr_tau(self, value: Any, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a value is acceptable for the learning rate and the decay time scale.
      
      :param value: value to check
      
      :raises TypeError: if **value** is neither :python:`int`, nor :python:`float`
      :raises ValueError: if :python:`value < 0`
      '''
      
      if not isinstance(value, (int, float)):
         raise TypeError('lr and tau must be int or float.')
         
      if value < 0:
         raise ValueError('learning rate and decay time scale must be positive.')
         
      return
   
   #########################################
   #          Setters and getters          #
   #########################################
   
   @property
   def initial_lr(self, *args, **kwargs) -> Union[int, float]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the initial learning rate.
      
      :returns: initial learning rate
      :rtype: :python:`int` or :python:`float`
      '''
      
      return self._initial_lr
   
   @initial_lr.setter
   def initial_lr(self, value: Union[int, float], *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of the initial learning rate.
      
      :param value: new initial learning rate
      :type value: :python:`int` or :python:`float`
      '''
      
      self._check_lr_tau(value)
      self._initial_lr = value
      return
   
   @property
   def tau(self, *args, **kwargs) -> Union[int, float]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the decay time scale.
      
      :returns: decay time scale
      :rtype: :python:`int` or :python:`float`
      '''
      
      return self._tau
   
   @tau.setter
   def tau(self, value: Union[int, float], *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of the decay time scale.
      
      :param value: new decay time scale
      :type value: :python:`int` or :python:`float`
      '''
      
      self._check_lr_tau(value)
      self._tau = value
      return
   
test = LinearLearningStrategy(lr=1)
print('_ntot' in test.__dict__)
