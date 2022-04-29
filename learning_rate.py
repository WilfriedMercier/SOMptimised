#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Strategies for the learning rate evolution which can be used in the SOM.

How to include a custom learning strategy ?
-------------------------------------------

To implement a custom learning strategy, please inherit from :py:class:`~.LearningStrategy` and implement your own :python:`LearningStrategy.__init__` and :python:`LearningStrategy.__call__` methods as follows

.. code::
    
   class NewLearningStrategy(LearningStrategy):
        
      def __init__(self, lr: Union[int, float]=1, **kwargs) -> None:
      
         super().__init__(lr)
         
         ...
       
      def __call__(step, *args, **kwargs) -> float:
         
         ...
          
         return learning_rate

The :python:`LearningStrategy.__call__` method must always have **step** as its first argument.

.. note::
   
   Additional arguments with ***args** and **\**kwargs** can be present in :python:`LearningStrategy.__call__` method but should not be used.

By default, the :python:`super().__init__(lr)` line will initialise the initial learning rate (:python:`self.initial_lr`) and the total number of iterations. (:python:`self.ntot`). Note that the SOM will automatically set :python:`self.ntot` when starting the fitting procedure. So if your strategy requires to know the total number of iterations, you can directly use :python:`self.ntot` without having to set its value by hand as the SOM will do it for you.

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

class LearningStrategy(ABC):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Abstract class for learning rate strategies.
   
   :param lr: (**Optional**) intial learning rate
   :type lr: :python:`int` or :python:`float`
   '''
   
   def __init__(self, lr: Union[int, float] = 1):
       r'''Init method.'''
       
       self._check_lr(lr, name='lr')
       self._initial_lr = lr
       self._ntot       = None
   
   @abstractmethod
   def __call__(self, step: int, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the learning rate at the given step.
      
      :param step: time step during the learning process
      :type step: :python:`int`
      
      :returns: Learning rate at the given time step
      :rtype: :python:`float`
      '''
      
      return
   
   #########################################
   #          Setters and getters          #
   #########################################
   
   @property
   def initial_lr(self, *args, **kwargs) -> Union[int, float]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the initial learning rate.
      
      :returns: Initial learning rate
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
      
      self._check_lr(value, name='lr')
      self._initial_lr = value
      return
   
   @property
   def ntot(self, *args, **kwargs) -> int:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the total number of iterations.
      
      :returns: Total number of iterations
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
      
      # Check for lr is the same as check for ntot
      self._check_lr(value, name='ntot')
      self._ntot = int(value)
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
   
   def _check_lr(self, value, name='lr', **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Check if a value is acceptable for the learning rate, that is, either a positive int or a positive float.
      
      :param value: value to check
      
      :param name: (**Optional**) name of the parameter
      :type: :python:`str`
      
      :raises TypeError: if :python:`not isinstance(value, (int, float))`
      :raises ValueError: if :python:`lr < 0`
      '''
      
      self._check_int_float(value, name, **kwargs)
      self._check_negative( value, name, **kwargs)
      return
        
      
class LinearLearningStrategy(LearningStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing a linear learning rate strategy given by
   
   .. math::
      
      \eta \times \left ( 1 - t / t_\rm{tot} \right ),
   
   where :math:`\eta` is the initial learning rate, :math:`t` is the time step and :math:`t_{\rm{tot}}` is the total number of iterations during the learning process.
   
   :param lr: (**Optional**) intial learning rate
   :type lr: :python:`int` or :python:`float`
   :param ntot: (**Optional**) maximum number of iterations
   :type ntot: :python:`int`
   '''
   
   def __init__(self, lr: Union[int, float]=1, **kwargs) -> None:
      r'''Init method.'''
      
      super().__init__(lr)
      
   def __call__(self, step: int) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the learning rate at the given step.
      
      :param step: iteration step
      :type step: :python:`int`
      
      :returns: Learning rate at the given step
      :rtype: :python:`float`
      '''
      
      if self._ntot is None:
         raise ValueError('ntot must be set before the learning rate can be computed.')
      
      if step < 0 or step > self.ntot:
         raise ValueError('step must be between 0 and ntot.')
      
      return (1 - step/self._ntot) * self._initial_lr
      

class ExponentialLearningStrategy(LearningStrategy):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Class implementing an exponential learning rate strategy given by
   
   .. math::
      
      \eta \times \exp \left \lbrace - t / \tau \right \rbrace,
   
   where :math:`\eta` is the initial learning rate, :math:`t` is the time step and :math:`\tau` is the decay time scale.
   
   :param lr: (**Optional**) intial learning rate
   :type lr: :python:`float`
   :param tau: (**Optional**) decay time scale
   :type tau: :python:`float`
   '''
   
   def __init__(self, lr: Union[int, float]=1, tau: Union[int, float]=1, **kwargs) -> None:
      r'''Init method.'''
      
      super().__init__(lr)
      
      self.tau = tau
      
   def __call__(self, step: int, *args, **kwargs) -> float:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the learning rate at the given step.
      
      :param step: iteration step
      :type step: :python:`int`
      '''
      
      if step < 0:
         raise ValueError('step must be greater than 0.')
      
      return np.exp(-step/self.tau) * self.initial_lr
      
   #########################################
   #          Setters and getters          #
   #########################################
   
   @property
   def tau(self, *args, **kwargs) -> Union[int, float]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Provide the value of the decay time scale.
      
      :returns: Decay time scale
      :rtype: :python:`int` or :python:`float`
      '''
      
      return self._tau
   
   @tau.setter
   def tau(self, value: Union[int, float], *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Set the value of the decay time scale.
      
      :param value: New decay time scale
      :type value: :python:`int` or :python:`float`
      '''
      
      self._check_lr(value, name='tau')
      self._tau = value
      return