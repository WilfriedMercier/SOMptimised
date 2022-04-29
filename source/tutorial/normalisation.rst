Of the importance of normalisation
##################################

This SOM implementation will initialise by default the weight vectors or the neurons to random values using a Gaussian distribution with zero mean and unity standard deviation. This means that the weight vectors will have values mostly between -1 and 1.

If the train and test data have values which are not normalised, that is with zero mean and a standard deviation of one, there is a chance that the fitting procedure will, in the best case, yield unsatisfactory resulst, and in the worst case, completely fail.

To solve this issue, it can be beneficial to normalise the data beforehand.

.. note::

   The normalisation of the data will will affect the fitting procedure in various ways. Weight vectors initialised far from the data will:
   
   - not train in a similar fashion as if they were intialised close to the data
   - require more steps to converge
   - potentially not converge enough since the convergence of the neurons is reduced as the training phase proceeds
   
As an example, we will do the clustering on the iris data set once more, but this time with the train and test data already scaled