API
===

The following classes and functions can be directly loaded from SOMptimised

- :py:class:`~.SOM`
- :py:class:`~.LinearLearningStrategy`
- :py:class:`~.ExponentialLearningStrategy`
- :py:class:`~.LearningStrategy`
- :py:class:`~.ConstantRadiusStrategy`
- :py:class:`~.ExponentialRadiusStrategy`
- :py:class:`~.NeighbourhoodStrategy`
- :py:func:`~.euclidianMetric`
- :py:func:`~.chi2Metric`
- :py:func:`~.chi2CigaleMetric`

For instance, one can do

.. code::

    from SOMptimised import SOM
    from SOMptimised import LinearLearningStrategy, ExponentialLearningStrategy, LearningStrategy
    from SOMptimised import ConstantRadiusStrategy, ExponentialRadiusStrategy, NeighbourhoodStrategy
    from SOMptimised import euclidianMetric, chi2Metric, chi2CigaleMetric

.. toctree::

    lr
    metric
    neighbourhood
    som