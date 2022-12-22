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
- :py:func:`~.chi2MetricPenalised`
- :py:func:`~.chi2CigaleMetric`
- :py:func:`~.chi2CigaleMetricPenalised`

For instance, one can do

.. code::

    from SOMptimised import SOM
    from SOMptimised import LinearLearningStrategy, ExponentialLearningStrategy, LearningStrategy
    from SOMptimised import ConstantRadiusStrategy, ExponentialRadiusStrategy, NeighbourhoodStrategy
    from SOMptimised import euclidianMetric, chi2Metric, chi2CigaleMetric, chi2MetricPenalised, chi2CigaleMetricPenalised

.. toctree::

    lr.rst
    metric.rst
    neighbourhood.rst
    som.rst