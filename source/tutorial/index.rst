Tutorial/Example
================

.. _iris dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set

.. important::

    This example requires pandas_ to be installed. This can be done the following way with pip
    
    .. code:: bash
        
        pip install pandas
        
    or with conda
    
    .. code:: bash
    
        conda install pandas

Preparing the data
##################

This SOM implementation requires the data to be given as a 2-dimensional numpy array where the first dimension corresponds to the observations or data points that you have and the second dimension corresponds to the features for each observation.

Let us run the SOM on the `iris dataset`_. To do so we will use `pandas`_ to load the dataset contained in the csv file

.. execute_code::
    
    import pandas
    
    table = pandas.read_csv('examples/iris_dataset/iris_dataset.csv')
    print(table.head(), end='\n\n')
    print(table.info())
    
Each line represents an observation and in this case we have 4 features: sepal length, sepal width, petal length and petal width. Let us train the SOM on three features

.. code::

    table  = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
    target = table['target']
    swidth = table['sepal width (cm)']
    print(table.info())

.. execute_code::
    :hide_code:

    import pandas
    
    table  = pandas.read_csv('examples/iris_dataset/iris_dataset.csv')
    target = table['target']
    swidth = table['sepal width (cm)']
    table  = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
    print(table.info())
    
where we have also extracted the target class for each observation to compare at the end and the sepal width to predict for the next section. We convert the data into a numpy array since this is the required format for the SOM to run and we check it has the correct shape (150, 3)

.. code::
    
    data  = table.to_numpy()
    print(data.shape)
    print(data[:5])
    
.. execute_code::
    :hide_code:

    import pandas
    
    table  = pandas.read_csv('examples/iris_dataset/iris_dataset.csv')
    target = table['target']
    swidth = table['sepal width (cm)']
    table  = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
    data   = table.to_numpy()
    print(data.shape)
    print(data[:5])
    
Finally, let us keep the last five observations apart to test the SOM

.. code::

    data_train = data[:-5]
    data_test  = data[-5:]
    print(len(data_train), len(data_test))

.. execute_code::
    :hide_code:
    
    import pandas
    
    table  = pandas.read_csv('examples/iris_dataset/iris_dataset.csv')
    target = table['target']
    swidth = table['sepal width (cm)']
    table  = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
    data   = table.to_numpy()
    
    data_train = data[:-5]
    data_test  = data[-5:]
    print(len(data_train), len(data_test))

.. include:: clustering_iris.rst
.. include:: predict_sepal_width.rst
.. include:: normalisation.rst
.. include:: set_and_get.rst
.. include:: save_and_load.rst