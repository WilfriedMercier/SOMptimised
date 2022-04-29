Save and load SOM
#################

A last feature of this SOM implementation is that the it can be saved at any moment and loaded back again. This is easily done using the :py:meth:`~.SOM.write` method

.. code::

    som.write('som_save')
    
The SOM can be loaded back at any time later on using the :py:meth:`~.SOM.read` method

.. code::
    
    newsom = SOM.read('som_save')
    
    print('BMUs for train data:')
    print(som.train_bmus_)
    print('\nBMUs for train data loaded from file:')
    print(newsom.train_bmus_)
    
.. execute_code::
    :hide_code:

    from   SOMptimised import SOM, LinearLearningStrategy, ConstantRadiusStrategy, euclidianMetric
    import pandas
    
    table      = pandas.read_csv('examples/iris_dataset/iris_dataset.csv')
    target     = table['target']
    swidth     = table['sepal width (cm)']
    table      = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
    data       = table.to_numpy()
    
    data_train = data[:-10]
    data_test  = data[-10:]
    
    lr         = LinearLearningStrategy(lr=1)
    sigma      = ConstantRadiusStrategy(sigma=0.8)
    metric     = euclidianMetric
    nf         = data_train.shape[1] # Number of features
    
    som        = SOM(m=1, n=3, dim=nf, lr=lr, sigma=sigma, metric=metric, max_iter=1e4, random_state=0)
    som.fit(data_train, epochs=1, shuffle=False, n_jobs=1)
    
    som.write('som_save')
    
    newsom     = SOM.read('som_save')
    
    print('BMUs for train data:')
    print(som.train_bmus_)
    print('\nBMUs for train data loaded from file:')
    print(newsom.train_bmus_)