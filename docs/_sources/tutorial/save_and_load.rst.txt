Save and load SOM
#################

A last feature of this SOM implementation is that the it can be saved at any moment and loaded back again. This is easily done using the :py:meth:`~.SOM.write` method

.. code::

    som.write('som_save')
    
The SOM can be loaded back at any time later on using the :py:meth:`~.SOM.read` method

.. code::
    
    newsom = SOM.read('som_save')
    
    print(som.train_bmus_)
    print(newsom.train_bmus_)
    
.. execute_code::
    :hide_code:

    from   SOMptimised import SOM
    import pandas
    
    table      = pandas.read_csv('examples/iris_dataset/iris_dataset.csv')
    target     = table['target']
    swidth     = table['sepal width (cm)']
    table      = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
    data       = table.to_numpy()
    
    data_train = data[:-10]
    data_test  = data[-10:]
    
    nf  = data_train.shape[1] # Number of features
    som = SOM(m=1, n=3, dim=nf, lr=1, sigma=1, max_iter=1e4, random_state=None)
    som.fit(data_train, epochs=1, shuffle=True)
    
    som.write('som_save')
    
    newsom = SOM.read('som_save')
    
    print(som.train_bmus_)
    print(newsom.train_bmus_)