Set and get extra parameters
############################

In the previous section we computed an additional parameter for each neuron in the SOM: the petal width. Rather than keeping such parameters in separate variables, we can integrate them into the SOM to easily access them later on.

This is done with the :py:meth:`~.SOM.set` method

.. code::

    som.set('petal width', swidth_med)
    som.set('petal width uncertainty', swidth_std)
    
To retrieve later on an extra parameter (e.g. petal width), then one can use the :py:meth:`~.SOM.get` method

.. code::

    pwidth = som.get('petal width')
    
    print('Petal width for the SOM neurons:')
    print(swidth_med)
    print('\nPetal width stored in the SOM:')
    print(pwidth)
    
.. execute_code::
    :hide_code:
    
    import warnings
    import pandas
    from   SOMptimised import SOM, LinearLearningStrategy, ConstantRadiusStrategy, euclidianMetric
    import numpy       as np

    # Extract data
    table        = pandas.read_csv('examples/iris_dataset/iris_dataset.csv').sample(frac=1)
    swidth       = table['sepal width (cm)'].to_numpy()
    data         = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']].to_numpy()

    data_train   = data[:-10]
    data_test    = data[-10:]
    swidth_train = swidth[:-10]
    swidth_test  = swidth[-10:]

    # Fit SOM
    m      = 5
    n      = 5
    lr     = LinearLearningStrategy(lr=1)
    sigma  = ConstantRadiusStrategy(sigma=0.8)
    metric = euclidianMetric
    nf     = data_train.shape[1] # Number of features
    
    som    = SOM(m=m, n=n, dim=nf, lr=lr, sigma=sigma, metric=metric, max_iter=1e4, random_state=0)
    som.fit(data_train, epochs=1, shuffle=False, n_jobs=1)

    pred_train = som.train_bmus_
    pred_test  = som.predict(data_test)

    # Compute median sepal width and uncertainty
    swidth_med = []
    swidth_std = []

    for i in range(m*n):
        tmp    = swidth_train[pred_train == i]
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            swidth_med.append(np.nanmedian(tmp))
            swidth_std.append(np.nanstd(tmp))
            
    som.set('petal width', swidth_med)
    som.set('petal width uncertainty', swidth_std)
    pwidth = som.get('petal width')
    
    print('Petal width for the SOM neurons:')
    print(swidth_med)
    print('\nPetal width stored in the SOM:')
    print(pwidth)