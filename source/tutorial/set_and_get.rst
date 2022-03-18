Set and get extra parameters
############################

In the previous section we computed an additional parameter for each neuron in the SOM: the petal width. Rather than keeping such parameters in separate variables, we can integrate them into the SOM to easily access them

.. code::

    som.set('petal width', swidth_med)
    som.set('petal width uncertainty', swidth_std)
    
To retrieve later on an extra parameter (e.g. petal width), then one can do

.. code::

    pwidth = som.get('petal width')
    print(pdwidth)
    
.. execute_code::
    :hide_code:
    
    import warnings
    import pandas
    from   SOMptimised import SOM
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
    m   = 5
    n   = 5
    nf  = data_train.shape[1] # Number of features
    som = SOM(m=m, n=n, dim=nf, lr=1, sigma=1, max_iter=1e4, random_state=None)
    som.fit(data_train, epochs=1, shuffle=True)

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
    print(pwidth)