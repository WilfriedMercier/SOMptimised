Predict sepal width
###################

We can also use the SOM to predict additional features of the test dataset. To do so, we first need to assign new features to the neurons of the SOM.

Let us predict the sepal width of the test dataset using our SOM. To do so we will train a new SOM with more neurons to have finer predictions

.. code::

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
    
Here we used a 5*5 SOM to fit the data. A larger SOM might give more precise results but some neurons might never map to any data point though. 

We also extract the sepal width column of the train and test datasets. The sepal width for the test dataset will be used to compare with the predicition from the SOM. That of the train dataset will be used to assign a sepal width for each neuron in the SOM.

To compute a sepal width estimate for each neuron, we can loop through them and find all data points whose best-matching unit is that neuron. The sepal width of the neuron can then be computed as the median value of the sepal width of all these data points

.. code::
    
    # Compute median sepal width and uncertainty for all neurons
    swidth_med = []
    swidth_std = []
    
    for i in range(m*n):
        tmp    = swidth_train[pred_train == i]
        
        swidth_med.append(np.nanmedian(tmp))
        swidth_std.append(np.nanstd(tmp))
    
In the code above, we also computed an estimate of the uncertainty on the sepal width as the standard deviation. Lets us predict the sepal width of the test set using the SOM

.. code::

    # Predict sepal width for test set
    swidth_test_pred     = np.array(swidth_med)[pred_test]
    swidth_test_pred_std = np.array(swidth_std)[pred_test]
    
    print('Predicted     Real')
    for pred, err, true in zip(swidth_test_pred, swidth_test_pred_std, swidth_test):
        print(f'{pred:.1f} +- {err:.1f}    {true:.1f}')
        
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
            
    # Predict sepal width for test set
    swidth_test_pred     = np.array(swidth_med)[pred_test]
    swidth_test_pred_std = np.array(swidth_std)[pred_test]


    print('Predicted     Real')
    for pred, err, true in zip(swidth_test_pred, swidth_test_pred_std, swidth_test):
        print(f'{pred:.1f} +- {err:.1f}    {true:.1f}')
        
Depending on the parameters of the SOM and the initialisation of the weights, it is not possible to predict a sepal width for all the data points in the test dataset.