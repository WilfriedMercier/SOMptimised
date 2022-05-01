import matplotlib.pyplot   as     plt
from   matplotlib.colors   import TwoSlopeNorm
from   matplotlib.gridspec import GridSpec
from   matplotlib          import rc
import matplotlib          as     mpl

from   SOMptimised         import SOM, LinearLearningStrategy, ConstantRadiusStrategy, euclidianMetric
import pandas

table  = pandas.read_csv('../../examples/iris_dataset/iris_dataset.csv')
target = table['target']
swidth = table['sepal width (cm)']
table  = table[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']]
data   = table.to_numpy()

data_train = data[:-5]
data_test  = data[-5:]

lr     = LinearLearningStrategy(lr=1)    # Learning rate strategy
sigma  = ConstantRadiusStrategy(sigma=0.8) # Neighbourhood radius strategy
metric = euclidianMetric                 # Metric used to compute BMUs
nf     = data_train.shape[1]             # Number of features
som    = SOM(m=1, n=3, dim=nf, lr=lr, sigma=sigma, max_iter=1e4, random_state=None)
som.fit(data_train, epochs=1, shuffle=True, n_jobs=1)

pred_train = som.train_bmus_
pred_test  = som.predict(data_test)

norm = TwoSlopeNorm(1, vmin=0, vmax=2)

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

f   = plt.figure(figsize=(10, 4.5))
gs  = GridSpec(1, 2, wspace=0)
ax1 = f.add_subplot(gs[0])
ax2 = f.add_subplot(gs[1])

for ax in [ax1, ax2]:
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='x', which='both', direction='in', labelsize=13, length=3)
    ax.tick_params(axis='y', which='both', direction='in', labelsize=13, length=3)
    ax.set_xlabel('Petal width (cm)', size=16)

ax1.scatter(data_train[:, 1], data_train[:, 0], c=pred_train, cmap='bwr', ec='k', norm=norm, marker='o', s=30)
ax1.scatter(data_test[:, 1],  data_test[:, 0],  c=pred_test,  cmap='bwr', ec='k', marker='o', norm=norm, s=60)
ax1.set_ylabel('Petal length (cm)', size=16)

target                              = target.to_numpy()
target[target == 'Iris-setosa']     = 0
target[target == 'Iris-versicolor'] = 1
target[target == 'Iris-virginica']  = 2

ax2.scatter(data_train[:, 1], data_train[:, 0], c=target[:-5], cmap='bwr', ec='k', norm=norm, marker='o', s=30)
ax2.scatter(data_test[:, 1],  data_test[:, 0],  c=target[-5:],  cmap='bwr', ec='k', marker='o', norm=norm, s=60)
ax2.set_yticks([0.2])
ax2.set_yticklabels([])

ax1.set_title('SOM clustering', size=18)
ax2.set_title('IRIS dataset', size=18)

plt.show()