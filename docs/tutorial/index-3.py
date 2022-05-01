from   SOMptimised         import SOM, LinearLearningStrategy, ConstantRadiusStrategy, euclidianMetric
import matplotlib.pyplot   as     plt
from   matplotlib          import rc
from   matplotlib.gridspec import GridSpec
import numpy               as     np
from   numpy.random        import default_rng
rng = default_rng()

# Random x coordinates
x = []
x = np.concatenate((x, rng.normal(loc=0, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=10, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=-3, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=16, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=-6, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=-10, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=13, scale=1, size=100)))
x = np.concatenate((x, rng.normal(loc=-6, scale=1, size=100)))

# Random y coordinates
y = []
y = np.concatenate((y, rng.normal(loc=20, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=-10, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=-10, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=7, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=-10, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=0, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=-16, scale=1, size=100)))
y = np.concatenate((y, rng.normal(loc=-8, scale=1, size=100)))

# Combine coordinates to have an array of the correct shape for the SOM to train onto
data = np.array([x, y]).T

# Normalise each feature
data_norm = (data - np.nanmean(data, axis=0))/np.nanstd(data, axis=0)

# SOM
lr     = LinearLearningStrategy(lr=1)      # Learning rate strategy
sigma  = ConstantRadiusStrategy(sigma=0.1) # Neighbourhood radius strategy
metric = euclidianMetric                   # Metric used to compute BMUs
nf     = data.shape[1]                     # Number of features
som    = SOM(m=2, n=4, dim=nf, lr=lr, sigma=sigma, metric=metric, max_iter=1e4, random_state=0)
som.fit(data_norm, epochs=3, shuffle=True)

# Prediction for the train dataset
pred   = som.train_bmus_

# Plotting
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

f  = plt.figure(figsize=(10, 4.5))
gs = GridSpec(1, 2, wspace=0, hspace=0)

#####################
#     Left plot     #
#####################

ax1 = f.add_subplot(gs[0])

ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(axis='x', which='both', direction='in', labelsize=13, length=3)
ax1.tick_params(axis='y', which='both', direction='in', labelsize=13, length=3)
ax1.set_ylabel('Y', size=16)
ax1.set_xlabel('X', size=16)
ax1.set_title('Initial weight vectors', size=16)

# We recompute the initial weight values (this is possible because we set the seed of the random generator)
rng     = np.random.default_rng(0)
weights = rng.normal(size=(2*4, nf))

colors  = ['yellow', 'r', 'b', 'orange', 'magenta', 'brown', 'pink', 'g']

# Plot neurons
for xx, yy, c in zip(weights[:, 0], weights[:, 1], colors):
   ax1.plot(xx, yy, marker='*', color=c, linestyle='none', markersize=10)

# Plot train data
ax1.plot(data_norm[:, 0], data_norm[:, 1], 'k.', linestyle='none')

######################
#     Right plot     #
######################

ax2 = f.add_subplot(gs[1])

ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax2.tick_params(axis='x', which='both', direction='in', labelsize=13, length=3)
ax2.tick_params(axis='y', which='both', direction='in', labelsize=13, length=3)
ax2.set_xlabel('X', size=16)
ax2.set_title('Weight vectors after training', size=16)
ax2.set_yticklabels([])

for idx, xx, yy, c in zip(range(som.weights.shape[0]), som.weights[:, 0], som.weights[:, 1], colors):

   # Find data which have this neuron as BMU
   data_idx = data_norm[pred == idx]

   ax2.plot(data_idx[:, 0], data_idx[:, 1], color=c, marker='.', linestyle='none')
   ax2.plot(xx, yy, marker='*', color=c, linestyle='none', markersize=10, mec='k')

plt.show()