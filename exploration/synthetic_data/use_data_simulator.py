import numpy as np
import matplotlib.pyplot as plt

import functions.data_simulator as ds

# %% set parameters
# mu1 = a1 sin(b11 x1 + b12 x2 + phi1)
#sigma1 = d1 + c1 sqrt(x1^2 + x2^2) / (abs(x5) + 0.05)
#mu2 = a2 sin(b21 x3 + b22 x4 + phi2)
#sigma2 = d2 + c2 sqrt(x3^2 + x4^2) / (abs(x5) + 0.05)
#pi = 1 / (1 + exp(-a3 x5 + b3))

params = dict(
    a1 = 8.,
    b1 = 10.,
    phi1 = 0,
    c1 = 2.,
    d1 = 1.,
    a2 = 4.,
    b2 = 5.,
    phi2 = np.pi/2,
    c2 = 2.,
    d2 = 1.,
    a3 = -1.,
    b3 = 0.2,
    delta_t = 0.01,
    target_range = [-20., 20.],
)

n_samples = 10000

# %% create data simulator
sim = ds.Bimodal_Gaussian_Target(**params)

# %%
t_range = sim.t_range
# %% create simulated data
X, t = sim.generate_data(n_samples)

# %% generate random index, plot the target distribution, target, and distribution ev
idx = np.random.choice(len(X), 1)[0]

xi = X[idx]
ti = t[idx]

dist = sim.target_dist(*xi)
tev = sim.target_ev(*xi)

mu1, mu2, sigma1, sigma2, pi = sim.target_params(*xi)

fig, ax = plt.subplots()
ax.plot(t_range, dist, label='target_dist')

ax.set(xlabel='t', ylabel='true_distribution')
ax.grid()

plt.axvline(x=ti, color='b', label='target')
plt.axvline(x=tev, color='r', label='distribution_ev')
plt.legend()

plt.show()

print({
    'mu1': float(mu1),
    'sigma1': float(sigma1),
    'mu2': float(mu2),
    'sigma2': float(sigma2),
    'pi': float(pi)
})

# %% plot the overall histogram of t
plt.hist(t, bins=200)
plt.xlim(-100, 100)
plt.show()