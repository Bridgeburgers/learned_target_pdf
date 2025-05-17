import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

import functions.data_simulator as ds

# %% set parameters
# =============================================================================
#         mu1 = self.a1 * np.sin(self.b1 * (x1 + x2) + self.phi1)
#         sigma1 = self.d1 + self.c1 * np.sqrt(x1**2 + x3**2) / (np.abs(x5) + 0.05)
# 
#         mu2 = self.a2 * np.sin(self.b2 * (x3 + x4) + self.phi2)
#         sigma2 = self.d2 + self.c2 * np.sqrt(x2**2 + x4**2) / (np.abs(x5) + 0.05)
# 
#         pi = 1 / (1 + np.exp(-self.a3 * x5 + self.b3))
# =============================================================================
        
params = dict(
    a1 = 50.,
    b1 = 2 * np.pi,
    phi1 = 0, 
    c1 = 8.,
    d1 = 0.,
    a2 = 50.,
    b2 = np.pi,
    phi2 = np.pi/2,
    c2 = 3.5,
    d2 = 0.,
    a3 = 1.,
    b3 = 0.5,
    delta_t = 0.01,
    target_range = [-80., 80.],
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

pprint({
    'x1': xi[0],
    'x2': xi[1],
    'x3': xi[2],
    'x4': xi[3],
    'x5': xi[4]
})

pprint({
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
plt.close()