'''
This file is to create a class that can simulate data
'''

import numpy as np
import matplotlib.pyplot as plt

def gaussian_dist(mu, sigma, x_range):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x_range - mu)**2 / (2 * sigma**2))

def gaussian_dist_2d(mu, sigma, x_range):
    mu, sigma = np.expand_dims(mu, axis=1), np.expand_dims(sigma, axis=1)
    x_range = np.expand_dims(x_range, axis=0)
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x_range - mu)**2 / (2 * sigma**2))

def numeric_integral(x_range, y_range):
    delta = x_range[1:] - x_range[:-1]
    y_mid = (y_range[1:] + y_range[:-1]) / 2
    return np.sum(y_mid * delta)

def numeric_integral_2d(x_range, y_range):
    delta = x_range[1:] - x_range[:-1]
    delta = np.expand_dims(delta, axis=0)
    y_mid = (y_range[:, 1:] + y_range[:, :-1]) / 2
    return np.sum(y_mid * delta, axis=1)    

class Bimodal_Gaussian_Target:

    def __init__(self, a1 = 8., b1 = 10., phi1=0., c1=2., d1 = 1.,
                      a2=4., b2=5., phi2 = np.pi/2, c2=2., d2=1.,
                      a3 = -1., b3 = 0.2, delta_t = 0.01, target_range=[-10., 10.]):
        self.delta_t = delta_t
        self.target_range = target_range
        self.t_range = np.arange(target_range[0], target_range[1] + delta_t, delta_t)

        self.a1 = a1
        self.b1 = b1
        self.phi1 = phi1
        self.c1 = c1
        self.d1 = d1

        self.a2 = a2
        self.b2 = b2
        self.phi2 = phi2
        self.c2 = c2
        self.d2 = d2

        self.a3 = a3
        self.b3 = b3

    def target_params(self, x1, x2, x3, x4, x5):

        mu1 = self.a1 * np.sin(self.b1 * (x1 + x2) + self.phi1)
        sigma1 = self.d1 + self.c1 * np.sqrt(x1**2 + x3**2) / (np.abs(x5) + 0.05)

        mu2 = self.a2 * np.sin(self.b2 * (x3 + x4) + self.phi2)
        sigma2 = self.d2 + self.c2 * np.sqrt(x2**2 + x4**2) / (np.abs(x5) + 0.05)

        pi = 1 / (1 + np.exp(-self.a3 * x5 + self.b3))

        return mu1, mu2, sigma1, sigma2, pi

    def target_dist(self, x1, x2, x3, x4, x5):
        mu1, mu2, sigma1, sigma2, pi = self.target_params(x1, x2, x3, x4, x5)
        return pi * gaussian_dist(mu1, sigma1, self.t_range) + (1 - pi) * gaussian_dist(mu2, sigma2, self.t_range)

    def target_dist_2d(self, X):
        mu1, mu2, sigma1, sigma2, pi = self.target_params(*X.T)
        pi = np.expand_dims(pi, 1)
        return pi * gaussian_dist_2d(mu1, sigma1, self.t_range) + (1 - pi) * gaussian_dist_2d(mu2, sigma2, self.t_range)

    def target_ev(self, x1, x2, x3, x4, x5):
        return numeric_integral(self.t_range, self.t_range * self.target_dist(x1, x2, x3, x4, x5))
    
    def target_ev_array(self, X, chunks=5000):
        if not chunks:
            return numeric_integral_2d(self.t_range, self.target_dist_2d(X))
        #break apart in chunks to not create too large of a 2D array in memory at once
        n_chunks = X.shape[0] // chunks
        X_list = np.array_split(X, n_chunks, axis=0)
        integral_list = [numeric_integral_2d(self.t_range, self.t_range * self.target_dist_2d(Xi)) for Xi in X_list]
        return np.concatenate(integral_list)

    def plot_target_dist(self, x1, x2, x3, x4, x5):
        plt.plot(self.t_range, self.target_dist(x1, x2, x3, x4, x5))
        plt.show()


    def sample_from_params(self, mu1, mu2, sigma1, sigma2, pi):
        inds = 1 - np.random.binomial(1, pi, len(pi))
        mu_stack = np.column_stack([mu1, mu2])
        sigma_stack = np.column_stack([sigma1, sigma2])
        mu = mu_stack[np.arange(len(mu_stack)), inds]
        sigma = sigma_stack[np.arange(len(sigma_stack)), inds]

        return np.random.normal(mu, sigma)


    def generate_data(self, n_samples):
        X = np.random.normal(0, 1, (n_samples, 5))
        mu1, mu2, sigma1, sigma2, pi = self.target_params(*X.T)
        t = self.sample_from_params(mu1, mu2, sigma1, sigma2, pi)

        return X, t



class Simple_Bimodal_Gaussian_Target:

    def __init__(self, a=1, b=1, c=1, d=1, e=1,
                 delta_t = 0.01, target_range=[-10., 10.]):
        self.delta_t = delta_t
        self.target_range = target_range
        self.t_range = np.arange(target_range[0], target_range[1] + delta_t, delta_t)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def target_params(self, x1, x2, x3, x4, x5):

        mu1 = self.a * x1
        mu2 = self.b * x2
        
        sigma1 = 1. + self.c * np.abs(x3)
        sigma2 = 1. + self.d * np.abs(x4)

        pi = 1 / (1 + np.exp(-self.e * x5))

        return mu1, mu2, sigma1, sigma2, pi

    def target_dist(self, x1, x2, x3, x4, x5):
        mu1, mu2, sigma1, sigma2, pi = self.target_params(x1, x2, x3, x4, x5)
        return pi * gaussian_dist(mu1, sigma1, self.t_range) + (1 - pi) * gaussian_dist(mu2, sigma2, self.t_range)

    def target_dist_2d(self, X):
        mu1, mu2, sigma1, sigma2, pi = self.target_params(*X.T)
        pi = np.expand_dims(pi, 1)
        return pi * gaussian_dist_2d(mu1, sigma1, self.t_range) + (1 - pi) * gaussian_dist_2d(mu2, sigma2, self.t_range)

    def target_ev(self, x1, x2, x3, x4, x5):
        return numeric_integral(self.t_range, self.t_range * self.target_dist(x1, x2, x3, x4, x5))
    
    def target_ev_array(self, X, chunks=5000):
        if not chunks:
            return numeric_integral_2d(self.t_range, self.target_dist_2d(X))
        #break apart in chunks to not create too large of a 2D array in memory at once
        n_chunks = X.shape[0] // chunks
        X_list = np.array_split(X, n_chunks, axis=0)
        integral_list = [numeric_integral_2d(self.t_range, self.t_range * self.target_dist_2d(Xi)) for Xi in X_list]
        return np.concatenate(integral_list)

    def plot_target_dist(self, x1, x2, x3, x4, x5):
        plt.plot(self.t_range, self.target_dist(x1, x2, x3, x4, x5))
        plt.show()


    def sample_from_params(self, mu1, mu2, sigma1, sigma2, pi):
        inds = 1 - np.random.binomial(1, pi, len(pi))
        mu_stack = np.column_stack([mu1, mu2])
        sigma_stack = np.column_stack([sigma1, sigma2])
        mu = mu_stack[np.arange(len(mu_stack)), inds]
        sigma = sigma_stack[np.arange(len(sigma_stack)), inds]

        return np.random.normal(mu, sigma)


    def generate_data(self, n_samples):
        X = np.random.normal(0, 1, (n_samples, 5))
        mu1, mu2, sigma1, sigma2, pi = self.target_params(*X.T)
        t = self.sample_from_params(mu1, mu2, sigma1, sigma2, pi)

        return X, t



