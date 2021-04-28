import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class EM:
    def __init__(self, k, n_iter=10, tol=0.001):
        self.k = k
        self.n_iter = n_iter
        self.tol = tol

        self.weights = np.array([1 / k for _ in range(k)])
        self.mu = None
        self.covars = None


    def fit(self, x):
        k = self.k
        n = x.shape[0]
        d = x.shape[1]
        posterior_log_likelihood = 1e-16
        self.mu = np.random.rand(k, d)

        iter = 0
        while iter <= self.n_iter:
            # Expectation step
            prior_log_likelihood, posterior_log_likelihood, responsibility = self.e_step(x, posterior_log_likelihood)
            if abs(posterior_log_likelihood - prior_log_likelihood) < self.tol:
                break

            # Maximization step
            self.m_step(x, responsibility)
            iter += 1


    def e_step(self, x, posterior_log_likelihood):
        prior_log_likelihood = posterior_log_likelihood
        posterior_log_likelihood, responsibility = self.weighted_likelihood(x)
        return prior_log_likelihood, posterior_log_likelihood, responsibility


    def m_step(self, x, responsibility):
        eps = np.finfo(float).eps
        weights = responsibility.sum(axis=0)
        self.weights = (weights / (weights.sum()))
        self.mu = np.dot(responsibility.T, x) / (weights[:, np.newaxis])


    def weighted_likelihood(self, x, ispredict=False):
        k = self.k
        J = np.ndarray(shape=(x.shape[0], k))

        for i in range(k):
            p = self.mu[i, :].clip(min=1e-16)
            J[:, i] = (np.sum(x * np.log(p), 1) + np.sum((1-x) * np.log((1-p)), 1))

        if ispredict == False:
            J_w = J + np.log(self.weights)
            loglikelihood = np.logaddexp.reduce(J_w, axis=1)
            responsibility = np.exp(J_w - loglikelihood[:, np.newaxis])
            return loglikelihood.mean(), responsibility

        return J


    def predict(self, x):
        exp_weighted_likelihood = np.exp(self.weighted_likelihood(x, ispredict=True))
        return np.sum(exp_weighted_likelihood, 1)
