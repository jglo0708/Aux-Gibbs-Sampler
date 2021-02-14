# -*- coding: utf-8 -*-
"""
Auxiliary Gibbs Sampler for a probit regression model

@author: LBB
"""
import statsmodels.api as sm
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

np.random.seed(123) #seed
p = 2      # number of independant variables
n = 1000    # number of observations
l = 6000   # Length of the chain
brnin = 3000   # Burn-in: periods to be disregarded for histogram and summary statistic calculation

#Generate covariates
X = np.ones(n, dtype=int) #intercept
X = np.reshape(X, (n, 1))
X = np.hstack((X, np.random.normal(size=(n, p)))) #gaussian covariates
true_beta = np.random.randint(5, size=p+1) #true coefficients
print("True betas:", true_beta)

#Generate binary response data based on probit link
eta = np.column_stack(X.dot(true_beta))
prob = scipy.stats.norm.cdf(eta)
Y = np.random.binomial(p = prob, n = 1, size = n).reshape((n,1))

## Noninformative priors of Beta
# Covariance Matrix
product_X = np.dot(X.T,X)
cov_0 = np.linalg.inv(product_X)

## Informative normal prior of Beta
# Covariance Matrix
v0 = np.eye((p+1))
v0_i = np.linalg.inv(v0)
cov_1 = np.linalg.inv(v0_i + product_X)

# Starting value of Beta (Least Squares estimate)
m_ls = np.dot(cov_0, np.dot(X.T,Y))
m_LS = np.reshape(m_ls, (p+1))
#print(m_LS)

# Starting value of Beta (Maximum Likelihood estimate)
m_ML = sm.Probit(Y, X).fit().params.reshape((1,p+1))
#print(m_ML)

def gibbs_sampler(iterations, burnin, m_pre):
    """
    Gibbs sampler implemented for Beta and latent variables Z. It uses proper conjugate priors.
    The function takes number of iterations and cycles between estimating a vector of latent variables and betas.
    The first cycle of estimates orginates outside of this function.
    It outputs a store of estimates and summary statistics.
    """
    # Storing the initial value of beta
    m_pre = np.reshape(m_pre, (p+1))
    beta_store = np.zeros((p+1,iterations))
    beta_store[:,0] = m_pre

    for its in range(iterations):
        # Sampling of Z based on the value of the previous beta
        exp_val = np.dot(X, beta_store[:,its-1])
        Z = np.zeros(n)
        for i in range(n):
            if Y[i] == 0:
                Z[i] = scipy.stats.truncnorm.rvs(a = -np.inf, b = -exp_val[i], loc = exp_val[i])
            else:
                Z[i] = scipy.stats.truncnorm.rvs(a = -exp_val[i], b = np.inf, loc = exp_val[i])
        
        # Informative Conditional Expectation updated
        m_post = np.dot(cov_1, (np.dot(v0_i, m_pre) + np.dot(X.T,Z))).reshape((p+1))
        beta_star = np.random.multivariate_normal(m_post, cov_1).reshape((p+1))
        beta_store[:,its] = beta_star  

    sample_mean = np.mean(beta_store[:,l-burnin:], axis=1)
    sample_mode = stats.mode(beta_store[:,l-burnin:], axis =1)
    sample_var = np.var(beta_store[:,l-burnin:], axis =1)

    return Z, beta_store, sample_mean, sample_mode, sample_var

# Get function results.  Unhash and input the relevant initial value (m_LS or m_ML)
aux, beta, sample_mean, sample_mode, sample_var = gibbs_sampler(l, brnin, m_LS)

def gibbs_sampler_non(iterations, burnin, m_pre):
    """
    Gibbs sampler implemented for Beta and latent variables Z. It uses non-informative priors.
    The function takes number of iterations and cycles between estimating a vector of latent variables and betas.
    The first cycle of estimates orginates outside of this function.
    It outputs a store of estimates and summary statistics.
    """
    # Storing the initial value of beta
    m_pre = np.reshape(m_pre, (p+1))
    beta_store = np.zeros((p+1,iterations))
    beta_store[:,0] = m_pre

    for its in range(iterations):
        # Sampling of Z based on the value of the previous beta
        exp_val = np.dot(X, beta_store[:,its-1])
        Z = np.zeros(n)
        for i in range(n):
            if Y[i] == 0:
                Z[i] = scipy.stats.truncnorm.rvs(a = -np.inf, b = -exp_val[i], loc = exp_val[i])
            else:
                Z[i] = scipy.stats.truncnorm.rvs(a = -exp_val[i], b = np.inf, loc = exp_val[i])
        
        # Non-Informative Conditional Expectation updated
        beta_z = np.dot(cov_0, np.dot(X.T,Z)).reshape((p+1))
        beta_star = np.random.multivariate_normal(beta_z, cov_0).reshape((p+1))
        beta_store[:,its] = beta_star 

    sample_mean = np.mean(beta_store[:,l-burnin:], axis=1)
    sample_mode = stats.mode(beta_store[:,l-burnin:], axis =1)
    sample_var = np.var(beta_store[:,l-burnin:], axis =1)

    return Z, beta_store, sample_mean, sample_mode, sample_var

# # Get function results. Unhash and input the relevant initial value (m_LS or m_ML)
# aux, beta, sample_mean, sample_mode, sample_var = gibbs_sampler_non(l, brnin, m_LS)


# Summary statistics

print("Number of observations: ",n,"  Length of chain: ",l,"  Burn-In Period: ",brnin, "\n", )

print("BETA 0")
print("True value: ",true_beta[0])
print("Sample results.\n Sample mean: ",sample_mean[0].round(3),"  Sample mode: ",sample_mode[0][0].round(3))
print("Deviation from true values.    Mean: ",abs((sample_mean[0]-true_beta[0]).round(3)),"  Mode: ",abs((sample_mode[0][0]-true_beta[0]).round(3)))
print("Sample Variance ",sample_var[0].round(3))
print("")

print("BETA 1")
print("True value: ",true_beta[1])
print("Sample results. \n Sample mean: ",sample_mean[1].round(3),"  Sample mode: ",sample_mode[0][1].round(3))
print("Deviation from true values.    Mean: ",abs((sample_mean[1]-true_beta[1]).round(3)),"  Mode: ",abs((sample_mode[0][1]-true_beta[1]).round(3)))
print("Sample Variance ",sample_var[1].round(3))

print("")
print("BETA 2")
print("True value: ",true_beta[2])
print("Sample results. \n Sample mean: ",sample_mean[2].round(3),"  Sample mode: ",sample_mode[0][2].round(3))
print("Deviation from true values.    Mean: ",abs((sample_mean[2]-true_beta[2]).round(3)),"  Mode: ",abs((sample_mode[0][2]-true_beta[2]).round(3)))
print("Sample Variance ",sample_var[2].round(3))

## VISUALIZATION
# Trace plots

fig0 = plt.figure()
plt.plot(beta[0,:], label='plot');
plt.axhline(y=true_beta[0], color='r', linestyle='-', label='true beta')
plt.title("Trace plot - Beta 0")
plt.legend()

fig1 = plt.figure()
plt.plot(beta[1,:], label='plot');
plt.axhline(y=true_beta[1], color='r', linestyle='-', label='true beta')
plt.title("Trace plot - Beta 1")
plt.legend()

fig2 = plt.figure()
plt.plot(beta[2,:], label='plot');
plt.axhline(y=true_beta[2], color='r', linestyle='-', label='true beta')
plt.title("Trace plot - Beta 2")
plt.legend()


# Histograms
nbins = 50 

fig3 = plt.figure()
plt.hist(beta[0,l-brnin:],  bins=nbins,label='hist');
plt.axvline(x=true_beta[0], color='r', linestyle='-', label='true beta / mode')
plt.title("Histogram - Beta 0")
plt.axvline(x=sample_mean[0], color='m', linestyle=':',label='sample mean')
plt.axvline(x=sample_mode[0][0], color='g', linestyle=':',label='sample mode')
plt.legend()

fig4 = plt.figure()
plt.hist(beta[1,l-brnin:],  bins=nbins,label='hist');
plt.axvline(x=true_beta[1], color='r', linestyle='-', label='true beta / mode')
plt.title("Histogram - Beta 1")
plt.axvline(x=sample_mean[1], color='m', linestyle=':',label= 'sample mean')
plt.axvline(x=sample_mode[0][1], color='g',linestyle=':', label= 'sample mode')
plt.legend()

fig5 = plt.figure()
plt.hist(beta[2,l-brnin:],  bins=nbins,label='hist');
plt.axvline(x=true_beta[2], color='r', linestyle='-', label='true beta / mode')
plt.title("Histogram - Beta 2")
plt.axvline(x=sample_mean[2], color='m', linestyle=':',label= 'sample mean')
plt.axvline(x=sample_mode[0][2], color='g',linestyle=':', label= 'sample mode')
plt.legend()

plt.show()