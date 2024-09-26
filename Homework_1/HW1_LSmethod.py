#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:33:26 2024

@author: agnesliang
"""

# loading libraries
import numpy as np

# setting initial parameters
r = 0.01
sigma = 0.1
T = 1
K = 95
N = 100000
n = 50
S0 = 100



############ a. one-step version
# Simulate stock price at t=T 
dt = T
p = 5 # NOT SURE ABUT THIS

np.random.seed(42)
Z = np.random.normal(0, 1, N)  
S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

# Calculate immediate payoff P(XT)
Payoff_XT = np.maximum(K - S_T, 0)

# Construct polynomials of asset prices: x_d = S_t_deltat^d
X_polynomials = np.vstack([S_T**d for d in range(p + 1)]).T

# Run LS regression to estimate beta coefficients
beta_hat, _, _, _ = np.linalg.lstsq(X_polynomials, Payoff_XT, rcond=None)

# Estimate expected value of holding the option
V_hat = X_polynomials @ beta_hat

# Decision-making
V0_a = np.maximum(Payoff_XT, np.exp(-r * dt) * V_hat)

# Calculate V0
V0_a = np.mean(V0_a)
print(f"For one-step version using LS method, the option value is: {V0_a:.4f}")



############ b. multi-step version
dt = T/n

# Initialize the stock price path matrix
Z = np.random.normal(0, 1, (N, n))  
S = np.zeros((N, n + 1))
S[:, 0] = S0

# Simulate paths 
for t in range(1, n + 1):
    S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Calculate immediate payoff P(XT) and VT
Payoff_T = np.maximum(K - S[:, -1], 0)
V = Payoff_T.copy()

# LS from T going backwards
for t in range(n - 1, -1, -1):
    X_polynomials = np.vstack([S[:, t]**d for d in range(p + 1)]).T
    beta_hat, _, _, _ = np.linalg.lstsq(X_polynomials, V * np.exp(-r * dt), rcond=None)
    V_hat = X_polynomials @ beta_hat
    Payoff_Xt = np.maximum(K - S[:, t], 0)
    V = np.maximum(Payoff_Xt, V_hat)
print(Payoff_Xt)
print(V_hat)
V0_b = np.mean(V)
print(f"For multi-step version using LS method, the option value is: {V0_b:.4f}")



############ c. Only including ITM samples 
for t in range(n - 1, -1, -1):
    Payoff_Xt = np.maximum(K - S[:, t], 0)
    in_the_money = Payoff_Xt > 0
    
    if np.sum(in_the_money) > 0:
        X_polynomials = np.vstack([S[in_the_money, t]**d for d in range(p + 1)]).T
        beta_hat, _, _, _ = np.linalg.lstsq(X_polynomials, V[in_the_money] * np.exp(-r * dt), rcond=None)
        V_hat = X_polynomials @ beta_hat
        V[in_the_money] = np.maximum(Payoff_Xt[in_the_money], V_hat)
V0_c = np.mean(V)
print(f"For multi-step version using LS method (with ITM samples only), the option value is: {V0_c:.4f}")
print(f"Performing regression only on in-the-money samples improves estimation accuracy and precision because when  P(Xt)=0, there is no likelihood of exercising the option at that moment. As a result, these out-of-the-money samples do not contribute meaningful information to the regression and only add noise.")