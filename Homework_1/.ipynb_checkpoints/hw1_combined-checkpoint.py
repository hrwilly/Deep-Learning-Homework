'''
MF850: Deep Learning, Statistical Learning
Homework 1: Pricing an American Option via LS and TvR

Authors:

Jiali Liang (U46600644)
Hannah Willy (U19019406)
'''

import numpy as np

# setting initial parameters
r = 0.01
sigma = 0.1
T = 1
K = 95
S0 = 100
N = 100000  
p = 5
n = 50  


def simulate_paths(S0, r, sigma, dt, N, n=1):
    '''
        This function simulates price paths using geometric
        brownian motion for a single stock.
    
        Parameters
        ----------
        S0 : Float / Integer
             Initial asset price.
        r : Float
            Risk free interest rate.
        sigma : Float
                Volatility of the asset.
        dt : Float / Integer
             Time step.
        N : Integer
            Number of simulations.
        n : Integer, optional
            Number of steps. The default is 1.
    
        Returns
        -------
        S : numpy.ndarray
            The matrix of simulated price paths.
    '''
    # Generating a normal random variable to be used as the stochastic term (ie random fluctuations)
    Z = np.random.normal(0, 1, (N, n))
    # Creating a matrix to hold the price paths
    # Giving default values of 0 to be updated with simulated values later on
    S = np.zeros((N, n + 1))
    # Starting value for all paths is S0
    S[:, 0] = S0
    # Applying the geometric brownian motion formula for all time steps and storing it in S
    for t in range(1, n + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    return S

# ls regression
def least_squares_regression(X, y):
    '''
        This function provides a prediction for the options
        value by running a least squares regression on the
        payoff for each simulated path

        Parameters
        ----------
        X : numpy.ndarray
            Price path.
        y : numpy.ndarray
            Future time step's options value.
    
        Returns
        -------
        V_hat : numpy.ndarray
                Predicted options value.

    '''
    # The x variables are polynomials that are the powers of the stock price
    # at time t-(delta t)
    X_polynomials = np.vstack([X**d for d in range(p + 1)]).T
    # Beta hat is the estimate of the regression coefficient
    beta_hat, _, _, _ = np.linalg.lstsq(X_polynomials, y, rcond=None)
    # The predicted options value is the matrix multiplication of the polynomial
    # variables and the regression coefficient
    V_hat = X_polynomials @ beta_hat
    return V_hat

# Calculate V0 using LS method
def compute_ls(S, K, r, dt, n, itm_only=False):
    '''
        This function estimates the options value at time t = 0 
        as an average of the simulated path's estimated payoffs.
        The value is estimated via the Least Squares Monte Carlo
        method.
        

        Parameters
        ----------
        S : numpy.ndarray
            Price paths.
        K : Float / Integer
            Strike price.
        r : Float
            Risk free interets rate.
        dt : Float / Integer
            Time step.
        n : Integer
            Number of steps.
        itm_only : Bool, optional
                   A value of True only includes in the money values when
                   determining whether to exercise. The default is False.
    
        Returns
        -------
        Float
        The option value at t = 0, averaged over all simulations.
    '''
    # The payoff at time T is the maximum of strike minus the final stock price and zero
    Payoff_T = np.maximum(K - S[:, -1], 0)
    # At maturity, the options value is the same as the payoff
    V = Payoff_T.copy()
    
    # from T to 0 --takes the time steps backwards
    for t in range(n - 1, -1, -1):
        # S[:, t] does not include the current time so this is the payoff
        # of the price before time t
        Payoff_Xt = np.maximum(K - S[:, t], 0)
        
        if itm_only:
            # Filter ITM samples
            in_the_money = Payoff_Xt > 0
            # If there are no in the money prices, don't run the regression
            # and instead go to the next time step
            if np.sum(in_the_money) == 0:
                continue
            V_hat = least_squares_regression(S[in_the_money, t], V[in_the_money])
            # Any values that are not in the money will stay zero, and only positions
            # that are in the money will update to a non-zero value
            # Payoff formula: if the payoff (P(Xt)) is greater than the discounted
            # estimated value, then the option value is the payoff. Otherwise, 
            # the option value is the discounted true value
            V[in_the_money] = np.where(Payoff_Xt[in_the_money] - V_hat * np.exp(-r * dt) > 0, Payoff_Xt[in_the_money], V[in_the_money] * np.exp(-r * dt))  
        # If not evaluating for just in the money prices, run the regression
        # as normal and unfiltered
        else:
            V_hat = least_squares_regression(S[:, t], V)  
            # Option value formula: if the payoff (P(Xt)) is greater than the discounted
            # estimated value, then the option value is the payoff. Otherwise, 
            # the option value is the discounted true value
            V = np.where(Payoff_Xt - V_hat * np.exp(-r * dt) > 0, Payoff_Xt, V * np.exp(-r * dt))          

    # Averaging the payoff over all paths
    return np.mean(np.maximum(V, Payoff_Xt))

# Calculate V0 using TvR method
def compute_tvr(S, K, r, dt, n):
    '''
        This function estimates the options value at time t = 0 
        as an average of the simulated path's estimated payoffs.
        The value is estimated via the True Value Region method.
        

        Parameters
        ----------
        S : numpy.ndarray
            Price paths.
        K : Float / Integer
            Strike price.
        r : Float
            Risk free interets rate.
        dt : Float / Integer
            Time step.
        n : Integer
            Number of steps.
    
        Returns
        -------
        Float
        The option value at t = 0, averaged over all simulations.
    '''
    # The payoff at time T is the maximum of strike minus the final stock price and zero
    Payoff_T = np.maximum(K - S[:, -1], 0)
    # At maturity, the options value is the same as the payoff
    V = Payoff_T.copy()

    # from T to 0 --takes the time steps backwards
    for t in range(n - 1, -1, -1):
        # S[:, t] does not include the current time so this is the payoff
        # of the price before time t
        Payoff_Xt = np.maximum(K - S[:, t], 0)
        V_hat = least_squares_regression(S[:, t], V)
        # Option value formula: the maximum of the payoff (P(Xt)) and the discounted
        # estimated option value
        V = np.maximum(Payoff_Xt, V_hat * np.exp(-r * dt))

    # Averaging the payoff over all paths
    return np.mean(np.maximum(V, Payoff_Xt))

def main():
    # a. one-step 
    dt_one_step = T
    S_one_step = simulate_paths(S0, r, sigma, dt_one_step, N, 1)

    # LS 
    V0_ls_one_step = compute_ls(S_one_step, K, r, dt_one_step, 1)
    print(f"(a).For one-step version using LS method, the option value is: {V0_ls_one_step:.4f}")

    # TvR 
    V0_tvr_one_step = compute_tvr(S_one_step, K, r, dt_one_step, 1)
    print(f"(a).For one-step version using TvR method, the option value is: {V0_tvr_one_step:.4f}")

    # b. multi-steps 
    dt_multi_step = T / n
    S_multi_step = simulate_paths(S0, r, sigma, dt_multi_step, N, n)

    # LS for multi-step version
    V0_ls_multi_step = compute_ls(S_multi_step, K, r, dt_multi_step, n)
    print(f"(b).For multi-step version using LS method, the option value is: {V0_ls_multi_step:.4f}")
    
    # TvR for multi-step version
    V0_tvr_multi_step = compute_tvr(S_multi_step, K, r, dt_multi_step, n)
    print(f"(b).For multi-step version using TvR method, the option value is: {V0_tvr_multi_step:.4f}")
   
    # c. only including ITM sample under ls method 
    V0_ls_multi_step_itm = compute_ls(S_multi_step, K, r, dt_multi_step, n, itm_only=True)
    print(f"(c).For multi-step version using LS method (with ITM samples only), the option value is: {V0_ls_multi_step_itm:.4f}")
    print(f"Performing regression only on in-the-money samples improves estimation accuracy and precision because when  P(Xt)=0, there is no likelihood of exercising the option at that moment. As a result, these out-of-the-money samples do not contribute meaningful information to the regression, add noise, and skew the results.")

if __name__ == "__main__":
    main()

