import numpy as np

# setting initial parameters
r = 0.01
sigma = 0.1
T = 1
K = 95
S0 = 100
N = 100000  
p = 5  # the question specified p>=5 so i just go with 5 
n = 50  

# Functions
def simulate_paths(S0, r, sigma, dt, N, n=1):
    Z = np.random.normal(0, 1, (N, n))
    S = np.zeros((N, n + 1))
    S[:, 0] = S0
    for t in range(1, n + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    return S

# ls regression
def least_squares_regression(X, y):
    X_polynomials = np.vstack([X**d for d in range(p + 1)]).T
    beta_hat, _, _, _ = np.linalg.lstsq(X_polynomials, y, rcond=None)
    V_hat = X_polynomials @ beta_hat
    return V_hat

# Calculate V0 using LS method
def compute_ls(S, K, r, dt, n, itm_only=False):
    Payoff_T = np.maximum(K - S[:, -1], 0)
    V = Payoff_T.copy()
    
    # from T to 0
    for t in range(n - 1, -1, -1):
        Payoff_Xt = np.maximum(K - S[:, t], 0)
        
        if itm_only:
            # Filter ITM samples
            in_the_money = Payoff_Xt > 0
            if np.sum(in_the_money) == 0:
                continue
            V_hat = least_squares_regression(S[in_the_money, t], V[in_the_money] * np.exp(-r * dt))
            V[in_the_money] = np.maximum(Payoff_Xt[in_the_money], V_hat)
        else:
            V_hat = least_squares_regression(S[:, t], V * np.exp(-r * dt))
            V = np.maximum(Payoff_Xt, V_hat)

    return np.mean(V)

# Calculate V0 using TvR method
def compute_tvr(S, K, r, dt, n):
    Payoff_T = np.maximum(K - S[:, -1], 0)
    V = Payoff_T.copy()

    # from T to 0
    for t in range(n - 1, -1, -1):
        Payoff_Xt = np.maximum(K - S[:, t], 0)
        V_hat = np.exp(-r * dt) * np.mean(V)
        V = np.maximum(Payoff_Xt, V_hat)

    return np.mean(V)

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
    print(f"Performing regression only on in-the-money samples improves estimation accuracy and precision because when  P(Xt)=0, there is no likelihood of exercising the option at that moment. As a result, these out-of-the-money samples do not contribute meaningful information to the regression and only add noise.")


if __name__ == "__main__":
    main()

