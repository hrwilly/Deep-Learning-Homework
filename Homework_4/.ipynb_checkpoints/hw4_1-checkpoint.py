import sympy as sp
import numpy as np

# 4.1 
def compute_last_state(yN, s):
    """
    Compute the last state (j = N), where all remaining probability budget should be assigned to q_N.

    Parameters:
    - yN: Observed count for the last category.
    - s: Remaining probability budget (typically 1).
    
    Returns:
    - Dictionary with V_j (log-likelihood at this step), q_j (allocated probability), remaining_budget.
    """
    qN = s  # Assign all remaining budget to q_N
    VN = yN * np.log(qN) if qN > 0 else float('-inf')  # Calculate log-likelihood for last step; handle log(0) case
    return {"V_j": float(VN), "q_j": float(qN), "remaining_budget": s}

def compute_optimal_q_and_V(y, V_N, s_initial):
    """
    Dynamic programming to maximize log-likelihood V_j from V_{N} to V_1.
    At each j, initialize q_j and s from the starting state, then find critical points 
    of the objective function within remaining budget s_j. Choose the optimal q_j as 
    the critical point maximizing V_j; fallback to initial allocation if no valid point.

    Parameters:
    - y: Array of observed counts y_i for each category.
    - V_N: Initial value for dynamic programming (starting from step N).
    - s_initial: Initial probability budget, set to 1.
    
    Output:
    - List of optimal V_j, q_j, and remaining s_j values for each step.
    """
    # initialize parameters
    K = np.sum(y)  
    results = []
    V_next = V_N  
    s_j = s_initial  # Start with  s = 1

    # Iterate backward from N to 1
    for j in range(len(y), 0, -1):
        y_j = y[j - 1]  # Get the observed count for step j
        initial_q_j = y_j / K * s_j  # Initial allocation based on the proportion of y_j
        q = sp.Symbol('q', positive=True)  
        
        # Define objective function 
        f = y_j * sp.log(q) + V_next  # Objective function 
        df_dq = sp.diff(f, q)  # Differentiate wrt q
        critical_points = sp.solve(df_dq, q)  # Solve for critical points
        
        # Filter critical points within the range [0, s_j] 
        q_j_candidates = [cp.evalf() for cp in critical_points if cp.is_real and cp > 0 and cp <= s_j]

        # Choose the best candidate that maximizes V_j; fallback to initial allocation otherwise
        if q_j_candidates:
            q_j = max(q_j_candidates, key=lambda q_val: y_j * np.log(q_val) if q_val > 0 else float('-inf'))
        else:
            q_j = initial_q_j  # Use initial allocation if no valid critical point

        # Calculate V_j based on the chosen q_j
        V_j = y_j * np.log(q_j) if q_j > 0 else float('-inf')
        s_j -= q_j  # Update remaining budget

        # Store results
        results.append({
            "step": f"V_{j}",
            "V_j": float(V_j),
            "q_j": float(q_j),
            "remaining_budget": s_j,
            "index": j
        })
        V_next = V_j  # update V_j 

    return results

# Sample inputs for N=6
y = np.array([5, 10, 15, 20, 25, 30])  
s_initial = 1.0

# (a): Compute last step V_N and q_N
last_step_result = compute_last_state(y[-1], s_initial)
print("\n(a) Result for last state V_N and q_N:")
print(f"V_N = {last_step_result['V_j']}, q_N = {last_step_result['q_j']}")

# (b) and (c):
optimal_results = compute_optimal_q_and_V(y, last_step_result["V_j"], s_initial)

# Part (b): V_{N-1} 
V_N_minus_1 = optimal_results[-2]["V_j"]  
print("\n(b) Result for V_{N-1}:")
print(f"V_{len(y) - 1} = {V_N_minus_1}")

# Part (c): q_i results
print("\n(c) Results for q_i values:")
for result in optimal_results:
    step_label = result["step"]
    q_j = result["q_j"]
    j = result["index"]
    print(f"q_{j} = {q_j}")
