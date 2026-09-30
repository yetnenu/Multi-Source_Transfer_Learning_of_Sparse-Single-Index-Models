import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

def frank_wolfe_spherical_combination(alpha_hats: List[np.ndarray], v_norm: np.ndarray, tolerance: float = 1e-6, use_line_search: bool = True,
    max_iterations: int = 1000, verbose: bool = False) -> np.ndarray:
    """
    Frank-Wolfe Algorithm with Line Search for Spherical Combination
    
    Parameters:
    -----------
    alpha_hats : List[np.ndarray]
        List of K unit vectors in R^d: [α̂₁, α̂₂, ..., α̂_K]
    v_norm : np.ndarray
        Target unit vector v_norm in R^d
    tolerance : float
        Convergence tolerance ε > 0
    use_line_search : bool
        Whether to use exact line search or diminishing step size
    max_iterations : int
        Maximum number of iterations
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    gamma_tilde : np.ndarray
        Optimal weights γ̃ in the (K-1)-dimensional simplex
    """
    
    K = len(alpha_hats)  # Number of unit vectors
    d = len(alpha_hats[0])  # Dimension of vectors
    
    # Step 1: Compute b ∈ R^K where b_i = v_norm^T α̂_i
    b = np.zeros(K)
    for i in range(K):
        b[i] = np.dot(v_norm, alpha_hats[i])
    
    # Step 2: Compute A ∈ R^{K×K} where A_ij = α̂_i^T α̂_j
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = np.dot(alpha_hats[i], alpha_hats[j])
    
    # Step 3: Initialize γ^(0) = (1/K, ..., 1/K)
    gamma_t = np.ones(K) / K
    t = 0
    
    # Store convergence history
    objective_history = []
    gamma_history = [gamma_t.copy()]
    
    def objective_function(gamma: np.ndarray) -> float:
        """Compute objective value F(γ) = (b^T γ) / sqrt(γ^T A γ)"""
        numerator = np.dot(b, gamma)
        denominator = np.sqrt(np.dot(gamma, np.dot(A, gamma)))
        return numerator / denominator if denominator > 1e-12 else 0.0
    
    def line_search_exact(gamma: np.ndarray, d: np.ndarray) -> float:
        """
        Exact line search: maximize w ∈ [0,1] for F(γ + w*d)
        
        Parameters:
        -----------
        gamma : current weight vector
        d : direction vector (s - gamma)
        
        Returns:
        --------
        w_star : optimal step size
        """
        # Define the objective along the line
        def f_line(w):
            gamma_new = gamma + w * d
            # Ensure non-negativity (project to simplex if needed)
            gamma_new = np.maximum(gamma_new, 0)
            gamma_new /= np.sum(gamma_new) if np.sum(gamma_new) > 0 else 1.0
            return objective_function(gamma_new)
        
        # Try multiple starting points for robustness
        w_candidates = np.linspace(0, 1, 100)
        f_values = [f_line(w) for w in w_candidates]
        w_star = w_candidates[np.argmax(f_values)]
        
        # Refine using golden section search around the best candidate
        try:
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(lambda w: -f_line(w), bounds=(0, 1), method='bounded')
            if result.success:
                w_star = result.x
        except ImportError:
            # Fallback if scipy is not available
            pass
            
        return w_star
    
    # Main optimization loop
    for t in range(max_iterations):
        # Step 4: Compute normalization factor h = sqrt((γ^(t))^T A γ^(t))
        h = np.sqrt(np.dot(gamma_t, np.dot(A, gamma_t)))
        
        # Step 5: Compute current objective value F = (b^T γ^(t)) / h
        F = np.dot(b, gamma_t) / h if h > 1e-12 else 0.0
        objective_history.append(F)
        
        # Step 6: Compute gradient ∇F = (1/h) * (b - (F/h) * A γ^(t))
        if h > 1e-12:
            gradient_F = (1/h) * (b - (F/h) * np.dot(A, gamma_t))
        else:
            gradient_F = b.copy()  # Fallback when h is too small
        
        # Step 7: Linear minimization oracle - find i* = argmax [∇F]_i
        i_star = np.argmax(gradient_F)
        
        # Step 8: s = e_i* (standard basis vector)
        s = np.zeros(K)
        s[i_star] = 1.0
        
        # Step 9: Update direction d = s - γ^(t)
        d = s - gamma_t
        
        # Step 10-12: Determine step size w_t
        if use_line_search:
            # Exact line search
            w_t = line_search_exact(gamma_t, d)
        else:
            # Diminishing step size
            w_t = 2.0 / (t + 2)
        
        # Step 13: Update weights γ^(t+1) = γ^(t) + w_t * d
        gamma_t_next = gamma_t + w_t * d
        
        # Project back to simplex to ensure constraints
        gamma_t_next = np.maximum(gamma_t_next, 0)
        gamma_t_next /= np.sum(gamma_t_next)
        
        # Step 14: Check convergence
        gamma_diff = np.linalg.norm(gamma_t_next - gamma_t)
        
        if verbose and t % 100 == 0:
            print(f"Iteration {t}: Objective = {F:.6f}, Gamma_diff = {gamma_diff:.6f}")
        
        gamma_t = gamma_t_next
        gamma_history.append(gamma_t.copy())
        
        # Step 15: Check convergence condition
        if gamma_diff < tolerance:
            if verbose:
                print(f"Converged at iteration {t} with difference {gamma_diff:.6f}")
            break
    
    if verbose and t == max_iterations - 1:
        print(f"Reached maximum iterations {max_iterations}")
    
    return gamma_t, objective_history, gamma_history

# Example usage and demonstration
def demonstrate_algorithm():
    """Demonstrate the Frank-Wolfe algorithm for spherical combination"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Problem parameters
    d = 10  # Dimension of vectors
    K = 5   # Number of unit vectors
    
    print("=== Frank-Wolfe Algorithm for Spherical Combination ===")
    print(f"Dimension: d = {d}, Number of vectors: K = {K}")
    
    # Step 1: Generate random unit vectors α̂₁, ..., α̂_K
    alpha_hats = []
    for i in range(K):
        alpha = np.random.randn(d)
        alpha_hats.append(alpha / np.linalg.norm(alpha))
    
    # Step 2: Generate target unit vector v_norm
    v_norm = np.random.randn(d)
    v_norm = v_norm / np.linalg.norm(v_norm)
    
    print("\nTarget vector correlations with basis vectors:")
    for i, alpha in enumerate(alpha_hats):
        correlation = np.dot(v_norm, alpha)
        print(f"  α̂_{i+1}: {correlation:.4f}")
    
    # Test with line search
    print("\n--- With Line Search ---")
    gamma_opt_ls, obj_history_ls, gamma_history_ls = frank_wolfe_spherical_combination(
        alpha_hats, v_norm, use_line_search=True, verbose=True
    )
    
    print(f"Optimal weights: {gamma_opt_ls}")
    print(f"Final objective value: {obj_history_ls[-1]:.6f}")
    
    # Test without line search (diminishing step size)
    print("\n--- With Diminishing Step Size ---")
    gamma_opt_ds, obj_history_ds, gamma_history_ds = frank_wolfe_spherical_combination(
        alpha_hats, v_norm, use_line_search=False, verbose=True
    )
    
    print(f"Optimal weights: {gamma_opt_ds}")
    print(f"Final objective value: {obj_history_ds[-1]:.6f}")
    
    # Verify the solution
    def compute_combined_vector(alpha_hats, gamma):
        """Compute the combined vector ∑ γ_i α̂_i"""
        combined = np.zeros_like(alpha_hats[0])
        for i, alpha in enumerate(alpha_hats):
            combined += gamma[i] * alpha
        return combined
    
    combined_ls = compute_combined_vector(alpha_hats, gamma_opt_ls)
    combined_ls_norm = combined_ls / np.linalg.norm(combined_ls)
    
    correlation_ls = np.dot(v_norm, combined_ls_norm)
    print(f"\nCorrelation with target (line search): {correlation_ls:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(obj_history_ls, 'b-', label='With Line Search', linewidth=2)
    plt.plot(obj_history_ds, 'r--', label='Diminishing Step Size', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Convergence History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Plot final weights comparison
    x_pos = np.arange(K)
    width = 0.35
    
    plt.bar(x_pos - width/2, gamma_opt_ls, width, label='Line Search', alpha=0.7)
    plt.bar(x_pos + width/2, gamma_opt_ds, width, label='Diminishing Step', alpha=0.7)
    plt.xlabel('Vector Index')
    plt.ylabel('Weight')
    plt.title('Optimal Weights Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return gamma_opt_ls, gamma_opt_ds

# Additional utility functions
def verify_solution(alpha_hats: List[np.ndarray], v_norm: np.ndarray, gamma: np.ndarray) -> dict:
    """
    Verify the quality of the solution
    
    Returns:
    --------
    metrics : dict
        Dictionary containing various quality metrics
    """
    # Compute combined vector
    combined = np.zeros_like(alpha_hats[0])
    for i, alpha in enumerate(alpha_hats):
        combined += gamma[i] * alpha
    
    combined_norm = combined / np.linalg.norm(combined)
    
    # Compute objective value
    A = np.zeros((len(alpha_hats), len(alpha_hats)))
    for i in range(len(alpha_hats)):
        for j in range(len(alpha_hats)):
            A[i, j] = np.dot(alpha_hats[i], alpha_hats[j])
    
    b = np.array([np.dot(v_norm, alpha) for alpha in alpha_hats])
    
    h = np.sqrt(np.dot(gamma, np.dot(A, gamma)))
    objective_value = np.dot(b, gamma) / h if h > 1e-12 else 0.0
    
    metrics = {
        'objective_value': objective_value,
        'correlation_with_target': np.dot(v_norm, combined_norm),
        'weights_sum': np.sum(gamma),
        'weights_non_negative': np.all(gamma >= -1e-10),
        'combined_vector_norm': np.linalg.norm(combined)
    }
    
    return metrics

if __name__ == "__main__":
    # Run demonstration
    gamma_ls, gamma_ds = demonstrate_algorithm()
    
    # Generate test vectors for verification
    np.random.seed(123)
    alpha_hats_test = [np.random.randn(5) for _ in range(3)]
    alpha_hats_test = [alpha/np.linalg.norm(alpha) for alpha in alpha_hats_test]
    v_norm_test = np.random.randn(5)
    v_norm_test = v_norm_test / np.linalg.norm(v_norm_test)
    
    # Test the algorithm
    gamma_opt, _, _ = frank_wolfe_spherical_combination(
        alpha_hats_test, v_norm_test, verbose=False
    )
    
    metrics = verify_solution(alpha_hats_test, v_norm_test, gamma_opt)
    print("\n=== Solution Verification ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")