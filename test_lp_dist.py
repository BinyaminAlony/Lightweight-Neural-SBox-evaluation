import math

def E_fixed(n, two_k):
    k = two_k // 2
    
    # Calculate the squared binomial term
    binomial_coef = math.comb(1 << (n - 1), (1 << (n - 2)) + k) ** 2
    
    # Simplify the factorials to a single denominator combination to avoid float overflow
    denominator = math.comb(1 << n, 1 << (n - 1))
    
    # Fix precedence: explicitly wrap (1 << n) before subtracting 1
    numerator = 2 * (((1 << n) - 1) ** 2) * binomial_coef
    
    return numerator / denominator

def prob_max_under_fixed(n, two_t):
    P = 1.0
    t = two_t // 2
    
    # Sum over the half-deviation k, from t up to max possible deviation 2^{n-2}
    for k in range(t, (1 << (n - 2)) + 1):
        P -= E_fixed(n, 2 * k)
        
    return P

# print(prob_max_under_fixed(8, 48))
print(prob_max_under_fixed(5, 10))