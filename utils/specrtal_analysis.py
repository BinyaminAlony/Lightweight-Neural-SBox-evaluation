import numpy as np
from itertools import combinations, product
from functools import reduce
# from differential_probability import differential_probability
# from linear_probability import linear_probability
# import binary_utils

def walsh_transform(f, n):
    """Computes the Walsh transform of a Boolean function f."""
    walsh_values = []
    for a in range(1<<n):
        sum_value = 0
        for x in range(1<<n):
            dot_product = bin(a & x).count('1') % 2  # Compute dot product in GF(2)
            sum_value += (-1) ** ((bin(f[x]).count('1')) % 2 ^ dot_product)
        walsh_values.append(sum_value)
    return walsh_values

def get_output_vector(func, num_vars):
    """
    Generate the output vector of a Boolean function.

    Parameters:
    - func: A function object representing the Boolean function. 
            It should take a list of binary inputs and return a binary output.
    - num_vars: Number of variables in the Boolean function.

    Returns:
    - output_vector: List of outputs of the Boolean function for all input combinations.
    """
    inputs = list(product([0, 1], repeat=num_vars))
    output_vector = [func(inp) for inp in inputs]
    return output_vector

def shannon_spectrum(func, in_bits):
    """
    Calculate the Shannon spectrum of a Boolean function = gettong the binary output vector.

    Parameters:
    - truth_vector: A 1D NumPy array representing the truth vector F of the function.
      Length should be 2^m, where m is the number of input bits.

    Returns:
    - Shannon spectrum vector as a 1D NumPy array.
    """
    # Determine the number of variables (m)
    num_vars = in_bits
    if 2**in_bits != len(func):
        raise ValueError("Length of truth vector must be a power of 2.")

    # Define the I(1) matrix
    I_1 = np.array([[1, 0],
                    [0, 1]])

    # Define the X_i(1) matrix
    X_1 = np.array([[1, 0],
                    [0, 1]])

    # Construct I(m) using Kronecker product iteratively
    I_m = reduce(np.kron, [I_1] * num_vars)

    # Construct X(m) using Kronecker product iteratively
    X_m = reduce(np.kron, [    X_1] * num_vars)

    # Compute the Shannon spectrum
    shannon_spectrum_vector = X_m @ I_m @ func

    return shannon_spectrum_vector

def multibit_shannon(F, in_bits):
    out_bits = binary_utils.get_output_bit_functions(F,in_bits)
    shannon = [shannon_spectrum(f, in_bits) for f in out_bits]
    shannon_t = np.transpose(shannon)
    res = [binary_utils.bin_vec_to_dec(w) for w in shannon_t]
    return res

def RM_spectrum(func, in_bits):
    """
    Calculate the Reed-Muller spectrum (coefficients) of a Boolean function.
    
    Parameters:
    - func: A 1D NumPy array representing the truth table of the Boolean function.
    - in_bits: Number of input bits.
    
    Returns:
    - A list of Reed-Muller coefficients corresponding to each subset of variables.
    """
    # # Determine the number of variables (m)
    # num_vars = in_bits
    # if 2**in_bits != len(func):
    #     raise ValueError("Length of truth vector must be a power of 2.")

    # Define the I(1) matrix
    R_1 = np.array([[1, 0],
                    [1, 1]])

    # Construct I(m) using Kronecker product iteratively
    R_m = reduce(np.kron, [R_1] * in_bits)

    # Compute the Shannon spectrum
    RM_spectrum_vector = R_m @ func

    return RM_spectrum_vector

if __name__ == "__main__":
    print(multibit_shannon([(i//2)%2 for i in range(256)], 8))