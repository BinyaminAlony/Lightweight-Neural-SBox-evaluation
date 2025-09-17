import numpy as np

def dec_to_bin_vec(n, bits):
    '''dec_to_bin_vec
    Parameters:
        n (int): number to convert
        bits (int): number of bits in output
    
    Returns:
        vec (int[]): binary vector
    '''
    vec = [int(i) for i in bin(n)[2:]]
    vec = [0]*(bits-len(vec)) + vec
    return vec

def bin_vec_to_dec(x):
    '''dec_to_bin_vec
    Parameters:
        x (int[]): binary vector
    
    Returns:
        n (int): decimal int
    '''
    res = 0
    for i in range(len(x)):
        res += (x[i] << i)
    return int(res)

def get_output_bit_functions(F, out_bits):
    '''get_output_bit_functions
    Parameters:
        F (int[]): number to convert
        out_bits (int): number of output bits
    
    Returns:
        out (int[out_bits][]): each subbarray is an output bit map
    '''
    binary_vectors = np.array([dec_to_bin_vec(y, out_bits) for y in F])
    return np.transpose(binary_vectors).tolist()[::-1]

def get_num_bits(n):
    """hamming_weight
    Parameters:
        n (int): integer

    Returns:
        out (int): number of bits required to represent n = ceiling(log2(n))
    """    
    return int(np.ceil(np.log2(n)))

def hamming_weight(n):
    """hamming_weight
    Parameters:
        n (int): integer

    Returns:
        out (int): HW(n)
    """    
    if n == 0:
        return 0
    return sum(dec_to_bin_vec(n, get_num_bits(n)))

def hamming_distance(a,b):
    """hamming_weight
    Parameters:
        a (int): integer
        b (int): integer

    Returns:
        out (int): HD(a,b)
    """
    return hamming_weight(a^b)