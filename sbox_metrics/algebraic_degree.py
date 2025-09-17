import numpy as np
import utils.binary_utils as binary_utils

def multi_bit_ANF(F, in_bits):
    '''multi_bit_ANF
    Parameters:
        F (int[]): substitution map, as an array. 
        in_bits (int): number of input bits.
    
    Returns:
        out (int[][]): ANF (reed-muller spectrum) of each output bit
    '''
    n_elems = 0x1<<in_bits
    output_bit_functions = binary_utils.get_output_bit_functions(F, in_bits)
    masks = binary_utils.get_output_bit_functions(range(n_elems),in_bits)
    spectrums = []
    for f in output_bit_functions:
        spectrum = f
        for output_bit in range(in_bits):
            shift = 0x1<<output_bit
            is_xored = masks[output_bit]
            for i in range(n_elems):
                if is_xored[i]:
                    spectrum[i] ^= spectrum[i-shift]
        spectrums.append(spectrum)
    return spectrums

def multi_bit_algebraic_degree(F, in_bits):
    '''multi_bit_algebraic_degree
    Parameters:
        F (int[]): substitution map, as an array. 
        in_bits (int): number of input bits.
    
    Returns:
    out (int[]): alebraic degree of each output bit
    '''
    spectrums = multi_bit_ANF(F, in_bits)
    return [np.where(spectrum)[0].max() for spectrum in spectrums]


def single_bit_ANF(f, in_bits):
    '''single_bit_ANF
    Parameters:
        f (binary int[]): substitution map, as an array. 
        in_bits (int): number of input bits.
    
    Returns:
        out (int[]): ANF (reed-muller spectrum)
    '''
    n_elems = 0x1<<in_bits
    masks = binary_utils.get_output_bit_functions(range(n_elems),in_bits)
    spectrum = f
    for output_bit in range(in_bits):
        shift = 0x1<<output_bit
        is_xored = masks[output_bit]
        for i in range(n_elems):
            if is_xored[i]:
                spectrum[i] ^= spectrum[i-shift]
    return spectrum

def single_bit_algebraic_degree(f, in_bits):
    '''single_bit_algebraic_degree
    Parameters:
        f (binary int[]): substitution map, as an array. 
        in_bits (int): number of input bits.
    
    Returns:
        out (int): algebraic degree
    '''
    n_elem = 1<<in_bits
    spectrum = single_bit_ANF(f, in_bits)
    weights = []
    for i in range(n_elem):
        if spectrum[i]==1:
            weights.append(binary_utils.hamming_weight(i))
        else:
            weights.append(0)
    # print (weights)
    return max(weights)

# print(single_bit_algebraic_degree([1,1,0,1,0,0,1,0,],3))