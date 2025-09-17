from operator import add
from collections import Counter

import numpy as np
from utils.binary_utils import bin_vec_to_dec, get_output_bit_functions
from sbox_metrics.linear_probability import compare_linear_probability

def add_ew(a,b):
    """element wise addition of two lists"""
    return list(map(add, a, b))

def add_ew_all(elements):
    """element wise addition of at least 2 lists"""
    res = [0]*len(elements[0])
    for el in elements:
        res = add_ew(res,el)
    return res

def wanted_sums(n_bits):
    """returns the wanted distribution of the HW of any subgroup of any size<=out_bits, of any bijection (specifically [0,1,2,...])"""
    F = get_output_bit_functions(range(1<<n_bits), n_bits)
    res = dict()
    for i in range(2,n_bits+1):
        ops = range(i)
        totest = [F[i] for i in ops]
        res[i]=(Counter(add_ew_all(totest)))
    return res

def check_match(elements, wanted_sum, in_bits, best_LP_found, optimal_LP):
    """check if a group of elements is a valid bijection, and has a partial linear probability below the threshold"""
    sum_counts = Counter(add_ew_all(elements))
    if sum_counts != wanted_sum:
        return False
    temp = np.transpose(elements)
    F = [bin_vec_to_dec(y) for y in temp]
    return compare_linear_probability(F, in_bits, len(elements), best_LP_found, optimal_LP)
