from functools import reduce

from tqdm import tqdm
# import SBox_maps

def xor_my_bits(x):
    """xor all bits of x"""
    vec = [int(i) for i in bin(x)[2:]]
    return reduce(lambda x, y: x ^ y, vec)

def linear_probability(F, in_bits, out_bits = -1):
    """
    Calculate the linear probability of a given substitution map (S-box).
    -----------
    Parameters:
        F : int[]
            Substitution map (S-box) represented as an array of integers.
        in_bits : int
            Number of input bits.
        out_bits : int, optional
            Number of output bits (default is =in_bits).
    Returns: (p, [x_mask, y_mask], Linarity), where:
        p : float
            Linear Probability.
        [x_mask, y_mask] : list of int
            The masks which create this probability.
        Linarity : int
            The maximum count value in the linear distribution table.
    """

    if out_bits == -1:
        out_bits = in_bits

    n_elem = 0x1<<in_bits
    n_outmasks = 0x1<<out_bits
    temp_max = 0
    best_index = -1
    x_masks = range(n_elem)
    y_masks = range(1, n_outmasks)
    for x_mask in (x_masks):
        for y_mask in (y_masks):
            masked_x = [x&x_mask for x in range(n_elem)]
            masked_y = [F[x]&y_mask for x in range(n_elem)]
            count = sum([xor_my_bits(masked_y[i]) == xor_my_bits(masked_x[i]) for i in range(n_elem)])
            lin_prob = (1-2*count/n_elem)**2 # 
            if lin_prob > temp_max:
                temp_max = lin_prob
                maxcount = abs(n_elem - 2 * count) # scriptL in LOW_AND_DEPTH_MASKED (itamar's paper that summerises all good SBoxes)
                best_index = [x_mask,y_mask]
    return temp_max, best_index, maxcount

def partial_linear_probability(F, in_bits, out_bits):
    '''partial_linear_probability
    this tests only the masks of size out_bits with MSB=1
    Parameters:
        F (int[]): substitution map, as an array. 
        in_bits (int): number of input bits.
        out_bits (int, optional): number of output bits (default: = n_bits).
    
    Returns:
    p, [x_mask, y_mask] Where:
        - p - Linear Probability.
        - [x_mask, y_mask] - the masks which create this probability.
        - maxcount - scriptL = linearity
    '''

    if out_bits == -1:
        out_bits = in_bits

    n_elem = 0x1<<in_bits
    partial_n_masks = 0x1<<(out_bits-1) # all numbers with n-1 bits (we will append a '1' MSB to them later)
    temp_max = 0
    best_index = -1
    x_masks = range(n_elem)
    # y_masks = list of numbers of n-1 bits + 2^(n-1) = numbers of n-1 bits, concatted by a '1' from the left.
    y_masks = [partial_n_masks + partial_mask for partial_mask in range(partial_n_masks-1)]
    for x_mask in (x_masks):
        for y_mask in (y_masks):
            masked_x = [x&x_mask for x in range(n_elem)]
            masked_y = [F[x]&y_mask for x in range(n_elem)]
            count = sum([xor_my_bits(masked_y[i]) == xor_my_bits(masked_x[i]) for i in range(n_elem)])
            lin_prob = (1-2*count/n_elem)**2 # 
            if lin_prob > temp_max:
                temp_max = lin_prob
                maxcount = abs(n_elem - 2 * count) # linearity, scriptL in LOW_AND_DEPTH_MASKED (itamar's paper that summerises all good SBoxes)
                best_index = [x_mask,y_mask]
    return temp_max, best_index, maxcount

def compare_linear_probability(F, in_bits, out_bits, best_found = 1, optimal = 0):
    """
    Calculate the linear probability of a given substitution map (S-box).
    -----------
    Parameters:
        F : int[]
            Substitution map (S-box) represented as an array of integers.
        in_bits : int
            Number of input bits.
        out_bits : int, optional
            Number of output bits (default is =in_bits).
        threshold : float
            Linear probability threshold for comparison.
    Returns:
        bool
            True if the LP <= threshold_eq or LP < threshold_neq, false otherwise.
    """

    if out_bits == -1:
        out_bits = in_bits

    n_elem = 0x1<<in_bits
    n_outmasks = 0x1<<out_bits

    x_masks = range(n_elem)
    y_masks = range(1, n_outmasks)
    for x_mask in (x_masks):
        for y_mask in (y_masks):
            masked_x = [x&x_mask for x in range(n_elem)]
            masked_y = [F[x]&y_mask for x in range(n_elem)]
            count = sum([xor_my_bits(masked_y[i]) == xor_my_bits(masked_x[i]) for i in range(n_elem)])
            lin_prob = (1-2*count/n_elem)**2
            if lin_prob >= best_found:
                if lin_prob > optimal: # not (lin_prob < best_found or lin_prob <= optimal)
                    return False
    return True