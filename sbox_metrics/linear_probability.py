# from functools import reduce
# from tqdm import tqdm
# import SBox_maps


x_masks = None
masked_x = None
y_masks = None
masked_y = None

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

    global x_masks
    global masked_x
    global y_masks
    global masked_y

    if x_masks is None:
        x_masks = range(n_elem)
        masked_x = [[x&x_mask for x in range(n_elem)] for x_mask in x_masks]
        y_masks = range(1, n_outmasks)
        masked_y = [[F[x]&y_mask for x in range(n_elem)] for y_mask in y_masks]

    for x_mask in (x_masks):
        for y_mask in (y_masks):
            count = sum([masked_y[y_mask-1][x].bit_count() % 2 == masked_x[x_mask][x].bit_count() % 2 for x in range(n_elem)])
            lin_prob = (1-2*count/n_elem)**2
            if lin_prob > temp_max:
                temp_max = lin_prob
                maxcount = abs(n_elem - 2 * count) # scriptL in LOW_AND_DEPTH_MASKED (itamar's paper that summerises all good SBoxes)
                best_index = [x_mask,y_mask]
    return temp_max, best_index, maxcount

# def partial_linear_probability(F, in_bits, out_bits):
#     '''partial_linear_probability
#     this tests only the masks of size out_bits with MSB=1
#     Parameters:
#         F (int[]): substitution map, as an array. 
#         in_bits (int): number of input bits.
#         out_bits (int, optional): number of output bits (default: = n_bits).
    
#     Returns:
#     p, [x_mask, y_mask] Where:
#         - p - Linear Probability.
#         - [x_mask, y_mask] - the masks which create this probability.
#         - maxcount - scriptL = linearity
#     '''

#     if out_bits == -1:
#         out_bits = in_bits

#     n_elem = 0x1<<in_bits
#     partial_n_masks = 0x1<<(out_bits-1) # all numbers with n-1 bits (we will append a '1' MSB to them later)
#     temp_max = 0
#     best_index = -1
#     x_masks = range(n_elem)
#     # y_masks = list of numbers of n-1 bits + 2^(n-1) = numbers of n-1 bits, concatted by a '1' from the left.
#     y_masks = [partial_n_masks + partial_mask for partial_mask in range(partial_n_masks-1)]
#     for x_mask in (x_masks):
#         for y_mask in (y_masks):
#             masked_x = [x&x_mask for x in range(n_elem)]
#             masked_y = [F[x]&y_mask for x in range(n_elem)]
#             count = sum([xor_my_bits(masked_y[i]) == xor_my_bits(masked_x[i]) for i in range(n_elem)])
#             lin_prob = (1-2*count/n_elem)**2 # 
#             if lin_prob > temp_max:
#                 temp_max = lin_prob
#                 maxcount = abs(n_elem - 2 * count) # linearity, scriptL in LOW_AND_DEPTH_MASKED (itamar's paper that summerises all good SBoxes)
#                 best_index = [x_mask,y_mask]
#     return temp_max, best_index, maxcount

# def compare_linear_probability(F, in_bits, out_bits, best_found = 1, optimal = 0):
#     """
#     Calculate the linear probability of a given substitution map (S-box).
#     -----------
#     Parameters:
#         F : int[]
#             Substitution map (S-box) represented as an array of integers.
#         in_bits : int
#             Number of input bits.
#         out_bits : int, optional
#             Number of output bits (default is =in_bits).
#         threshold : float
#             Linear probability threshold for comparison.
#     Returns:
#         bool
#             True if the LP <= threshold_eq or LP < threshold_neq, false otherwise.
#     """

#     if out_bits == -1:
#         out_bits = in_bits

#     n_elem = 0x1<<in_bits
#     n_outmasks = 0x1<<out_bits

#     x_masks = range(n_elem)
#     y_masks = range(1, n_outmasks)
#     for x_mask in (x_masks):
#         for y_mask in (y_masks):
#             masked_x = [x&x_mask for x in range(n_elem)]
#             masked_y = [F[x]&y_mask for x in range(n_elem)]
#             count = sum([xor_my_bits(masked_y[i]) == xor_my_bits(masked_x[i]) for i in range(n_elem)])
#             lin_prob = (1-2*count/n_elem)**2
#             if lin_prob >= best_found:
#                 if lin_prob > optimal: # not (lin_prob < best_found or lin_prob <= optimal)
#                     return False
#     return True

if __name__ == "__main__":
    # Example usage:

    AES_Sbox = [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
    
    in_bits = 8
    out_bits = 8

    F = [18, 0, 24, 29, 10, 25, 8, 12, 30, 17, 21, 13, 1, 15, 2, 27, 16, 22, 11, 26, 20, 19, 7, 23, 31, 4, 5, 9, 28, 6, 14, 3]
    in_bits = 5
    out_bits = 5
    lp, masks, lin_count = linear_probability(F, in_bits, out_bits)
    print(f"Linear Probability: {lp}, Masks: {masks}, Linearity Count: {lin_count}")