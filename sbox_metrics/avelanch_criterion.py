import sbox_metrics.SBox_maps as SBox_maps
import utils.binary_utils as binary_utils
# import SBox_maps
def avelanch_criterion(F, in_bits, out_bits = -1):
    '''avelanch_criterion
    Parameters:
        F (int[]): substitution map, as an array. 
        in_bits (int): number of input bits.
        in_bits (int): number of input (and output) bits (defaults to =in_bits).
    
    Returns:
        out: average probability of an output bit flipping when an input bit is flipped. (optimally 0.5)
    '''
    if out_bits == -1:
        out_bits = in_bits

    n_elem = 0x1<<in_bits
    outputs_before_flipped_input = binary_utils.get_output_bit_functions(F, in_bits)
    outputs_after_flipped_input = [] # int[][][] - [flipped input bit][output bit][map of output bit]

    # getting the single-bit functions for every bit flipped
    for flipped_bit in range(in_bits):
        mask = 0x1<<flipped_bit
        F_after_flipped_input = [F[x^mask] for x in range(n_elem)]
        outputs_after_flipped_input.append(binary_utils.get_output_bit_functions(F_after_flipped_input, in_bits))
    
    #calculating the probability of flipping an output bit when an input bit is flipped
    probabilities = []
    for output_bit in range(out_bits):
        times_output_flipped = 0
        for flipped_bit in range(in_bits):
            for y in range(n_elem):
                if outputs_after_flipped_input[flipped_bit][output_bit][y] != outputs_before_flipped_input[output_bit][y]:
                    times_output_flipped += 1
        probabilities.append(times_output_flipped/(in_bits*n_elem))
    return probabilities

# print(avelanch_criterion(SBox_maps.AES_Sbox, 8,8))


