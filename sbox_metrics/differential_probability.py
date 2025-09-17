from collections import Counter

# import SBox_maps

def differential_probability(F, in_bits):
    """
    Calculate the differential probability of a given substitution map (S-box).
    -----------
    Parameters:
        F : int[]
            Substitution map (S-box) represented as an array of integers.
        in_bits : int
            Number of input bits.
    -----------
    Returns:
        p: float 
            Differential Probability.
        max_diff: str 
            The most common differential, 'Dx,Dy'.
        max: int
            The maximum value in the Differential Distribution Table (DDT).
    -----------
    Notes:
        - 'Dx' and 'Dy' represent the input and output differences, respectively.
        - The function computes the Differential Distribution Table (DDT) and finds the most common differential.
    """

    n_elem = 0x1<<in_bits
    temp_max = 0
    diffs = []
    # calculating DDT
    for x1 in range(n_elem):
        not_x1 = list(range(n_elem))
        not_x1.remove(x1)
        for x2 in not_x1:
            diffs.append(f'{x1^x2},{F[x1]^F[x2]}')
    
    temp_max = 0
    max_diff = diffs[0]

    # finding DP, delta and max_diff
    occurence_count = Counter(diffs)
    temp_max = occurence_count.most_common(1)[0][1]
    max_diff = occurence_count.most_common(1)[0][0]
    return temp_max/n_elem, max_diff, temp_max # last element is delta in LOW_AND_DEPTH_MASKED (itamar's paper that summerises all good SBoxes)

# print(differential_probability([1,1,0,1,0,1,0,0,],3))