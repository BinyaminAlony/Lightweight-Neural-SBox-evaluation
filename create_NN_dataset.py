from get_NN_dataset_LP_AV_ADEG  import generate_dataset
import math
millnames = ['', '_Thou', '_Mil']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

n_samples = 100_000_000
filename = f"{millify(n_samples)}_samples_LP_DEG_SAC.pt"
dataset = generate_dataset(n_samples, 5, filename)
print(dataset.__len__())