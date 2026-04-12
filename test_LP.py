
from collections import Counter
from tqdm import tqdm
from sbox_metrics.linear_probability import linear_probability
from sbox_metrics.differential_probability import differential_probability
from sbox_metrics.avelanch_criterion import avelanch_criterion
from utils.file_management import load_from_csv

n=5
funcs = load_from_csv('new_nn_dataset_5x5\\good_sboxes_5_7000_0.5_(1).csv')[:100000] #TODO: change file name
# funcs = [
#     [1,16,17,7,22,21,23,25,13,6,28,20,3,14,9,29,11,4,19,15,27,0,24,8,2,5,12,30,31,26,18,10],
#     [14,30,6,12,26,0,29,11,3,15,28,31,22,25,21,2,18,19,10,7,16,23,17,9,4,24,8,13,5,20,1,27],
#     [16,0,6,19,25,30,27,7,28,8,12,2,24,11,29,10,26,23,5,9,20,13,18,17,1,3,14,22,4,15,31,21],
#     [10,13,20,2,4,23,6,22,25,17,31,21,8,18,27,30,9,26,7,11,28,19,12,3,24,0,1,29,14,15,5,16],
#     [28,8,0,25,11,29,30,24,2,15,17,27,5,9,23,3,18,10,21,16,26,19,7,12,14,13,22,1,20,6,31,4],
#     [20,13,0,18,8,7,31,23,17,26,24,15,11,9,19,3,25,4,30,22,16,21,29,1,27,28,5,6,14,10,12,2],
#     [16,28,17,2,22,0,20,30,1,8,19,4,18,31,27,11,5,10,6,26,15,13,21,23,9,24,12,7,25,3,29,14],
#     [12,10,11,7,8,2,28,4,13,29,31,20,0,24,15,18,25,9,14,16,19,5,17,27,23,21,6,1,30,26,22,3],
#     [6,11,28,9,12,10,17,31,23,13,16,25,5,22,14,27,7,20,29,15,4,8,0,24,18,19,26,21,30,2,3,1],
#     [14,24,4,26,8,12,25,11,22,30,2,29,0,1,9,17,10,7,16,31,13,6,3,21,27,15,5,20,18,19,23,28]
# ]
metric_funcs = [
    lambda f: linear_probability(f, n, n)[0],
    lambda f: differential_probability(f, n)[0],
    lambda f: max([abs(0.5-i) for i in avelanch_criterion(f, n)])
]
metric_names = ["LP", "DP", "SAC"]
for metric_func, metric_name in zip(metric_funcs, metric_names):
    counts = dict()
    for f in tqdm(funcs, leave=False):
        metric = metric_func(f)
        if metric in counts:
            counts[metric] += 1
        else:
            counts[metric] = 1
    print(f"{metric_name}: {counts}")
    
    