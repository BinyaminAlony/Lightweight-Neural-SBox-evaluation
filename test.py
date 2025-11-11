import numpy as np
import sympy as sp

# test.pl - prints table of ((-t[:,0] * l[:,0]) * (TN_FN_COEFF - l[:,0])) for t,l in {-1,1}

TN_FN_COEFF = 1.1
t_vals = [1,-1]
l_vals = [1,-1]

for gamma in np.arange(0.1, 2.1, 0.1):
    expr = lambda t, l: sp.simplify(-t * l * abs(gamma - l))
    print(f"g={gamma:.1f}", end="\t")
    for t in t_vals:
        print(f"t={t}", end="\t")
    print()
    for l in l_vals:
        print(f"l={l}", end="\t")
        for t in t_vals:
            print(f"{expr(t, l):.1f}", end="\t")
        print()
    print()