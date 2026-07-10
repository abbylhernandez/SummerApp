"""
Feed the EXACT SAME feature matrix + labels to LDA_train in both DLLs and
compare the resulting Wg/Cg. This removes tdfeats/feature_normalization
(and any cross-compiler float noise in them) as a variable, isolating whether
the two LDA_train implementations are algorithmically identical.
"""
import sys, ctypes, array, random
from pathlib import Path

old = ctypes.CDLL(sys.argv[1])
new = ctypes.CDLL(sys.argv[2])

FD = 12          # feature_dim
K = 2            # num_class
WPT = 19         # win_per_trial
TPC = 5          # trial_per_class  -> samples_per_class = 95
M = WPT * TPC
N = K * M        # total rows

for lib in (old, new):
    lib.LDA_train.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

# One fixed, shared feature matrix (deterministic pseudo-data with class shift)
random.seed(1234)
base = [random.gauss(0, 1) for _ in range(N * FD)]
feat_template = array.array('f', base)
# add a per-class mean shift so classes are separable
for c in range(K):
    for i in range(M):
        for j in range(FD):
            feat_template[(c * M + i) * FD + j] += 0.5 * (c + 1) * (1 + (j % 3))

labels = array.array('i', [(i // M) + 1 for i in range(N)])


def run(lib):
    feat = array.array('f', feat_template)   # fresh copy (LDA_train doesn't modify, but be safe)
    Wg = array.array('f', [0.0] * (K * FD))
    Cg = array.array('f', [0.0] * K)
    fp = ctypes.cast(feat.buffer_info()[0], ctypes.POINTER(ctypes.c_float))
    lp = ctypes.cast(labels.buffer_info()[0], ctypes.POINTER(ctypes.c_int))
    wp = ctypes.cast(Wg.buffer_info()[0], ctypes.POINTER(ctypes.c_float))
    cp = ctypes.cast(Cg.buffer_info()[0], ctypes.POINTER(ctypes.c_float))
    lib.LDA_train(fp, lp, wp, cp, FD, K, WPT, TPC)
    return Wg, Cg


Wg_o, Cg_o = run(old)
Wg_n, Cg_n = run(new)

dWg = max(abs(a - b) for a, b in zip(Wg_o, Wg_n))
dCg = max(abs(a - b) for a, b in zip(Cg_o, Cg_n))
print(f"ISOLATED LDA_train (identical input feature matrix)")
print(f"max abs diff Wg : {dWg:.3e}")
print(f"max abs diff Cg : {dCg:.3e}")
print("Wg magnitude    :", f"{max(abs(x) for x in Wg_o):.3f}")
print("MATCH" if max(dWg, dCg) < 1e-4 else "MISMATCH")
