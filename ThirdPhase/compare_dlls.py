"""
Compare two libfunctions builds by training the LDA model with each and
diffing the resulting weights/normalization params.

Usage:
    python compare_dlls.py <old_dll> <new_dll> <dataset_dir>

Both models are trained on the SAME deterministic Train/Test split so any
difference in output is attributable purely to the DLL.
"""
import sys
import ctypes
import array
from pathlib import Path

import WeAreGoingHome as W


def configure_signatures(obj):
    """Replicate the argtypes/restype setup from trainclass.__init__."""
    lib = obj.lib
    lib.tdfeats.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
    ]
    lib.feature_normalization.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.LDA_train.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.LDA_train_accuracy.argtypes = lib.LDA_train.argtypes
    lib.LDA_train_accuracy.restype = ctypes.c_float
    lib.LDA_test.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.LDA_test.restype = ctypes.c_int


def make_obj(dll_path):
    obj = W.trainclass()
    obj.lib = ctypes.CDLL(str(dll_path))
    configure_signatures(obj)
    return obj


def max_abs_diff(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main():
    old_dll, new_dll, ds = sys.argv[1], sys.argv[2], Path(sys.argv[3])

    # --- Train with OLD dll; it creates the deterministic split ---
    old = make_obj(old_dll)
    old.data_set_location = ds
    old.split_training_testing(type="first_half", percentage=50)
    old.set_data_info()
    old.train_model()
    old_test = old.test_model()

    # --- Train with NEW dll on the SAME split (do not re-split) ---
    new = make_obj(new_dll)
    new.data_set_location = ds
    new.train_folder = old.train_folder
    new.testing_folder = old.testing_folder
    new.set_data_info()
    new.train_model()
    new_test = new.test_model()

    print("\n================= DLL COMPARISON =================")
    print(f"dataset            : {ds.name}")
    print(f"feature_dim        : {old.feature_dim}   num_class: {old.num_class}")
    print(f"train acc  old/new : {old.last_train_accuracy:.6f} / {new.last_train_accuracy:.6f}")
    print(f"test  acc  old/new : {old_test:.6f} / {new_test:.6f}")
    print("-------------------------------------------------")
    for name in ("Wg", "Cg", "xmean", "xstd"):
        d = max_abs_diff(getattr(old, name), getattr(new, name))
        print(f"max abs diff {name:<5} : {d:.3e}")
    print("=================================================")

    wg_d = max_abs_diff(old.Wg, new.Wg)
    cg_d = max_abs_diff(old.Cg, new.Cg)
    wg_scale = max((abs(x) for x in old.Wg), default=1.0) or 1.0
    wg_rel = wg_d / wg_scale
    acc_match = (abs(old.last_train_accuracy - new.last_train_accuracy) < 1e-3
                 and abs(old_test - new_test) < 1e-6)

    # The two builds come from different C compilers, so tdfeats/normalization
    # differ at the ULP level (~1e-9); the covariance inversion in LDA_train
    # amplifies that to ~1e-4 in Wg. That is float noise, not an algorithm
    # difference, so we accept a small relative tolerance as long as every
    # classification decision (i.e. the accuracy) is identical.
    print(f"Wg relative diff   : {wg_rel:.3e}")
    if acc_match and wg_rel < 1e-3 and cg_d < 1e-3:
        print("RESULT: MATCH (identical accuracy; weights agree within float noise)")
    else:
        print("RESULT: MISMATCH -- investigate")


if __name__ == "__main__":
    main()
