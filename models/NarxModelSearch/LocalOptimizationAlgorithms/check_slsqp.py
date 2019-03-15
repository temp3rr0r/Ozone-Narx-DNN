import numpy as np
from LocalOptimizationAlgorithms.slsqp import fmin_slsqp

# from numpy import (zeros, array, linalg, append, asfarray, concatenate, finfo,
#                    sqrt, vstack, exp, inf, isfinite, atleast_1d)
# from .optimize import wrap_function, OptimizeResult, _check_unknown_options
def func1(x):
    return x[0] ** 2


x, f = fmin_slsqp(func1, [4, 2, 5], bounds=[(-50, 11), (0, 1), (0, 2)], disp=1, full_output=True)[:2]