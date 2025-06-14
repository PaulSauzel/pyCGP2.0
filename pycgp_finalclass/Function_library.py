import numpy as np 
import math

# Secure helpers
def clamp(x, min_val=-1.0, max_val=1.0):
    return max(min(x, max_val), min_val)

def safe_exp(x):
    try:
        return (np.exp(x) - 1.0) / (np.exp(1.0) - 1.0)
    except OverflowError:
        return 1.0  # or a capped value

# List of functions
def f_sum(args, const_params):
    return 0.5 * (args[0] + args[1])

def f_aminus(args, const_params):
    return 0.5 * abs(args[0] - args[1])

def f_mult(args, const_params):
    return args[0] * args[1]

def f_exp(args, const_params):
    return safe_exp(args[0])

def f_abs(args, const_params):
    return abs(args[0])

def f_sqrt(args, const_params):
    return np.sqrt(abs(args[0]))  # always positive input

def f_sqrtxy(args, const_params):
    return np.sqrt(args[0]**2 + args[1]**2) / np.sqrt(2.0)

def f_squared(args, const_params):
    return args[0] ** 2

def f_pow(args, const_params):
    base = abs(args[0])
    exp = abs(args[1])
    try:
        return pow(base, exp)
    except OverflowError:
        return float('inf')

def f_one(args, const_params):
    return 1.0

def f_const(args, const_params):
    return const_params[0]

def f_zero(args, const_params):
    return 0.0

def f_inv(args, const_params):
    return args[0] / abs(args[0]) if args[0] != 0.0 else 0.0

def f_gt(args, const_params):
    return float(args[0] > args[1])

def f_acos(args, const_params):
    return math.acos(clamp(args[0])) / np.pi

def f_asin(args, const_params):
    return 2.0 * math.asin(clamp(args[0])) / np.pi

def f_atan(args, const_params):
    return 4.0 * math.atan(args[0]) / np.pi

def f_min(args, const_params):
    return np.min(args)

def f_max(args, const_params):
    return np.max(args)

def f_round(args, const_params):
    return round(args[0])

def f_floor(args, const_params):
    return math.floor(args[0])

def f_ceil(args, const_params):
    return math.ceil(args[0])

def f_sin(args, const_params):
    return math.sin(args[0])
