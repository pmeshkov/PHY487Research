from sympy.physics.wigner import wigner_3j
from quantum_state import IntegerAngularState
import math

def neg_1_pow(x):
    return 1 - 2 * (x % 2)

def delta(x1, x2):
    if x1 == x2:
        return True
    else:
        return False

def rotational(state1: IntegerAngularState, state2:IntegerAngularState):
    n = state1.l
    n_ = state2.l
    mn = state1.m
    mn_ = state2.m

    if delta(n, n_) and delta(mn, mn_):
        return n*(n+1)
    return 0

def stark(state1: IntegerAngularState, state2:IntegerAngularState):
    
    n = state1.l
    n_ = state2.l
    mn = state1.m
    mn_ = state2.m

    if not (delta(mn, mn_)):
        return 0
    
    wig1 = wigner_3j(n, 1, n_, -mn, 0, mn)
    if wig1 == 0:
        return 0
    
    wig2 = wigner_3j(n, 1, n_, 0, 0, 0)
    if wig2 == 0:
        return 0
    
    other = -neg_1_pow(mn) * math.sqrt((2*n + 1) * (2*n_ + 1))
    return wig1 * wig2 * other