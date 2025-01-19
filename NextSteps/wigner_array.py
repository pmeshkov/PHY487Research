import numpy as np
from sympy.physics.wigner import wigner_3j

def T(n3, n1, m3, m1, n2, m2, N=10):
    m1_ = m1 + N
    m2_ = m2 + 2
    m3_ = m3 + N
    
    return m2_ + 5*n2 + 15*m1_ + (15 + 30*N)*m3_ + (15+ 60*N + 60*N*N)*n1 + (15 + 75*N + 120*N*N + 60 *N*N*N)*n3

N = 10
arr = np.zeros(800415, dtype = np.float64)

i=0
for n3 in range(0, N+1):
    for n1 in range(0, N+1):
        for m3 in range(-N, N+1):
            for m1 in range(-N, N+1):
                for n2 in range(0, 3):
                    for m2 in range(-2,3):
                        arr[i] = wigner_3j(n1,n2,n3,m1,m2,m3)
                        i += 1

np.save("jennoid.npy", arr)