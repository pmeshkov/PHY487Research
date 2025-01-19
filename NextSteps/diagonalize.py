import numpy as np
import scipy
import math
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt

def int_or_half_int(x):
    return isinstance(x, int) or abs(round(2*x) - 2*x) == 0

def add_ang_mom(L1, L2):
    #
    # given the values of two different angular momenta, returns an array 
    # of the possible values of their sum L1+L2 = l_tot
    #
    
    if not int_or_half_int(L1) or not int_or_half_int(L2):
        raise Exception("Angular momentum values L1, L2 must be non-negative integers or half integers.", "L1 =", L1, "L2 =", L2)

    l_tot_max = L1 + L2
    if L1 == 0 or L2 == 0:
        return [l_tot_max]

    l_tot_min = abs(L1-L2)
    
    ct = int((l_tot_max - l_tot_min + 1)/ 1)
    l_tot_range = [None] * ct

    l_tot_cur = l_tot_min
    for i in range(ct):
        l_tot_range[i] = l_tot_cur
        l_tot_cur += 1
    
    return l_tot_range

def get_m_range(j):
    #
    # given some angular momentum, returns the m values associated with it
    #

    if not int_or_half_int(j) or j < 0:
        raise Exception("Angular momentum value j must be a non-negative integer or half-integer.")

    if j == 0:
        return [0]

    ct = int(2*j +1)
    m_range = [None] * ct

    m_min = -j
    m_cur = m_min
    for i in range(ct):
        m_range[i] = m_cur
        m_cur += 1
    
    return m_range

def neg_1_pow(x):
    return 1 - 2 * (x % 2)


def delta(x1, x2):
    if x1 == x2:
        return True
    else:
        return False
    
def inverse_cm_to_MHz(E_in_inverse_cm):
    return 29979.2458 * E_in_inverse_cm

class Interaction:
    def __init__(self, const, functn):
        # const is some constant associated with the interaction energy
        # funct is a function which evaluates the matrix element <state1|interaction|state2>
        self.const = const
        self.functn = functn
        
    def eval_interaction(self, state1, state2):
        return self.const * self.functn(state1, state2)
        
class State:
    def __init__(self, n, mn, m1, m2):
        # n is the rotational quantum number N
        # mn is the projection of N onto the z axis
        # m1 is the projection of I1 onto the z axis
        # m2 is the projection of I2 onto the z axis

        self.n = n
        self.mn = mn
        self.m1 = m1
        self.m2 = m2
        
    def __str__(self):
        # returns a string representation of the physical values for each element within the state
        return "n: " + str(self.n) + ", "+ "mn: " + str(self.mn) + ", "+ "m1: " + str(self.m1) + ", "+ "m2: " + str(self.m2)
    
    def get_state_vector(self):
        ''' Returns the state vector [self.n, self.f1, self.f, self.mf]'''
        return [self.n, self.mn, self.m1, self.m2]
    
class Molecule:
    
    def __init__(self, Nrange, I1, I2, n_Ch_Itrcns = 1):
        # Nrange is an array holding the rotational quantum number range to consider in H
        # I1 is the nuclear spin of atom one
        # I2 is the nuclear spin of atom two
        #
        # We double each input so that we can work with integer values
        self.Nrange = Nrange
        self.I1 = I1
        self.I2 = I2
        self.states = []
        self.n_Ch_Itrcns = n_Ch_Itrcns
        
        for n in self.Nrange:
            for mn in get_m_range(n):
                for m1 in get_m_range(I1):
                    for m2 in get_m_range(I2):
                        self.states.append(State(n, mn, m1, m2))
                        #print(n,mn,m1,m2)

        self.dim = len(self.states)
        print("H has dim", self.dim)
        
        # array of interaction functions
        self.interactions = []
        self.changing_interactions = []
        
        #Initialize static Hamiltonian, changing Hamiltonian, and total hamiltonian
        self.H_zero = np.zeros((self.dim,self.dim))
        self.H_primes = np.zeros((n_Ch_Itrcns,self.dim,self.dim))
        
    def add_interaction(self, interaction):
        self.interactions.append(interaction)
        
    def add_changing_interaction(self, interaction):
        # change to add changing interaction
        if(len(self.changing_interactions) > self.n_Ch_Itrcns):
            print("Max changing interactions exceeded for the given molecule object. \n \
                Make a new molecule object with the correct n_Ch_Itrcns parameter.")
        else:
            self.changing_interactions.append(interaction)
    
    def find_H_zero(self):
        if len(self.interactions) == 0:
            print("There are no interactions in the interaction array.")

        # Fill Hamiltonian matrix with term by term 
        for i in range(self.dim): #tqdm(range(self.dim)):
            for j in range(i,self.dim):
                term_zero = 0
                for interaction in self.interactions:
                    term_zero += interaction.eval_interaction(self.states[i], self.states[j])
                self.H_zero[i][j] = term_zero
                self.H_zero[j][i] = np.conjugate(term_zero)

        return self.H_zero
    
    def find_H_prime(self):
        for index, interaction in enumerate(self.changing_interactions):
            for i in range(self.dim): #tqdm(range(self.dim)):
                    for j in range(i,self.dim):
                        term_prime = interaction.eval_interaction(self.states[i], self.states[j])
                        self.H_primes[index][i][j] = term_prime
                        self.H_primes[index][j][i] = np.conjugate(term_prime)
                    
        return self.H_primes

    def compute_eigenval_over_range(self, ChItrcnMagnitudes):
        # ChItrcnMagnitudes is a 2d array with 
        # #rows = len(self.changing_interactions) 
        # #cols = len(interaction range to consider)
        #
        # We invert reshape it (just transpose) before running the code below so that:
        # 
        # Each row represents a "frame" of the changing interactions to consider;
        # a given case where each changing interaction equals something
        #
        # Each column represents one of the changing interactions
        #
        # Here we simply multiply H_prime 
        
        
        ChItrcnMagnitudes = np.transpose(ChItrcnMagnitudes)
        eigen_val_vec_pairs = []
        for frame in ChItrcnMagnitudes:
            #print(frame)
            # each given instance of interaction magnitudes to consider
            H = self.H_zero.copy()
            for interaction_magnitude in frame:
                #print(interaction_magnitude)
                # each interactions respective magnitude, at this given frame
                for H_prime in self.H_primes:
                    H = np.add(H, H_prime*interaction_magnitude)
            eigen_val_vec_pairs.append(np.linalg.eigh(H))
            
        return eigen_val_vec_pairs
    
    def get_H_zero(self):
        return self.H_zero
    
    def get_H_prime(self):
        return self.H_primes
    
    