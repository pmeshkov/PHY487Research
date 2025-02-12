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

def T_int(n3, n1, m3, m1, n2, m2, N=10):
    m1_ = m1 + N
    m2_ = m2 + 2
    m3_ = m3 + N
    
    return m2_ + 5*n2 + 15*m1_ + (15 + 30*N)*m3_ + (15+ 60*N + 60*N*N)*n1 + (15 + 75*N + 120*N*N + 60 *N*N*N)*n3

def T_half_int(n3, n1, m3, m1, n2, m2, N = 7/2):
    n3 = n3 - 0.5 # 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3 (max is (N - 0.5))
    n1 = n1 - 0.5 # 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3 (max is (N - 0.5))
    
    n2 = n2 # 0, 1, 2 (max is 2)
    m2 = m2 + 2 # -2, -1, 0, 1, 2 -> 0, 1, 2, 3, 4 (max is 4)
    
    m3 = m3 + N # -7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3, 4, 5, 6, 7 (max is N*2)
    m1 = m1 + N # -7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3, 4, 5, 6, 7 (max is N*2)
    
    # max: (4)    (4)+(2*5)    (4+2*5)+(15*N*2)    (4+2*5+15*N*2)+(15+30*N)*N*2   (4+2*5+15*N*2)+(15+30*N)*N*2 + (15+60*N+60*N*N)*(N-0.5)
    return int(m2     + 5 * n2         + 15 * m1               + (15+30*N)*m3  + (15+60*N+60*N*N)*n1     + (60*N*N*N + 90*N*N + 45*N + 7.5)*n3)

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
    
class Hamiltonian:
    
    def __init__(self, Nrange, I1, I2, int_wigner_arr, halfint_wigner_arr, n_Ch_Itrcns = 1):
        # Nrange is an array holding the rotational quantum number range to consider in H
        # I1 is the nuclear spin of atom one
        # I2 is the nuclear spin of atom two
        #
        # We double each input so that we can work with integer values
        self.Nrange = Nrange
        self.I1 = I1
        self.I2 = I2
        self.states = []
        self.int_wigner_arr = int_wigner_arr
        self.halfint_wigner_arr = halfint_wigner_arr
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
        self.H_primes = np.zeros((self.n_Ch_Itrcns,self.dim,self.dim))
        
    def wigner_3j(self, n1, n2, n3, m1, m2, m3):
        if n1 == int(n1) and n2 == int(n2) and n3 == int(n3):
            return self.int_wigner_arr[T_int(n3,n1,m3,m1,n2,m2)]
        return self.halfint_wigner_arr[T_half_int(n3,n1,m3,m1,n2,m2)]
        
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
    
    def reset(self):
        # array of interaction functions
        self.interactions = []
        self.changing_interactions = []
        
        #Initialize static Hamiltonian, changing Hamiltonian, and total hamiltonian
        self.H_zero = np.zeros((self.dim,self.dim))
        self.H_primes = np.zeros((self.n_Ch_Itrcns,self.dim,self.dim))
        

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
    
    def T_int(n3, n1, m3, m1, n2, m2, N=10):
        m1_ = m1 + N
        m2_ = m2 + 2
        m3_ = m3 + N
        
        return m2_ + 5*n2 + 15*m1_ + (15 + 30*N)*m3_ + (15+ 60*N + 60*N*N)*n1 + (15 + 75*N + 120*N*N + 60 *N*N*N)*n3

    def T_half_int(n3, n1, m3, m1, n2, m2, N = 7/2):
        n3 = n3 - 0.5 # 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3 (max is (N - 0.5))
        n1 = n1 - 0.5 # 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3 (max is (N - 0.5))
        
        n2 = n2 # 0, 1, 2 (max is 2)
        m2 = m2 + 2 # -2, -1, 0, 1, 2 -> 0, 1, 2, 3, 4 (max is 4)
        
        m3 = m3 + N # -7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3, 4, 5, 6, 7 (max is N*2)
        m1 = m1 + N # -7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2 -> 0, 1, 2, 3, 4, 5, 6, 7 (max is N*2)
        
        # max: (4)    (4)+(2*5)    (4+2*5)+(15*N*2)    (4+2*5+15*N*2)+(15+30*N)*N*2   (4+2*5+15*N*2)+(15+30*N)*N*2 + (15+60*N+60*N*N)*(N-0.5)
        return int(m2     + 5 * n2         + 15 * m1               + (15+30*N)*m3  + (15+60*N+60*N*N)*n1     + (60*N*N*N + 90*N*N + 45*N + 7.5)*n3)
        
    def solveNaCs(self, E_range):
        from scipy.constants import physical_constants
        from scipy.constants import c
        from scipy.constants import epsilon_0
        
        h = scipy.constants.h
        muN = scipy.constants.physical_constants['nuclear magneton'][0]
        bohr = scipy.constants.physical_constants['Bohr radius'][0]
        eps0 = scipy.constants.epsilon_0
        c = scipy.constants.c
        DebyeSI = 3.33564e-30

        Na23Cs133 = {"I1":1.5,
                    "I2":3.5,
                    "g1":1.478,
                    "g2":0.738,
                    "d0":4.69*DebyeSI,
                    "Brot":0.058*c*100*h,
                    "Drot":0*h,
                    "Q1":-0.097e6*h,
                    "Q2":0.150e6*h,
                    "C1":14.2*h,
                    "C2":854.5*h,
                    "C3":105.6*h,
                    "C4":3941.8*h,
                    "MuN":0*muN,
                    "Mu1":1.478*muN,
                    "Mu2":0.738*muN,
                    "a0":0*h, #Not reported
                    "a2":0*h, #Not reported
                    "Beta":0}

        int_rotational = Interaction(Na23Cs133["Brot"], InteractionTypes.rotational)
        int_centrifugal = Interaction(Na23Cs133["Drot"], InteractionTypes.centrifugal)
        int_quad_na = Interaction(Na23Cs133["Q1"], InteractionTypes.quad_Na)
        int_quad_cs = Interaction(Na23Cs133["Q2"], InteractionTypes.quad_Cs)
        const_nuc_spin_spin_dip = Na23Cs133["g1"] * Na23Cs133["g2"] * physical_constants["nuclear magneton"][0]**2 / (4 * math.pi * epsilon_0 * c * c)
        int_nuc_spin_spin_dip = Interaction(const_nuc_spin_spin_dip, InteractionTypes.nuc_spin_spin_dip)
        int_nuc_spin_spin = Interaction(Na23Cs133["C4"], InteractionTypes.nuc_spin_spin)
        int_spin_rot_Na = Interaction(Na23Cs133["C1"], InteractionTypes.spin_rot_Na)
        int_spin_rot_Cs = Interaction(Na23Cs133["C2"], InteractionTypes.spin_rot_Cs)
        int_stark = Interaction(Na23Cs133["d0"], InteractionTypes.stark)
        
        self.add_interaction(int_rotational)
        #NaCs.add_interaction(int_centrifugal)
        #NaCs.add_interaction(int_quad_na)
        #NaCs.add_interaction(int_quad_cs)
        #NaCs.add_interaction(int_nuc_spin_spin)
        #NaCs.add_interaction(int_nuc_spin_spin_dip)
        #NaCs.add_interaction(int_spin_rot_Na)
        #NaCs.add_interaction(int_spin_rot_Cs)
        self.add_changing_interaction(int_stark)
        
        self.find_H_zero()
        self.find_H_prime()
        
        E_range = E_range
        eigenvalues_and_eigenvectors = self.compute_eigenval_over_range([E_range])
        
        return eigenvalues_and_eigenvectors
    
    def dipole_dipole_matrix_elem(evec1, evec2, basis):
        if (len(evec1) != len(evec2) or len(evec1) != len(basis)):
            raise('ArithmeticError')
        else:
            value = 0
            for i, a in enumerate(evec1):
                for j, b in enumerate(evec2):
                    value += a * b * InteractionTypes.stark(basis[i], basis[j])
            return value
                    
class InteractionTypes:
    def rotational(state1: State, state2:State):
        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()

        if delta(n, n_) and delta(mn, mn_) and delta(m1, m1_) and delta(m2, m2_):
            return n*(n+1)
        return 0

    def centrifugal(state1: State, state2:State):
        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()

        if delta(n, n_) and delta(mn, mn_) and delta(m1, m1_) and delta(m2, m2_):
            return -n*(n+1)**2
        return 0

    def quad_Na(state1: State, state2:State):
        i1 = 3/2
        i2 = 7/2

        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        if not delta(m2, m2_):
            return 0
        
        sum_term = 0
        for p in range(-2,3):
            wigp1 = wigner_3j(n, 2, n_, -mn, p, mn_)
            wigp2 = wigner_3j(i1, 2, i1, -m1, -p, m1_)
            sum_term += neg_1_pow(p) * wigp1 * wigp2
        
        if sum_term == 0:
            return 0 
            
        wig3 = wigner_3j(n, 2, n_, 0, 0, 0)
        if wig3 == 0:
            return 0

        wig4 = wigner_3j(i1, 2, i1, -i1, 0, i1)
        if wig4 == 0:
            raise("ArithmeticError; wigner coefficient is 0 but must be inverted")
        wig4 = 1.0/wig4
        
        other = neg_1_pow(-mn+i1-m1)*math.sqrt((2*n + 1) * (2*n_ + 1))/4

        return sum_term * wig3 * wig4 * other

    def quad_Cs(state1:State, state2:State):
        i1 = 3/2
        i2 = 7/2

        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        if not delta(m1, m1_):
            return 0
        
        sum_term = 0
        for p in range(-2,3):
            wigp1 = wigner_3j(n, 2, n_, -mn, p, mn_)
            wigp2 = wigner_3j(i2, 2, i2, -m2, -p, m2_)
            sum_term += neg_1_pow(p) * wigp1 * wigp2
        
        if sum_term == 0:
            return 0 
            
        wig3 = wigner_3j(n, 2, n_, 0, 0, 0)
        if wig3 == 0:
            return 0

        wig4 = wigner_3j(i2, 2, i2, -i2, 0, i2)
        if wig4 == 0:
            raise("ArithmeticError; wigner coefficient is 0 but must be inverted")
        wig4 = 1.0/wig4
        
        other = neg_1_pow(-mn+i2-m2)*math.sqrt((2*n + 1) * (2*n_ + 1))/4

        return sum_term * wig3 * wig4 * other

    def spin_rot_Na(state1:State, state2:State):
        i1 = 3/2
        i2 = 7/2

        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        if not (delta(n, n_) and delta(m2, m2_)):
            return 0
        
        sum_term = 0
        for p in range(-1,2):
            wigp1 = wigner_3j(n, 1, n, mn, p, -mn_)
            wigp2 = wigner_3j(i1, 1, i1, m1, -p, -m1_)
            sum_term += wigp1 * wigp2 * neg_1_pow(p)
        
        if sum_term == 0:
            return 0
            
        other = neg_1_pow(n+mn_+i1+m1_) * math.sqrt(n * (n + 1) * (2*n + 1) * i1 * (i1 + 1) * (2*i1 + 1))
        if other == 0:
            return 0 

        return sum_term * other

    def spin_rot_Cs(state1:State, state2:State):
        i1 = 3/2
        i2 = 7/2

        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        if not (delta(n, n_) and delta(m1, m1_)):
            return 0
        
        other = neg_1_pow(n+mn_+i2+m2_) * math.sqrt(n * (n + 1) * (2*n + 1) * i2 * (i2 + 1) * (2*i2 + 1))
        if other == 0:
            return 0 
        
        sum_term = 0
        for p in range(-1,2):
            wigp1 = wigner_3j(n, 1, n, mn, p, -mn_)
            wigp2 = wigner_3j(i2, 1, i2, m2, -p, -m2_)
            sum_term += wigp1 * wigp2 * neg_1_pow(p)
        
        if sum_term == 0:
            return 0

        return sum_term * other

    def nuc_spin_spin(state1: State, state2:State):
        i1 = 3/2
        i2 = 7/2

        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        if not (delta(n, n_) and delta(mn, mn_)):
            return 0
        
        other = neg_1_pow(i1+m1_+i2+m2_) * math.sqrt(i1 * (i1 + 1) * (2*i1 + 1) * i2 * (i2 + 1) * (2*i2 + 1))
        
        sum_term = 0
        for p in range(-1,2):
            wigp1 = wigner_3j(i1, 1, i1, m1, p, -m1_)
            wigp2 = wigner_3j(i2, 1, i2, m2, -p, -m2_)
            sum_term += wigp1 * wigp2 * neg_1_pow(p)
        
        if sum_term == 0:
            return 0

        return sum_term * other

    def nuc_spin_spin_dip(state1: State, state2:State):
        i1 = 3/2
        i2 = 7/2

        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        sqrtpart = -math.sqrt(30 * i1 * (i1 + 1) * (2*i1 + 1) * i2 * (i2 + 1) * (2*i2 + 1) * (2*n + 1) * (2*n_ + 1))
        
        negpart = neg_1_pow(i1 + i1 - m1 - m2 - mn)
        
        wig0 = wigner_3j(n, 2, n_, 0, 0, 0)
        
        sum = 0
        
        for p in range(-1,2):
            wig1 = wigner_3j(i2, 1, i2, -m2, -p, m2)
            
            for p1 in range(-1,2):
                for p2 in range(-1,2):
                    
                    wig2 = wigner_3j(n, 1, n_, -mn, p1, mn_)
                    wig3 = wigner_3j(i1, 1, i1, -m1, p2, m1_)
                    wig4 = wigner_3j(1, 2, 1, p1, p2, -p)
            sum += wig1 * wig2 * wig3 * wig4
            
        if sum == 0:
            return 0
        
        return sqrtpart * negpart * wig0 * sum

    def stark(state1: State, state2:State):
        
        n, mn, m1, m2  = state1.get_state_vector()
        n_, mn_, m1_, m2_  = state2.get_state_vector()
        
        if not (delta(mn, mn_) and delta(m1, m1_) and delta(m2, m2_)):
            return 0
        
        wig1 = wigner_3j(n, 1, n_, -mn, 0, mn)
        if wig1 == 0:
            return 0
        
        wig2 = wigner_3j(n, 1, n_, 0, 0, 0)
        if wig2 == 0:
            return 0
        
        other = -neg_1_pow(mn+m1+m2) * math.sqrt((2*n + 1) * (2*n_ + 1))
        #print(state1,state2, other)
        
        return wig1 * wig2 * other