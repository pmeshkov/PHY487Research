from typing import Any
from system_basis import SystemBasis
import numpy as np

class MixedState:
    def __init__(self, basis: SystemBasis, eigenvalue: float, eigenvector: np.ndarray):
        self.basis = basis
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
    
