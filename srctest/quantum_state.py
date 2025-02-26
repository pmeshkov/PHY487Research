from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class QuantumState:
    """
        A base class for quantum states. Represents an eigenvector 
        of the hamiltonian, in some basis, to be determined in an
        inheriting class.
    """
    quantum_numbers: List[int] = field(default_factory=list, init=False)

    def __str__(self) -> str:
        return f"QuantumState(quantum_numbers={self.quantum_numbers})"
    
@dataclass(frozen=True)
class IntegerAngularState(QuantumState):
    """
    Represents an angular momentum eigenvector in quantum mechanics,
    with integer values.

    Attributes:
        l (int): Orbital angular momentum quantum number (l >= 0).
        m (int): Projection of l onto the z-axis (magnetic number),
            m is in {-l, -l+1, ... , l-1, l}.
    """
    l: int  
    m: int

    def __post_init__(self):
        if self.l < 0:
            raise ValueError("l must be non-negative")
        if not (-self.l <= self.m <= self.l):
            raise ValueError(f"invalid m={self.m} for l={self.l}")
        
        object.__setattr__(self, "quantum_numbers", [self.l, self.m])

    def __str__(self) -> str:
        return f"IntegerAngularState(l={self.l}, m={self.m})"
    