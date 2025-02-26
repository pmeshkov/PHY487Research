from dataclasses import dataclass, field
from typing import List

from quantum_state import QuantumState, IntegerAngularState

@dataclass(frozen=True)
class SystemBasis:
    """
    A base class representing the basis of eigenstates. 
    Stores a list of QuantumState objects. It is a singleton
    as we only need one single basis when we are examining
    a system.
    """
    
    _instance = None

    _quantum_states: List[QuantumState] = field(default_factory=list, init=False)
    _dimension: int = field(init=False, default=0)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
@dataclass(frozen=True)
class IntegerAngularBasis(SystemBasis):
    """
    Represents a basis for an angular momentum system in quantum mechanics,
    where angular momenta are in integer values. It is a singleton.

    Attributes:
        _l_max (int): Upper bound of the l value.
        _quantum_states (List[IntegerAngularState]): Array of angular momentum quantum states.
        _dimension (int): Dimension of the basis, it equals (l+1)^2, but it is useful to store
            it separately. 
    """

    _l_max: int
    
    def __post_init__(self):
        if self.l_max < 0:
            raise ValueError("l_max must be non-negative")
        
        dimension: int
        dimension = int((self.l_max + 1) ** 2)

        angular_states: list[IntegerAngularState]
        angular_states = [None] * dimension

        index: int = 0
        for l in range(self.l_max + 1):
            mrange: int = [m0 - l for m0 in range(2 * l + 1)]
            for m in mrange:
                angular_states[index] = IntegerAngularState(l, m)
                index += 1

        object.__setattr__(self, "_quantum_states", angular_states)
        object.__setattr__(self, "_dimension", dimension)