from typing import Tuple, List

# Helper function to generate the Fock basis in descending lexicographic order
# This exactly matches Perceval and Merlin's internal state ordering.
def generate_fock_basis(photons: int, modes: int) -> List[Tuple[int, ...]]:
    states = []
    def _gen(p_left, m_left, current_state):
        if m_left == 1:
            states.append(tuple(current_state + [p_left]))
            return
        for p in range(p_left, -1, -1):
            _gen(p_left - p, m_left - 1, current_state + [p])
    _gen(photons, modes, [])
    return states