from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

from Peptide_sequence_lib.AminoAcid_Properties import AminoAcid_Properties


@dataclass
class PeptideCandidate:
    """Represents a peptide candidate with properties"""
    sequence: str
    binding_affinity: float
    stability: float
    druggability: float
    diversity_score: float
    total_score: float


class ProteinTarget:
    """Represents a protein target for peptide binding"""
    
    def __init__(self, name: str, sequence: str, binding_site: Optional[str] = None):
        self.name = name
        self.sequence = sequence
        self.binding_site = binding_site or ""
        self.properties = self._calculate_properties()
    
    def _calculate_properties(self) -> Dict:
        return {
            'length': len(self.sequence),
            'molecular_weight': self._calculate_mw(),
            'isoelectric_point': self._calculate_pi(),
            'hydrophobicity': self._calculate_hydrophobicity(),
            'charge': self._calculate_charge()
        }

    def _calculate_mw(self) -> float:
        if BIOPYTHON_AVAILABLE:
            return ProteinAnalysis(self.sequence).molecular_weight()
        else:
            aa_weights = {'A': 71.04, 'R': 156.10, 'N': 114.04, 'D': 115.03, 'C': 103.01}
            return sum(aa_weights.get(aa, 110.0) for aa in self.sequence)

    def _calculate_pi(self) -> float:
        if BIOPYTHON_AVAILABLE:
            return ProteinAnalysis(self.sequence).isoelectric_point()
        else:
            return 7.0

    def _calculate_hydrophobicity(self) -> float:
        aa_props = AminoAcid_Properties()
        return np.mean([aa_props.hydrophobicity.get(aa, 0) for aa in self.sequence])

    def _calculate_charge(self) -> float:
        aa_props = AminoAcid_Properties()
        return sum(aa_props.charge.get(aa, 0) for aa in self.sequence)

