
from Peptide_sequence_lib.Peptide_Target_Candidate import PeptideCandidate, ProteinTarget
from typing import Optional, Dict, List
from Peptide_sequence_lib.Quantum_Optimizer import QuantumOptimizer
from Peptide_sequence_lib.AminoAcid_Properties import AminoAcid_Properties
import logging
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class PeptideDesigner:
    """Main peptide design framework"""
    
    def __init__(self, use_quantum: bool = True):
        self.use_quantum = use_quantum
        self.quantum_optimizer = QuantumOptimizer(use_simulator=not use_quantum)
        self.aa_props = AminoAcid_Properties()
    
    def design_peptides(self, target: ProteinTarget, peptide_length: int = 10,
                       num_candidates: int = 10, constraints: Optional[Dict] = None,
                       use_classical: bool = False, maxiter_cobyla: int = 100, optimizer: str = 'COBYLA', algorithm: str = 'vqe') -> List[PeptideCandidate]:
        """Design peptides for a given target"""
        logger.info(f"Designing peptides for target: {target.name}")
        
        if constraints is None:
            constraints = self._default_constraints(target)
        
        # Quantum optimization
        solutions = self.quantum_optimizer.optimize_peptide(
            peptide_length, constraints, num_reads=num_candidates * 100,
            use_classical=use_classical, maxiter_cobyla=maxiter_cobyla, optimizer=optimizer, algorithm=algorithm
        )
        
        if solutions is None:
            print("[ADVERTENCIA] No se generaron candidatos (solutions=None).")
            solutions = []
        if len(solutions) == 0:
            print("[ADVERTENCIA] No se generaron candidatos (solutions vacío).")

        # Evaluate candidates
        candidates = []
        seen_sequences = set()
        
        for solution in solutions:
            sequence = solution['sequence']
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            
            # Calculate properties
            binding_affinity = self._calculate_binding_affinity(sequence, target)
            stability = self._calculate_stability(sequence)
            druggability = self._calculate_druggability(sequence)
            diversity_score = self._calculate_diversity_score(sequence)
            
            total_score = (binding_affinity * 0.4 + stability * 0.3 + 
                          druggability * 0.2 + diversity_score * 0.1)
            
            candidate = PeptideCandidate(
                sequence=sequence,
                binding_affinity=binding_affinity,
                stability=stability,
                druggability=druggability,
                diversity_score=diversity_score,
                total_score=total_score
            )
            candidates.append(candidate)
            
            if len(candidates) >= num_candidates:
                break
        
        # Sort by total score
        candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"Generated {len(candidates)} peptide candidates")
        return candidates
    
    def _default_constraints(self, target: ProteinTarget) -> Dict:
        """Generate default constraints based on target properties"""
        constraints = {
            'hydrophobic_reward': 0.5 if target.properties['hydrophobicity'] > 0 else 0.2,
            'position_penalty': 10.0,
            'diversity_reward': 0.1,
            'stability_weight': 0.3,
            'charge_complement': -target.properties['charge'] * 0.1
        }
        return constraints
    
    def _calculate_binding_affinity(self, sequence: str, target: ProteinTarget) -> float:
        """Calculate predicted binding affinity"""
        # Simplified binding affinity calculation
        score = 0.0
        
        # Hydrophobic complementarity
        peptide_hydro = np.mean([self.aa_props.hydrophobicity.get(aa, 0) for aa in sequence])
        target_hydro = target.properties['hydrophobicity']
        hydro_complement = 1.0 - abs(peptide_hydro - target_hydro) / 10.0
        score += hydro_complement * 0.4
        
        # Charge complementarity
        peptide_charge = sum(self.aa_props.charge.get(aa, 0) for aa in sequence)
        target_charge = target.properties['charge']
        charge_complement = 1.0 - abs(peptide_charge + target_charge) / 10.0
        score += charge_complement * 0.3
        
        # Size complementarity
        peptide_volume = np.mean([self.aa_props.volume.get(aa, 0) for aa in sequence])
        size_score = min(peptide_volume / 150.0, 1.0)  # Normalize
        score += size_score * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_stability(self, sequence: str) -> float:
        """Calculate peptide stability"""
        score = 0.0
        
        # Secondary structure propensity
        helix_score = np.mean([self.aa_props.helix_propensity.get(aa, 1.0) for aa in sequence])
        score += (helix_score - 0.5) * 0.5
        
        # Avoid problematic sequences
        if 'PP' in sequence:  # Proline-proline
            score -= 0.3
        if sequence.count('C') > 2:  # Too many cysteines
            score -= 0.2
        
        # Charge distribution
        charges = [self.aa_props.charge.get(aa, 0) for aa in sequence]
        charge_variance = np.var(charges)
        score += min(charge_variance / 2.0, 0.3)
        
        return max(0.0, min(1.0, score + 0.5))
    
    def _calculate_druggability(self, sequence: str) -> float:
        """Calculate druggability score"""
        score = 0.0
        
        # Lipinski-like rules for peptides
        mw = sum(110.0 for _ in sequence)  # Approximate MW
        if mw < 1000:  # Good for drug-like peptides
            score += 0.3
        
        # Hydrophobicity balance
        hydro_score = np.mean([self.aa_props.hydrophobicity.get(aa, 0) for aa in sequence])
        if -2 < hydro_score < 2:  # Balanced hydrophobicity
            score += 0.3
        
        # Avoid problematic amino acids
        problematic = ['C', 'M', 'W']  # Cysteine, Methionine, Tryptophan
        penalty = sum(sequence.count(aa) for aa in problematic) * 0.1
        score -= penalty
        
        # Prefer drug-like amino acids
        drug_like = ['A', 'L', 'I', 'V', 'F', 'Y', 'S', 'T']
        bonus = sum(sequence.count(aa) for aa in drug_like) / len(sequence) * 0.4
        score += bonus
        
        return max(0.0, min(1.0, score))
    
    def _calculate_diversity_score(self, sequence: str) -> float:
        """Calculate chemical diversity score"""
        unique_aa = len(set(sequence))
        diversity = unique_aa / len(sequence)
        
        # Bonus for balanced representation
        aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
        variance = np.var(list(aa_counts.values()))
        balance_score = 1.0 - min(variance / 4.0, 0.5)
        
        return (diversity * 0.7 + balance_score * 0.3)
