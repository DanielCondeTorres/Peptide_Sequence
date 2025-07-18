#!/usr/bin/env python3
"""
De Novo Design of Protein-Binding Peptides by Quantum Computing
Implementation based on arXiv:2503.05458

This framework integrates classical and quantum computing for peptide design
using D-Wave quantum annealer for optimization.
"""

import qiskit
import qiskit_aer
print(qiskit.__version__)
print(qiskit_aer.__version__)

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer.primitives import Estimator
# VQE y optimizadores modernos (Qiskit >=1.0)
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import itertools
from dataclasses import dataclass
import logging
import time
import platform
import psutil
from datetime import datetime
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import json
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

# Quantum Computing Libraries
try:
    import dimod
    DWAVE_AVAILABLE = False  # For compatibility, but we won't use D-Wave
except ImportError:
    print("dimod not available. Please install dimod.")
    raise
from qiskit_aer import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Bioinformatics Libraries
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqUtils import ProtParam
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("BioPython not available. Using basic sequence analysis.")

# Molecular Modeling (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available. Using simplified molecular properties.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)










# Mis  librerias
from Peptide_sequence_lib import *

class VQEResult:
    '''
    Clase para almacenar los resultados del VQE clásico/manual
    '''
    def __init__(self, optimize_result, best_bitstring, selected_qubits, ansatz, initial_point, cobyla_params, circuit_transpiled, start_time):
        self.selected_qubits = selected_qubits
        # Simulación: no hay probabilidades reales, pero puedes poner un dict dummy
        probabilities = {best_bitstring: 1.0}
        self.eigenstate = type('EigenState', (), {
            'binary_probabilities': lambda self: probabilities
        })()
        self.protein_info = {
            "sequence": "manual_sequence",
            "length": len(best_bitstring),
            "interface_params": {
                "interface_axis": None,
                "interface_weight": None,
                "interface_displacement": None
            }
        }
        self.optimal_value = optimize_result.fun
        self.optimal_parameters = optimize_result.x
        self.best_measurement = {
            "state": int(best_bitstring, 2),
            "bitstring": best_bitstring,
            "value": optimize_result.fun,
            "probability": probabilities[best_bitstring]
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_state = (best_bitstring, 1.0)
        self.measurement_results = {
            "best_measurement": self.best_measurement,
            "aux_operators_evaluated": {
                "qubit_mapping": self.selected_qubits
            }
        }
        self.optimal_point = optimize_result.x
        self.optimal_value = optimize_result.fun
        self.optimal_parameters = optimize_result.x
        self.cost_function_evals = optimize_result.nfev
        self.input_parameters = {
            "sequence": "manual_sequence",
            "interface_axis": None,
            "interface_weight": None,
            "interface_displacement": None
        }
        self.execution_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cost_function_evals": optimize_result.nfev,
            "num_qubits": ansatz.num_qubits,
            "execution_time_seconds": time.time() - start_time,
            "hardware_info": {
                "system": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "selected_qubits": self.selected_qubits
            }
        }
        self.vqe_configuration = {
            "ansatz_params": {
                "num_qubits": ansatz.num_qubits,
                "num_parameters": ansatz.num_parameters,
                "depth": circuit_transpiled.depth() if hasattr(circuit_transpiled, 'depth') else None
            },
            "optimizer_params": cobyla_params,
            "initial_point": str(list(initial_point))
        }
        self.optimization_results = {
            "optimal_value": {
                "value": optimize_result.fun,
                "description": "Lowest energy found for the protein conformation",
                "units": "arbitrary energy units"
            },
            "circuit_info": {
                "total_qubits": ansatz.num_qubits,
                "total_parameters": ansatz.num_parameters,
                "depth": circuit_transpiled.depth() if hasattr(circuit_transpiled, 'depth') else None,
                "type": "TwoLocal",
                "description": "TwoLocal ansatz with ry rotations and cz gates",
                "ascii_file": f"circuit_ascii_{timestamp}.txt",
                "image_file": f"circuit_diagram_{timestamp}.png",
                "operations": len(circuit_transpiled.data) if hasattr(circuit_transpiled, 'data') else None,
                "gate_counts": dict(circuit_transpiled.count_ops()) if hasattr(circuit_transpiled, 'count_ops') else {},
                "physical_qubits": self.selected_qubits,
                "transpiled_info": {
                    "depth": circuit_transpiled.depth() if hasattr(circuit_transpiled, 'depth') else None,
                    "size": len(circuit_transpiled.data) if hasattr(circuit_transpiled, 'data') else None,
                    "initial_layout": self.selected_qubits
                }
            },
            "optimal_parameters": self._format_optimal_parameters(optimize_result.x, ansatz, self.selected_qubits)
        }
    def _format_optimal_parameters(self, parameters, ansatz, selected_qubits):
        formatted_params = {}
        for i, param in enumerate(parameters):
            formatted_params[f"angle_{i}"] = {
                "value": param,
                "qubit_logical": i % ansatz.num_qubits,
                "qubit_physical": selected_qubits[i % ansatz.num_qubits] if selected_qubits else None,
                "layer": i // ansatz.num_qubits,
                "description": f"Ry rotation on logical qubit {i % ansatz.num_qubits} (physical qubit {selected_qubits[i % ansatz.num_qubits] if selected_qubits else None}) in layer {i // ansatz.num_qubits}",
                "gate_type": "Ry",
                "radians": param,
                "degrees": param * 180 / np.pi
            }
        return formatted_params
        # --- Modo cuántico (VQE manual) ---
        # print(f"Number of qubits (binary variables) used: {n_qubits}")
        # print(f"Modo: Optimización cuántica (VQE manual con scipy.optimize.{optimizer})")
        # reps = 16  # Mayor profundidad para más entrelazamiento
        # ansatz = RealAmplitudes(n_qubits, reps=reps, entanglement='full')
        # fig = ansatz.draw(output='mpl', filename='circuito_ansatz.png', style='iqx')
        # plt.show()
        # print(ansatz.draw(output='text'))
        # initial_point = 2 * np.pi * np.random.random(ansatz.num_parameters) - np.pi
        # selected_qubits = list(range(n_qubits))
        # cobyla_params = {
        #     'maxiter': maxiter_cobyla,
        #     'rhobeg': 1.0,
        #     'disp': True,
        #     'catol': 1e-4
        # }
        # history = []
        # def evaluate_expectation_shots(params):
        #     sequence = self.params_to_sequence(params, peptide_length, aa_list)
        #     return self.calculate_energy(sequence, constraints)
        # def store_intermediate_result(xk):
        #     val = evaluate_expectation_shots(xk)
        #     history.append(val)
        # start_time = time.time()
        # if optimizer.upper() == 'COBYLA':
        #     raw_result = minimize(
        #         fun=evaluate_expectation_shots,
        #         x0=list(initial_point),
        #         method='COBYLA',
        #         options=cobyla_params,
        #         callback=store_intermediate_result
        #     )
        # elif optimizer.lower() == 'differential_evolution':
        #     bounds = [(-2*np.pi, 2*np.pi)] * ansatz.num_parameters
        #     raw_result = differential_evolution(
        #         func=evaluate_expectation_shots,
        #         bounds=bounds,
        #         maxiter=maxiter_cobyla,
        #         callback=lambda xk, convergence: history.append(evaluate_expectation_shots(xk))
        #     )
        # else:
        #     raise ValueError(f"Optimizador no soportado: {optimizer}")
        # if history:
        #     plt.figure()
        #     plt.plot(history, marker='o')
        #     plt.xlabel('Iteración')
        #     plt.ylabel('Valor función objetivo')
        #     plt.title(f'Proceso de minimizado {optimizer}')
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        # best_sequence = self.params_to_sequence(raw_result.x, peptide_length, aa_list)
        # best_bitstring = '1' * n_qubits
        # circuit_transpiled = ansatz
        # vqe_result = VQEResult(
        #     optimize_result=raw_result,
        #     best_bitstring=best_bitstring,
        #     selected_qubits=selected_qubits,
        #     ansatz=ansatz,
        #     initial_point=initial_point,
        #     cobyla_params=cobyla_params,
        #     circuit_transpiled=circuit_transpiled,
        #     start_time=start_time
        # )
        # return [{
        #     'sequence': best_sequence,
        #     'energy': raw_result.fun,
        #     'sample': {},
        #     'vqe_result': vqe_result
        # }]

class PeptideDesigner:
    """Main peptide design framework"""
    
    def __init__(self, use_quantum: bool = True):
        self.use_quantum = use_quantum
        self.quantum_optimizer = QuantumOptimizer(use_simulator=not use_quantum)
        self.aa_props = AminoAcid_Properties()
    
    def design_peptides(self, target: ProteinTarget, peptide_length: int = 10,
                       num_candidates: int = 10, constraints: Optional[Dict] = None,
                       use_classical: bool = False, maxiter_cobyla: int = 100, optimizer: str = 'COBYLA', use_qaoa: bool = True) -> List[PeptideCandidate]:
        """Design peptides for a given target"""
        logger.info(f"Designing peptides for target: {target.name}")
        
        if constraints is None:
            constraints = self._default_constraints(target)
        
        # Quantum optimization
        solutions = self.quantum_optimizer.optimize_peptide(
            peptide_length, constraints, num_reads=num_candidates * 100,
            use_classical=use_classical, maxiter_cobyla=maxiter_cobyla, optimizer=optimizer, use_qaoa=use_qaoa
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

class PeptideAnalyzer:
    """Analyze and visualize peptide candidates"""
    
    def __init__(self):
        self.aa_props = AminoAcid_Properties()
    
    def analyze_candidates(self, candidates: List[PeptideCandidate]) -> Dict:
        """Comprehensive analysis of peptide candidates"""
        analysis = {
            'summary': self._generate_summary(candidates),
            'composition': self._analyze_composition(candidates),
            'properties': self._analyze_properties(candidates),
            'clustering': self._cluster_candidates(candidates)
        }
        return analysis
    
    def _generate_summary(self, candidates: List[PeptideCandidate]) -> Dict:
        """Generate summary statistics"""
        if not candidates:
            return {}
        
        scores = {
            'binding_affinity': [c.binding_affinity for c in candidates],
            'stability': [c.stability for c in candidates],
            'druggability': [c.druggability for c in candidates],
            'diversity_score': [c.diversity_score for c in candidates],
            'total_score': [c.total_score for c in candidates]
        }
        
        summary = {}
        for metric, values in scores.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return summary 
    
    def _analyze_composition(self, candidates: List[PeptideCandidate]) -> Dict:
        """Analyze amino acid composition"""
        all_sequences = [c.sequence for c in candidates]
        aa_counts = defaultdict(int)
        
        for sequence in all_sequences:
            for aa in sequence:
                aa_counts[aa] += 1
        
        total_count = sum(aa_counts.values())
        composition = {aa: count / total_count for aa, count in aa_counts.items()}
        
        return {
            'frequencies': composition,
            'most_common': sorted(composition.items(), key=lambda x: x[1], reverse=True)[:5],
            'least_common': sorted(composition.items(), key=lambda x: x[1])[:5]
        }
    
    def _analyze_properties(self, candidates: List[PeptideCandidate]) -> Dict:
        """Analyze physicochemical properties"""
        properties = {
            'hydrophobicity': [],
            'charge': [],
            'molecular_weight': [],
            'isoelectric_point': []
        }
        
        for candidate in candidates:
            seq = candidate.sequence
            
            # Calculate properties
            hydro = np.mean([self.aa_props.hydrophobicity.get(aa, 0) for aa in seq])
            charge = sum(self.aa_props.charge.get(aa, 0) for aa in seq)
            mw = sum(110.0 for _ in seq)  # Approximate
            
            properties['hydrophobicity'].append(hydro)
            properties['charge'].append(charge)
            properties['molecular_weight'].append(mw)
            properties['isoelectric_point'].append(7.0)  # Placeholder
        
        return properties
    
    def _cluster_candidates(self, candidates: List[PeptideCandidate]) -> Dict:
        """Cluster candidates by similarity"""
        # Simple clustering based on sequence similarity
        clusters = []
        processed = set()
        
        for i, candidate in enumerate(candidates):
            if i in processed:
                continue
            
            cluster = [candidate]
            processed.add(i)
            
            for j, other in enumerate(candidates[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(candidate.sequence, other.sequence)
                if similarity > 0.6:  # Threshold
                    cluster.append(other)
                    processed.add(j)
            
            clusters.append(cluster)
        print('NUMERO DE CLUSTER: ', len(clusters), clusters)
        return {
            'num_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'clusters': clusters
        }
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity"""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def visualize_results(self, candidates: List[PeptideCandidate], 
                         analysis: Dict, save_path: str = None):
        """Create visualizations of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score distribution
        scores = [c.total_score for c in candidates]
        axes[0, 0].hist(scores, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Total Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Property correlation
        properties = analysis['properties']
        axes[0, 1].scatter(properties['hydrophobicity'], properties['charge'], 
                          alpha=0.7, color='coral')
        axes[0, 1].set_title('Hydrophobicity vs Charge')
        axes[0, 1].set_xlabel('Hydrophobicity')
        axes[0, 1].set_ylabel('Charge')
        
        # 3. Amino acid composition
        composition = analysis['composition']['frequencies']
        aa_list = list(composition.keys())
        frequencies = list(composition.values())
        axes[1, 0].bar(aa_list, frequencies, color='lightgreen')
        axes[1, 0].set_title('Amino Acid Composition')
        axes[1, 0].set_xlabel('Amino Acid')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Multi-metric comparison
        metrics = ['binding_affinity', 'stability', 'druggability', 'diversity_score']
        metric_data = [[getattr(c, metric) for c in candidates] for metric in metrics]
        
        axes[1, 1].boxplot(metric_data, labels=metrics)
        axes[1, 1].set_title('Multi-metric Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    import argparse
    print("🧬 Quantum Peptide Design Framework")
    print("=====================================")
    parser = argparse.ArgumentParser(description="Peptide Design: Quantum or Classical")
    parser.add_argument('--mode', choices=['quantum', 'classical'], default='quantum', help='Modo de resolución: quantum o classical')
    parser.add_argument('--maxiter_cobyla', type=int, default=100, help='Número de iteraciones para COBYLA (solo modo quantum)')
    parser.add_argument('--num_candidates', type=int, default=10, help='Número de péptidos candidatos a obtener')
    parser.add_argument('--optimizer', choices=['COBYLA', 'differential_evolution'], default='COBYLA', help='Optimizador a usar en modo quantum')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_qaoa', dest='use_qaoa', action='store_true', help='Usar QAOA')
    group.add_argument('--no_use_qaoa', dest='use_qaoa', action='store_false', help='No usar QAOA (usar VQE)')
    parser.set_defaults(use_qaoa=True)

    args = parser.parse_args()

    # Define target protein
    target = ProteinTarget(
        name="SARS-CoV-2 Spike RBD",
        sequence="NLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP"
    )

    # Initialize designer
    designer = PeptideDesigner(use_quantum=(args.mode == 'quantum'))

    # Diseñar péptidos candidatos
    candidates = designer.design_peptides(
        target,
        peptide_length=4,
        num_candidates=args.num_candidates,
        constraints=None,
        use_classical=(args.mode == 'classical'),
        maxiter_cobyla=args.maxiter_cobyla,
        optimizer=args.optimizer,
        use_qaoa = args.use_qaoa
    )

    # Imprimir resultados
    print("\nPeptide Candidates:")
    for i, cand in enumerate(candidates, 1):
        print(f"{i:2d}. Seq: {cand.sequence} | Binding: {cand.binding_affinity:.3f} | Stability: {cand.stability:.3f} | Druggability: {cand.druggability:.3f} | Diversity: {cand.diversity_score:.3f} | Total: {cand.total_score:.3f}")

    # Analizar y mostrar resumen
    analyzer = PeptideAnalyzer()
    analysis = analyzer.analyze_candidates(candidates)
    print("\nSummary:")
    for metric, stats in analysis['summary'].items():
        print(f"{metric.capitalize():16s} -> Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, Min: {stats['min']:.3f}, Max: {stats['max']:.3f}, Median: {stats['median']:.3f}")

    # (Opcional) Visualizar resultados
    # analyzer.visualize_results(candidates, analysis)

if __name__ == '__main__':
    main() 
