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
    parser.add_argument('--algorithm', type=str, choices=['qaoa', 'vqe', 'annealer'], default='vqe',
                    help='Algoritmo cuántico a usar: qaoa, vqe o annealer (default: vqe)')


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
        algorithm=args.algorithm,
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
