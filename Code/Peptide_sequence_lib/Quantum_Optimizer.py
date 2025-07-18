
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from Peptide_sequence_lib.AminoAcid_Properties import AminoAcid_Properties
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
import logging
import time
import qubovert as qv
class QuantumOptimizer:
    """Quantum optimizer using VQE (Qiskit 2.x, manual classical optimizer) - Improved Version"""
    
    def __init__(self, use_simulator: bool = True, seed: int = 42):
        self.use_simulator = use_simulator
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        
    def calculate_energy(self, sequence: str, constraints: Dict) -> float:
        """
        Calcula la energía de una secuencia de péptido usando las mismas reglas que el modo cuántico.
        Versión mejorada con mejor balance de términos.
        """
        aa_props = AminoAcid_Properties()
        energy = 0.0
        
        # Binding affinity term (hidrofobicidad) - Mejorado
        hydrophobic_reward = constraints.get('hydrophobic_reward', 1.0)
        hydrophobic_energy = 0.0
        for aa in sequence:
            hydrophobic_energy -= hydrophobic_reward * aa_props.hydrophobicity.get(aa, 0)
        energy += hydrophobic_energy
        
        # Stability term (penaliza combinaciones inestables) - Mejorado
        stability_penalty = constraints.get('stability_penalty', 1.0)
        stability_energy = 0.0
        for i in range(len(sequence) - 1):
            aa1, aa2 = sequence[i], sequence[i+1]
            
            # Penalización por repetición
            if aa1 == aa2:
                stability_energy += stability_penalty
            
            # Penalización adicional por patrones desfavorables
            unfavorable_pairs = constraints.get('unfavorable_pairs', {})
            pair = aa1 + aa2
            if pair in unfavorable_pairs:
                stability_energy += unfavorable_pairs[pair] * stability_penalty
        
        energy += stability_energy
        
        # Diversity term - Mejorado
        diversity_reward = constraints.get('diversity_reward', 0.5)
        unique_aa = len(set(sequence))
        max_diversity = min(len(sequence), len(aa_props.amino_acids))
        diversity_ratio = unique_aa / max_diversity
        energy -= diversity_reward * diversity_ratio * len(sequence)
        
        # Término de longitud óptima
        target_length = constraints.get('target_length', len(sequence))
        length_penalty = constraints.get('length_penalty', 0.1)
        energy += length_penalty * abs(len(sequence) - target_length)
        
        return energy

    def params_to_sequence(self, params: np.ndarray, peptide_length: int, aa_list: List[str]) -> str:
        """
        Mapea los parámetros del ansatz a una secuencia usando softmax por posición.
        Versión mejorada con mejor normalización.
        """
        params = np.array(params)
        n_aa = len(aa_list)
        
        # Asegurar que tenemos suficientes parámetros
        required_params = peptide_length * n_aa
        if len(params) < required_params:
            params = np.pad(params, (0, required_params - len(params)), mode='constant')
        
        params = params[:required_params].reshape((peptide_length, n_aa))
        
        # Softmax mejorado con temperatura
        temperature = 1.0
        exp_params = np.exp((params - np.max(params, axis=1, keepdims=True)) / temperature)
        probs = exp_params / np.sum(exp_params, axis=1, keepdims=True)
        
        # Selección determinística (argmax) o estocástica
        use_stochastic = False  # Cambiar a True para selección estocástica
        if use_stochastic:
            indices = [np.random.choice(n_aa, p=prob_dist) for prob_dist in probs]
        else:
            indices = np.argmax(probs, axis=1)
        
        sequence = ''.join([aa_list[i] for i in indices])
        return sequence

    def optimize_peptide(self, 
                        peptide_length: int, 
                        constraints: Dict, 
                        num_reads: int = 10000, 
                        use_classical: bool = False, 
                        maxiter_cobyla: int = 100, 
                        optimizer: str = 'COBYLA', 
                         algorithm: str = 'vqe')-> List[Dict]:
        """
        Optimiza el péptido usando método cuántico o clásico.
        """
        start_time = time.time()
        print('METODOS: ',optimizer,algorithm)
        # Validación de parámetros
        if peptide_length <= 0:
            raise ValueError("peptide_length debe ser positivo")
        
        if not isinstance(constraints, dict):
            raise ValueError("constraints debe ser un diccionario")
        
        n_aminoacids = len(AminoAcid_Properties().amino_acids)
        aa_list = AminoAcid_Properties().amino_acids
        
        if use_classical:
            return self._optimize_classical(peptide_length, constraints, num_reads, aa_list)
        
        if algorithm == 'qaoa':
            return self._optimize_qaoa_improved(peptide_length, constraints, maxiter_cobyla, aa_list)
        elif algorithm == 'vqe':
            return self._optimize_vqe(peptide_length, constraints, maxiter_cobyla, optimizer, aa_list)
        elif algorithm == 'annealer':
            return self._optimize_annealer(peptide_length, constraints, aa_list)
        else:
            raise ValueError(f"Algoritmo desconocido: {algorithm}")

    def _optimize_classical(self, peptide_length: int, constraints: Dict, num_reads: int, aa_list: List[str]) -> List[Dict]:
        """Optimización clásica mejorada con mejor sampling"""
        self.logger.info(f"Modo: Optimización clásica (random search reproducible, {num_reads} muestras)")
        
        solutions = []
        
        # Estrategia 1: Búsqueda completamente aleatoria (70%)
        random_samples = int(0.7 * num_reads)
        for _ in range(random_samples):
            sequence = ''.join(np.random.choice(aa_list, peptide_length))
            energy = self.calculate_energy(sequence, constraints)
            solutions.append({
                'sequence': sequence,
                'energy': energy,
                'sample': {},
                'method': 'random'
            })
        
        # Estrategia 2: Búsqueda sesgada hacia aminoácidos hidrofóbicos (30%)
        if 'hydrophobic_reward' in constraints and constraints['hydrophobic_reward'] > 0:
            biased_samples = num_reads - random_samples
            # Crear pesos basados en hidrofobicidad
            aa_props = AminoAcid_Properties()
            weights = np.array([max(0, aa_props.hydrophobicity.get(aa, 0)) for aa in aa_list])
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(aa_list)) / len(aa_list)
            
            for _ in range(biased_samples):
                sequence = ''.join(np.random.choice(aa_list, peptide_length, p=weights))
                energy = self.calculate_energy(sequence, constraints)
                solutions.append({
                    'sequence': sequence,
                    'energy': energy,
                    'sample': {},
                    'method': 'biased_hydrophobic'
                })
        
        solutions.sort(key=lambda x: x['energy'])
        return solutions

    def _optimize_qaoa_improved(self, peptide_length: int, constraints: Dict, maxiter: int, aa_list: List[str]) -> List[Dict]:
        """
        Modo QAOA mejorado con función de coste específica y optimización differential evolution.
        Si el número de qubits es mayor a 30, se usa un QAOA temporal simplificado.
        """
        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit.circuit.library import QAOAAnsatz
            from qiskit.primitives import Estimator
            from qiskit_aer import Aer
            from scipy.optimize import differential_evolution
        except ImportError as e:
            self.logger.error(f"Librerías de Qiskit no disponibles: {e}")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        n_blocks = peptide_length
        n_aa = len(aa_list)
        n_qubits = n_blocks * n_aa

        if n_qubits > 30:
            self.logger.info(f"Qubits = {n_qubits} > 30, ejecutando QAOA temporal (tmp).")
            return self._optimize_qaoa_tmp(peptide_length, constraints, maxiter, aa_list)

        self.logger.info("Modo: Optimización cuántica (QAOA con función de coste y differential evolution)")
        print('QAOA USED')

        # Construir QUBO mejorado
        linear, quadratic = self._build_improved_qubo(n_blocks, n_aa, aa_list, constraints)

        # Crear problema cuadrático
        qp = QuadraticProgram()
        for i in range(n_qubits):
            qp.binary_var(name=f'x_{i}')
        qp.minimize(linear=linear, quadratic=quadratic)

        # Convertir a QUBO
        from qiskit_optimization.converters import QuadraticProgramToQubo
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)

        # Crear circuito QAOA
        p = 1  # profundidad del ansatz
        qaoa_ansatz = QAOAAnsatz(qubo, reps=p)

        # Configurar Estimator backend
        backend = Aer.get_backend('aer_simulator_statevector')
        estimator = Estimator(backend=backend)

        def cost_function(params):
            # params es un array de ángulos del ansatz
            qc = qaoa_ansatz.assign_parameters(params)
            result = estimator.run(qc).result()
            energy = result.values[0]
            return energy

        # Optimización con differential evolution
        bounds = [(0, 2 * 3.141592653589793)] * qaoa_ansatz.num_parameters

        result = differential_evolution(cost_function, bounds, maxiter=maxiter)

        best_params = result.x
        best_energy = result.fun

        # Interpretar resultado: decodificar la mejor configuración (binaria)
        qc_best = qaoa_ansatz.assign_parameters(best_params)
        counts = estimator.run(qc_best).result().quasi_dists[0]  # quasi-distribution
        # Buscar la configuración con mayor probabilidad
        best_bitstring = max(counts, key=counts.get)

        # Convertir bitstring a secuencia de aminoácidos
        sequence = self._bitstring_to_sequence(best_bitstring, peptide_length, aa_list)

        return [{
            'sequence': sequence,
            'energy': best_energy,
            'sample': {'bitstring': best_bitstring},
            'method': 'qaoa_improved',
            'optimizer': 'differential_evolution',
            'result': result
        }]



    def _optimize_qaoa_tmp(self, peptide_length: int, constraints: Dict, maxiter: int, aa_list: List[str]) -> List[Dict]:
        """
        Implementación simplificada de QAOA para fallback cuando hay más de 30 qubits.
        Aquí puedes usar un método clásico, un ansatz más simple, o simplemente devolver una solución básica.
        """
        self.logger.info("Ejecutando QAOA temporal simplificado para muchos qubits (>30).")

        # Por ejemplo, llamar a la optimización clásica como fallback:
        return self._optimize_classical(peptide_length, constraints, maxiter, aa_list)


    def _build_improved_qubo(self, n_blocks: int, n_aa: int, aa_list: List[str], constraints: Dict) -> Tuple[Dict, Dict]:
        """
        Construye un QUBO mejorado con mejor representación del problema.
        """
        linear = {}
        quadratic = {}
        
        # Penalización one-hot más fuerte
        penalty = constraints.get('one_hot_penalty', 10.0)
        
        # Restricciones one-hot por posición
        for p in range(n_blocks):
            # Término lineal: -penalty * sum(x_{p,a})
            for a in range(n_aa):
                idx = p * n_aa + a
                linear[idx] = linear.get(idx, 0) - penalty
            
            # Término cuadrático: penalty * sum_{a<b} x_{p,a} * x_{p,b}
            for a in range(n_aa):
                for b in range(a + 1, n_aa):
                    idx1 = p * n_aa + a
                    idx2 = p * n_aa + b
                    quadratic[(idx1, idx2)] = quadratic.get((idx1, idx2), 0) + penalty
        
        # Términos de energía mejorados
        aa_props = AminoAcid_Properties()
        
        # Término de hidrofobicidad
        hydrophobic_reward = constraints.get('hydrophobic_reward', 1.0)
        for p in range(n_blocks):
            for a in range(n_aa):
                idx = p * n_aa + a
                hydrophobic_contrib = hydrophobic_reward * aa_props.hydrophobicity.get(aa_list[a], 0)
                linear[idx] = linear.get(idx, 0) - hydrophobic_contrib
        
        # Término de estabilidad (interacciones entre posiciones adyacentes)
        stability_penalty = constraints.get('stability_penalty', 1.0)
        for p in range(n_blocks - 1):
            for a in range(n_aa):
                for b in range(n_aa):
                    idx1 = p * n_aa + a
                    idx2 = (p + 1) * n_aa + b
                    
                    # Penalizar repeticiones
                    if a == b:
                        quadratic[(idx1, idx2)] = quadratic.get((idx1, idx2), 0) + stability_penalty
        
        return linear, quadratic

    def _bitstring_to_sequence(self, bitstring: str, n_blocks: int, n_aa: int, aa_list: List[str]) -> str:
        """
        Convierte un bitstring a secuencia de aminoácidos.
        Maneja casos donde no hay exactamente un bit activo por bloque.
        """
        sequence = []
        bitstring_reversed = bitstring[::-1]  # Qiskit usa little-endian
        
        for block in range(n_blocks):
            block_bits = bitstring_reversed[block * n_aa:(block + 1) * n_aa]
            
            # Buscar el primer bit activo
            active_indices = [i for i, bit in enumerate(block_bits) if bit == '1']
            
            if len(active_indices) == 1:
                # Caso ideal: exactamente un bit activo
                sequence.append(aa_list[active_indices[0]])
            elif len(active_indices) > 1:
                # Múltiples bits activos: tomar el primero
                sequence.append(aa_list[active_indices[0]])
            else:
                # Ningún bit activo: usar aminoácido por defecto
                sequence.append(aa_list[0])  # 'A' por defecto
        
        return ''.join(sequence)
    def _optimize_vqe(self, peptide_length: int, constraints: Dict, maxiter: int, optimizer: str, aa_list: List[str]) -> List[Dict]:
        """
        Optimización VQE cuántica real usando Qiskit, RealAmplitudes y Estimator.
        """
        self.logger.info("Modo: Optimización VQE cuántico real")
        print('VQE QUANTUM USED')

        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.converters import QuadraticProgramToQubo
            from qiskit.circuit.library import RealAmplitudes
            from qiskit.primitives import Estimator
            from qiskit_aer import Aer
            from qiskit import QuantumCircuit
        except ImportError as e:
            self.logger.error(f"Qiskit requerido para VQE cuántico: {e}")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        n_blocks = peptide_length
        n_aa = len(aa_list)
        n_qubits = n_blocks * n_aa

        # 🚫 Verificación de qubits
        if n_qubits > 30:
            self.logger.error(f"Demasiados qubits para VQE cuántico: {n_qubits}. Máximo recomendado: 30")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        # Crear QUBO
        linear, quadratic = self._build_improved_qubo(n_blocks, n_aa, aa_list, constraints)

        qp = QuadraticProgram()
        for i in range(n_qubits):
            qp.binary_var(name=f'x_{i}')
        qp.minimize(linear=linear, quadratic=quadratic)

        # Convertir a Ising Hamiltonian
        try:
            qubo = QuadraticProgramToQubo().convert(qp)
            cost_operator, offset = qubo.to_ising()
        except Exception as e:
            self.logger.error(f"Fallo al convertir a operador Ising: {e}")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        # Crear ansatz
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement='linear')

        estimator = Estimator()

        def vqe_cost_function(params):
            try:
                result = estimator.run([ansatz], [cost_operator], [params]).result()
                return result.values[0] + offset
            except Exception as e:
                self.logger.warning(f"Error al evaluar VQE: {e}")
                return float('inf')

        # Optimización clásica
        bounds = [(-2*np.pi, 2*np.pi)] * ansatz.num_parameters
        best_result = differential_evolution(
            vqe_cost_function,
            bounds,
            maxiter=maxiter,
            seed=self.seed,
            atol=1e-6,
            tol=1e-6,
            polish=True,
            workers=1
        )

        if not best_result.success:
            self.logger.warning(f"VQE no convergió: {best_result.message}")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        # Construir circuito final
        qc = QuantumCircuit(n_qubits)
        qc.append(ansatz.assign_parameters(best_result.x), range(n_qubits))
        qc.measure_all()

        backend = Aer.get_backend('aer_simulator')
        job = backend.run(qc, shots=2048)
        counts = job.result().get_counts()

        # Analizar bitstrings
        best_solutions = []
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            if count / total_shots > 0.01:
                sequence = self._bitstring_to_sequence(bitstring, n_blocks, n_aa, aa_list)
                actual_energy = self.calculate_energy(sequence, constraints)
                best_solutions.append({
                    'sequence': sequence,
                    'energy': actual_energy,
                    'probability': count / total_shots,
                    'bitstring': bitstring
                })

        best_solutions.sort(key=lambda x: x['energy'])

        if not best_solutions:
            self.logger.error("No se obtuvieron soluciones válidas del circuito VQE")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        return [{
            'sequence': best_solutions[0]['sequence'],
            'energy': best_solutions[0]['energy'],
            'sample': counts,
            'method': 'vqe_quantum',
            'vqe_result': best_result,
            'optimizer': 'differential_evolution',
            'probability': best_solutions[0]['probability'],
            'bitstring': best_solutions[0]['bitstring'],
            'num_evaluations': best_result.nfev,
            'all_solutions': best_solutions[:5]
        }]


    def _smart_initialization(self, peptide_length: int, constraints: Dict, aa_list: List[str]) -> np.ndarray:
        """
        Inicialización inteligente de parámetros basada en las restricciones.
        """
        n_aa = len(aa_list)
        params = np.zeros(peptide_length * n_aa)
        
        # Sesgar hacia aminoácidos hidrofóbicos si es beneficial
        if constraints.get('hydrophobic_reward', 0) > 0:
            aa_props = AminoAcid_Properties()
            for pos in range(peptide_length):
                for aa_idx, aa in enumerate(aa_list):
                    param_idx = pos * n_aa + aa_idx
                    hydrophobic_value = aa_props.hydrophobicity.get(aa, 0)
                    params[param_idx] = hydrophobic_value * 0.5  # Sesgo suave
        
        return params
    
    
    
    
    
    
    #Annealer:


    def _optimize_annealer(self, peptide_length: int, constraints: dict, aa_list: list, num_anneals: int = 10000) -> list:
        """
        Resuelve el problema QUBO usando qubovert annealer.
        """
        self.logger.info("Modo: Optimización clásica con qubovert annealer")
        print('QUBO Annealer USED')

        n_blocks = peptide_length
        n_aa = len(aa_list)
        n_vars = n_blocks * n_aa

        # Construir QUBO (matrices linear y quadratic)
        linear, quadratic = self._build_improved_qubo(n_blocks, n_aa, aa_list, constraints)

        # Crear objeto QUBO qubovert
        Q = qv.QUBO()
        for i in range(n_vars):
            Q.create_var(f'x_{i}')

        # Agregar términos lineales
        for i, coeff in enumerate(linear):
            if coeff != 0:
                Q[(f'x_{i}',)] = coeff

        # Agregar términos cuadráticos
        for (i, j), coeff in quadratic.items():
            if coeff != 0:
                Q[(f'x_{i}', f'x_{j}')] = coeff

        # Ejecutar el annealer
        anneal_res = qv.sim.anneal_pubo(Q, num_anneals=num_anneals)

        best_solutions = []
        # anneal_res es una lista de AnnealResult con .state y .value
        for result in anneal_res:
            state_dict = result.state      # dict de nombres de variable → 0/1
            qubo_energy = result.value     # energía QUBO
            # construir bitstring ordenado
            bitstring = ''.join(str(state_dict[f'x_{i}']) for i in range(n_vars))
            sequence = self._bitstring_to_sequence(bitstring, n_blocks, n_aa, aa_list)
            actual_energy = self.calculate_energy(sequence, constraints)
            best_solutions.append({
                'sequence': sequence,
                'energy': actual_energy,
                'bitstring': bitstring,
                'qubo_energy': qubo_energy
            })

        # Ordenar según energía real
        best_solutions.sort(key=lambda x: x['energy'])

        if not best_solutions:
            self.logger.error("No se obtuvieron soluciones válidas del annealer")
            return self._optimize_classical(peptide_length, constraints, 1000, aa_list)

        # Devolver la mejor y las top‑5
        return [{
            'sequence': best_solutions[0]['sequence'],
            'energy': best_solutions[0]['energy'],
            'method': 'qubovert_annealer',
            'all_solutions': best_solutions[:5],
        }]
