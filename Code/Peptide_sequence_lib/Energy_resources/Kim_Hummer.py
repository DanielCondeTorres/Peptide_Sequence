import numpy as np
from typing import Dict, List, Tuple, Optional

import numpy as np
import qiskit
from .interaction import Interaction
from .energy_matrix_loader import (
    _load_energy_matrix_file,
)

import numpy as np
from typing import Dict, List, Tuple, Optional

# Datos de aminoácidos Kim-Hummer
KimHummer_Values = {
    'ASP': {'label':'D','radii': 5.6,'behavior': 'acidic'},
    'GLU': {'label':'E','radii': 5.9,'behavior': 'acidic'},
    'LYS': {'label':'K','radii': 6.4,'behavior': 'basic'},
    'ARG': {'label':'R','radii': 6.6,'behavior': 'basic'},
    'HIS': {'label':'H','radii': 6.1,'behavior': 'basic'},
    'HISH': {'label':'H','radii': 6.1,'behavior': 'basic'},
    'GLY': {'label':'G','radii': 4.5,'behavior': 'hydrophobic'},
    'ALA': {'label':'A','radii': 5.0,'behavior': 'hydrophobic'},
    'VAL': {'label':'V','radii': 5.9,'behavior': 'hydrophobic'},
    'LEU': {'label':'L','radii': 6.2,'behavior': 'hydrophobic'},
    'ILE': {'label':'I','radii': 6.2,'behavior': 'hydrophobic'},
    'PRO': {'label':'P','radii': 5.6,'behavior': 'hydrophobic'},
    'PHE': {'label':'F','radii': 6.4,'behavior': 'hydrophobic'},
    'MET': {'label':'M','radii': 6.2,'behavior': 'hydrophobic'},
    'TRP': {'label':'W','radii': 6.8,'behavior': 'hydrophobic'},
    'SER': {'label':'S','radii': 5.2,'behavior': 'polar'},
    'THR': {'label':'T','radii': 5.6,'behavior': 'polar'},
    'CYS': {'label':'C','radii': 5.5,'behavior': 'polar'},
    'TYR': {'label':'Y','radii': 6.5,'behavior': 'polar'},
    'ASN': {'label':'N','radii': 5.7,'behavior': 'polar'},
    'GLN': {'label':'Q','radii': 6.0,'behavior': 'polar'}
}

# Diccionario de mapeo de 1-letra a 3-letras
one_to_three = {v['label']: k for k, v in KimHummer_Values.items()}

class AminoAcidClusterer:
    """Clase para hacer clustering de aminoácidos según el paper."""
    
    def __init__(self, mj_interaction: np.ndarray, amino_acids_list: List[str]):
        self.mj_interaction = mj_interaction
        self.amino_acids_list = amino_acids_list
        self.n_aa = len(amino_acids_list)
        
        # Crear matriz de energía simétrica
        self.energy_matrix = np.zeros((self.n_aa, self.n_aa))
        for i in range(self.n_aa):
            for j in range(self.n_aa):
                self.energy_matrix[i, j] = mj_interaction[min(i, j), max(i, j)]
        
        # Crear vector sigma
        self.sigma_vector = np.array([
            KimHummer_Values[one_to_three[aa]]['radii'] for aa in amino_acids_list
        ])
        
        # Cache para clustering
        self._clustering_cache = {}
    
    def calculate_loss_function(self, assignment: np.ndarray, D: int) -> float:
        """Calcula la función de pérdida L = Σ(eij - e'_a(i)a(j))^2"""
        effective_matrix = self._create_effective_energy_matrix(assignment, D)
        
        loss = 0.0
        for i in range(self.n_aa):
            for j in range(self.n_aa):
                cluster_i = assignment[i]
                cluster_j = assignment[j]
                loss += (self.energy_matrix[i, j] - effective_matrix[cluster_i, cluster_j])**2
        
        return loss
    
    def _create_effective_energy_matrix(self, assignment: np.ndarray, D: int) -> np.ndarray:
        """Crea matriz efectiva e'ij según ecuación (9)"""
        effective_matrix = np.zeros((D, D))
        
        for i in range(D):
            for j in range(D):
                cluster_i_indices = np.where(assignment == i)[0]
                cluster_j_indices = np.where(assignment == j)[0]
                
                if len(cluster_i_indices) > 0 and len(cluster_j_indices) > 0:
                    # Ecuación (9): e'ij = (1/N) Σ ekl δa(k),i δa(l),j
                    total = sum(self.energy_matrix[k, l] for k in cluster_i_indices for l in cluster_j_indices)
                    count = len(cluster_i_indices) * len(cluster_j_indices)
                    effective_matrix[i, j] = total / count
        
        return effective_matrix
    
    def _create_effective_sigma_vector(self, assignment: np.ndarray, D: int) -> np.ndarray:
        """Crea vector sigma efectivo según ecuación (10)"""
        effective_sigma = np.zeros(D)
        
        for i in range(D):
            cluster_indices = np.where(assignment == i)[0]
            if len(cluster_indices) > 0:
                # Ecuación (10): σ'i = (1/N) Σ δa(k),i σk
                effective_sigma[i] = np.mean([self.sigma_vector[k] for k in cluster_indices])
        
        return effective_sigma
    
    def optimize_clustering(self, D: int, max_iter: int = 100, n_trials: int = 5) -> Dict:
        """Optimiza el clustering para D clusters"""
        # Usar cache si ya se calculó
        if D in self._clustering_cache:
            return self._clustering_cache[D]
        
        best_assignment = None
        best_loss = float('inf')
        
        for trial in range(n_trials):
            # Inicialización aleatoria
            assignment = np.random.randint(0, D, size=self.n_aa)
            
            # Optimización local
            for iteration in range(max_iter):
                improved = False
                
                for aa_idx in range(self.n_aa):
                    current_cluster = assignment[aa_idx]
                    current_loss = self.calculate_loss_function(assignment, D)
                    
                    # Probar mover a cada otro cluster
                    for new_cluster in range(D):
                        if new_cluster != current_cluster:
                            assignment[aa_idx] = new_cluster
                            new_loss = self.calculate_loss_function(assignment, D)
                            
                            if new_loss < current_loss:
                                current_loss = new_loss
                                improved = True
                            else:
                                assignment[aa_idx] = current_cluster
                
                if not improved:
                    break
            
            # Evaluar esta solución
            final_loss = self.calculate_loss_function(assignment, D)
            if final_loss < best_loss:
                best_loss = final_loss
                best_assignment = assignment.copy()
        
        # Crear matrices efectivas
        effective_energy = self._create_effective_energy_matrix(best_assignment, D)
        effective_sigma = self._create_effective_sigma_vector(best_assignment, D)
        
        # Crear mapeo de aminoácidos a clusters
        aa_to_cluster = {}
        for i, aa in enumerate(self.amino_acids_list):
            aa_to_cluster[aa] = best_assignment[i]
        
        result = {
            'assignment': best_assignment,
            'aa_to_cluster': aa_to_cluster,
            'effective_energy_matrix': effective_energy,
            'effective_sigma_vector': effective_sigma,
            'loss': best_loss,
            'D': D
        }
        
        # Guardar en cache
        self._clustering_cache[D] = result
        return result

class KimHummerInteraction:
    """A class defining a Kim-Hummer interaction between beads of a peptide.
    Y. C. Kim and G. Hummer, Coarse-grained models for simulations of multiprotein complexes: application to ubiquitin binding, Journal of Molecular Biology 375, 1416 (2008)."""
    def __init__(self):
        self.mj_interaction, self.list_aa = _load_energy_matrix_file()
        self.KimHummer_Values = KimHummer_Values
        self.one_to_three = one_to_three
        self.clusterer = None
        self.clustering_result = None

    def calculate_energy_matrix(self, residue_sequence: str, lamda: float = 0.159, T: float = 273, kb: float = 1.380649 * 10**-23, D: int = None) -> np.ndarray:
        chain_len = len(residue_sequence)
        pair_energies = np.zeros((chain_len + 1, 2, chain_len + 1, 2))

        # Clustering opcional
        if D is not None:
            if self.clusterer is None:
                self.clusterer = AminoAcidClusterer(self.mj_interaction, self.list_aa)
            if self.clustering_result is None or self.clustering_result['D'] != D:
                self.clustering_result = self.clusterer.optimize_clustering(D)

        for i in range(1, chain_len + 1):
            for j in range(i + 1, chain_len + 1):
                # Convertir de 1-letra a 3-letras
                aa_i_name = self.one_to_three[residue_sequence[i - 1]]
                aa_j_name = self.one_to_three[residue_sequence[j - 1]]

                if D is not None:
                    # Usar el label de 1 letra para acceder a aa_to_cluster
                    aa_i_label = self.KimHummer_Values[aa_i_name]['label']
                    aa_j_label = self.KimHummer_Values[aa_j_name]['label']
                    cluster_i = self.clustering_result['aa_to_cluster'][aa_i_label]
                    cluster_j = self.clustering_result['aa_to_cluster'][aa_j_label]
                    e_ij = self.clustering_result['effective_energy_matrix'][cluster_i, cluster_j]
                    sigma_i = self.clustering_result['effective_sigma_vector'][cluster_i]
                    sigma_j = self.clustering_result['effective_sigma_vector'][cluster_j]
                    sigma_i_j = 0.5 * (sigma_i + sigma_j)
                else:
                    # Buscar el índice usando el label de 1 letra
                    aa_i_label = self.KimHummer_Values[aa_i_name]['label']
                    aa_j_label = self.KimHummer_Values[aa_j_name]['label']
                    aa_i = self.list_aa.index(aa_i_label)
                    aa_j = self.list_aa.index(aa_j_label)
                    e_ij = self.mj_interaction[min(aa_i, aa_j), max(aa_i, aa_j)]
                    sigma_i_j = 0.5 * (self.KimHummer_Values[aa_i_name]['radii'] + self.KimHummer_Values[aa_j_name]['radii'])

                epsilon_ij = (e_ij - (-2.27*T*kb)) * lamda
                r_ij_0 = 2 ** (1/6) * sigma_i_j
                r = abs(i - j) # cambiar por distancia en qubits
                if r == 0:
                    r = 1e-6

                if epsilon_ij < 0:
                    pair_energies[i, 0, j, 0] = 4 * abs(epsilon_ij) * ((sigma_i_j/r) ** 12 - (sigma_i_j/r) ** 6)
                elif epsilon_ij >= 0 and r < r_ij_0:
                    pair_energies[i, 0, j, 0] = 4 * epsilon_ij * ((sigma_i_j/r) ** 12 - (sigma_i_j/r) ** 6) + 2 * epsilon_ij
                elif r > 8.5:
                    pair_energies[i, 0, j, 0] = 0
                else:
                    pair_energies[i, 0, j, 0] = -4 * epsilon_ij * ((sigma_i_j/r) ** 12 - (sigma_i_j/r) ** 6)
        return pair_energies


    



