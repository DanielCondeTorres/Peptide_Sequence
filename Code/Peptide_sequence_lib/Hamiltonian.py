print('Escribir Hamiltonino para interactuar peptido membrana')
from AminoAcid_Properties import AminoAcid_Properties
import matplotlib.pyplot as plt
import numpy as np 
import math
# Diseñar peptidos en helice alpha, que interactue paralelo a la membrana.
# Carga peptido, momento hidrofobico.
# 16 aminoacidos 
# Calcular el momento hidrofobico y electrostatico total del peptido con lo de Fabián.


def get_mj_energy(aa1, aa2):
    """Obtiene energía MJ entre dos aminoácidos"""
    key = tuple(sorted([aa1, aa2]))
    return  AminoAcid_Properties().mj_matrix_sample.get(key, -1.0)  # Valor por defecto

class MembraneHelixHamiltonian:
    def __init__(self, sequence_length, membrane_charge=0, membrane_depth_profile='interfacial',
                 w_helix=1.0, w_mj=1.0, w_hydrophobic=1.0, w_electrostatic=1.0, w_membrane_interaction=2.0):
        """
        Hamiltoniano para diseño de péptidos helicoidales que interactúan con membranas
        
        Args:
            sequence_length (int): Longitud de la secuencia
            membrane_charge (float): Carga neta de la membrana lipídica
            membrane_depth_profile (str): 'interfacial', 'transmembrane', 'surface'
            w_helix (float): Peso para la estabilidad de hélice
            w_mj (float): Peso para interacciones MJ
            w_hydrophobic (float): Peso para momento hidrofóbico
            w_electrostatic (float): Peso para momento electrostático
            w_membrane_interaction (float): Peso para interacción con membrana
        """
        self.L = sequence_length
        self.Y = membrane_charge  # Carga de la membrana
        self.membrane_profile = membrane_depth_profile
        self.w_helix = w_helix
        self.w_mj = w_mj
        self.w_hydrophobic = w_hydrophobic
        self.w_electrostatic = w_electrostatic
        self.w_membrane_interaction = w_membrane_interaction
        self.aa_props = AminoAcid_Properties()
        # Parámetros de hélice alfa
        self.helix_rise = 1.5  # Angstroms por residuo
        self.helix_twist = 100  # Grados por residuo (360/3.6)
        
        # Define el perfil de la membrana para cada posición
        self.membrane_environment = self._define_membrane_environment()
        
    def _define_membrane_environment(self):
        """
        Define el ambiente de membrana para cada posición del péptido
        Returns: list con el tipo de ambiente para cada posición
        """
        if self.membrane_profile == 'interfacial':
            # Péptido en la interfaz: N-terminal en cabezas polares, C-terminal en colas
            environment = []
            for i in range(self.L):
                if i < self.L // 3:
                    environment.append('polar_heads')  # Región de cabezas polares
                elif i < 2 * self.L // 3:
                    environment.append('interface')    # Región interfacial
                else:
                    environment.append('hydrophobic_tails')  # Región de colas hidrofóbicas
                    
        elif self.membrane_profile == 'transmembrane':
            # Péptido transmembrana: extremos en cabezas, centro en colas
            environment = []
            for i in range(self.L):
                if i < 2 or i >= self.L - 2:
                    environment.append('polar_heads')
                else:
                    environment.append('hydrophobic_tails')
                    
        elif self.membrane_profile == 'surface':
            # Péptido completamente en la superficie (cabezas polares)
            environment = ['polar_heads'] * self.L
            
        else:
            # Por defecto: interfacial
            environment = ['interface'] * self.L
            
        return environment
    
    def membrane_interaction_term(self, sequence):
        """
        Término de interacción específica con diferentes regiones de la membrana
        Considera la preferencia de cada aminoácido por cabezas polares vs colas hidrofóbicas
        """
        energy = 0
        
        for i, aa in enumerate(sequence):
            env = self.membrane_environment[i]
            
            if env == 'polar_heads':
                # Favorece aminoácidos polares/cargados que pueden interactuar con cabezas
                energy += self.aa_props.polar_head_affinity[aa]
                
            elif env == 'hydrophobic_tails':
                # Favorece aminoácidos hidrofóbicos que prefieren colas apolares
                energy += self.aa_props.hydrophobic_tail_affinity[aa]
                
            elif env == 'interface':
                # Región interfacial: promedio ponderado
                polar_contribution = 0.3 * self.aa_props.polar_head_affinity[aa]
                hydrophobic_contribution = 0.7 * self.aa_props.hydrophobic_tail_affinity[aa]
                energy += polar_contribution + hydrophobic_contribution
        
        return self.w_membrane_interaction * energy
    
    def membrane_depth_penalty(self, sequence):
        """
        Penalización adicional por aminoácidos en ambientes muy desfavorables
        """
        penalty = 0
        
        for i, aa in enumerate(sequence):
            env = self.membrane_environment[i]
            
            # Penalización extra para casos muy desfavorables
            if env == 'hydrophobic_tails' and aa in ['R', 'K', 'D', 'E']:
                penalty += 5.0  # Aminoácidos muy cargados en región hidrofóbica
            elif env == 'polar_heads' and aa in ['L', 'I', 'V', 'F'] and i < self.L//2:
                penalty += 2.0  # Aminoácidos muy hidrofóbicos en región polar (menos crítico)
        
        return penalty
        """
        Término de estabilidad de hélice alfa
        H_helix = Σ(i) w_helix * helix_propensity[i]
        """
        energy = 0
        for aa in sequence:
            energy += helix_propensity[aa]
        return self.w_helix * energy
    
    def miyazawa_jernigan_term(self, sequence):
        """
        Término de estabilidad interna usando matriz MJ
        H_MJ = Σ(i<j) w_mj * MJ(i,j) * contact_function(i,j)
        """
        energy = 0
        for i in range(len(sequence)):
            for j in range(i+1, len(sequence)):
                # Factor de contacto en hélice (máximo para i,i+3 y i,i+4)
                contact_weight = self.helix_contact_probability(i, j)
                mj_energy = get_mj_energy(sequence[i], sequence[j])
                energy += mj_energy * contact_weight
        return self.w_mj * energy
    
    def helix_contact_probability(self, i, j):
        """
        Probabilidad de contacto en hélice alfa
        Máxima para separaciones i,i+3 y i,i+4
        """
        sep = abs(j - i)
        if sep == 3 or sep == 4:
            return 1.0
        elif sep == 1 or sep == 2:
            return 0.3
        else:
            return 0.1 * np.exp(-0.5 * (sep - 3.5)**2)
    
    def hydrophobic_moment_term(self, sequence):
        """
        Momento hidrofóbico de la hélice
        μH = |Σ(i) H(i) * exp(i * δ * i)|
        donde δ = 100° (ángulo por residuo en hélice alfa)
        """
        moment_real = 0
        moment_imag = 0
        
        for i, aa in enumerate(sequence):
            angle = math.radians(self.helix_twist * i)
            h_value = self.aa_props.hydrophobicity[aa]
            moment_real += h_value * math.cos(angle)
            moment_imag += h_value * math.sin(angle)
        
        magnitude = math.sqrt(moment_real**2 + moment_imag**2)
        # Penalizamos momento hidrofóbico bajo (queremos amphipathic helices)
        return -self.w_hydrophobic * magnitude
    
    def electrostatic_moment_term(self, sequence):
        """
        Momento electrostático de la hélice
        μE = |Σ(i) q(i) * exp(i * δ * i)|
        Incluye interacción con carga de membrana
        """
        moment_real = 0
        moment_imag = 0
        
        for i, aa in enumerate(sequence):
            angle = math.radians(self.helix_twist * i)
            charge = self.aa_props.charges[aa]
            moment_real += charge * math.cos(angle)
            moment_imag += charge * math.sin(angle)
        
        magnitude = math.sqrt(moment_real**2 + moment_imag**2)
        
        # Interacción con membrana: favorece momento que complementa carga Y
        membrane_interaction = -self.Y * magnitude  # Si Y>0, favorece momento negativo
        
        return self.w_electrostatic * (membrane_interaction - 0.1 * magnitude**2)
    
    def membrane_insertion_penalty(self, sequence):
        """
        Penalización por aminoácidos poco favorables para membrana
        """
        penalty = 0
        unfavorable_aas = ['D', 'E', 'K', 'R']  # Muy cargados
        for aa in sequence:
            if aa in unfavorable_aas:
                penalty += 2.0  # Penalización por carga en membrana
        return penalty
    
    def helix_stability_term(self, sequence):
        """
        Término de estabilidad de hélice alfa
        H_helix = Σ(i) w_helix * helix_propensity[i]
        """
        energy = 0
        for aa in sequence:
            energy += self.aa_props.helix_propensity[aa]
        return self.w_helix * energy
    
    def miyazawa_jernigan_term(self, sequence):
        """
        Término de estabilidad interna usando matriz MJ
        H_MJ = Σ(i<j) w_mj * MJ(i,j) * contact_function(i,j)
        """
        energy = 0
        for i in range(len(sequence)):
            for j in range(i+1, len(sequence)):
                # Factor de contacto en hélice (máximo para i,i+3 y i,i+4)
                contact_weight = self.helix_contact_probability(i, j)
                mj_energy = get_mj_energy(sequence[i], sequence[j])
                energy += mj_energy * contact_weight
        return self.w_mj * energy
    
    def helix_contact_probability(self, i, j):
        """
        Probabilidad de contacto en hélice alfa
        Máxima para separaciones i,i+3 y i,i+4
        """
        sep = abs(j - i)
        if sep == 3 or sep == 4:
            return 1.0
        elif sep == 1 or sep == 2:
            return 0.3
        else:
            return 0.1 * np.exp(-0.5 * (sep - 3.5)**2)
    
    def hydrophobic_moment_term(self, sequence):
        """
        Momento hidrofóbico de la hélice
        μH = |Σ(i) H(i) * exp(i * δ * i)|
        donde δ = 100° (ángulo por residuo en hélice alfa)
        """
        moment_real = 0
        moment_imag = 0
        
        for i, aa in enumerate(sequence):
            angle = math.radians(self.helix_twist * i)
            h_value = self.aa_props.hydrophobicity[aa]
            moment_real += h_value * math.cos(angle)
            moment_imag += h_value * math.sin(angle)
        
        magnitude = math.sqrt(moment_real**2 + moment_imag**2)
        # Penalizamos momento hidrofóbico bajo (queremos amphipathic helices)
        return -self.w_hydrophobic * magnitude
    
    def electrostatic_moment_term(self, sequence):
        """
        Momento electrostático de la hélice
        μE = |Σ(i) q(i) * exp(i * δ * i)|
        Incluye interacción con carga de membrana
        """
        moment_real = 0
        moment_imag = 0
        
        for i, aa in enumerate(sequence):
            angle = math.radians(self.helix_twist * i)
            charge = self.aa_props.charge[aa]
            moment_real += charge * math.cos(angle)
            moment_imag += charge * math.sin(angle)
        
        magnitude = math.sqrt(moment_real**2 + moment_imag**2)
        
        # Interacción con membrana: favorece momento que complementa carga Y
        membrane_interaction = -self.Y * magnitude  # Si Y>0, favorece momento negativo
        
        return self.w_electrostatic * (membrane_interaction - 0.1 * magnitude**2)
    
    def calculate_electrostatic_moment(self, sequence):
        """Calcula solo la magnitud del momento electrostático"""
        moment_real = 0
        moment_imag = 0
        
        for i, aa in enumerate(sequence):
            angle = math.radians(self.helix_twist * i)
            charge = self.aa_props.charge[aa]
            moment_real += charge * math.cos(angle)
            moment_imag += charge * math.sin(angle)
        
        return math.sqrt(moment_real**2 + moment_imag**2)
    
    def total_hamiltonian(self, sequence):
        """
        Hamiltoniano total para optimización de secuencia
        H_total = H_helix + H_MJ + H_hydrophobic + H_electrostatic + H_membrane_interaction + H_membrane_penalty
        """
        if len(sequence) != self.L:
            return float('inf')  # Secuencia de longitud incorrecta
        
        # Términos de energía
        h_helix = self.helix_stability_term(sequence)
        h_mj = self.miyazawa_jernigan_term(sequence)
        h_hydrophobic = self.hydrophobic_moment_term(sequence)
        h_electrostatic = self.electrostatic_moment_term(sequence)
        h_membrane_interaction = self.membrane_interaction_term(sequence)
        h_membrane_penalty = self.membrane_depth_penalty(sequence)
        
        total = (h_helix + h_mj + h_hydrophobic + h_electrostatic + 
                h_membrane_interaction + h_membrane_penalty)
        
        return total
    
    def analyze_sequence(self, sequence):
        """
        Análisis detallado de una secuencia
        """
        results = {
            'sequence': ''.join(sequence),
            'length': len(sequence),
            'helix_stability': self.helix_stability_term(sequence),
            'mj_energy': self.miyazawa_jernigan_term(sequence),
            'hydrophobic_moment': -self.hydrophobic_moment_term(sequence) / self.w_hydrophobic,
            'electrostatic_moment': self.calculate_electrostatic_moment(sequence),
            'membrane_interaction': self.membrane_interaction_term(sequence),
            'membrane_penalty': self.membrane_depth_penalty(sequence),
            'total_energy': self.total_hamiltonian(sequence),
            'net_charge': sum(self.aa_props.charge[aa] for aa in sequence),
            'membrane_environment': self.membrane_environment,
        }
        return results
    
    def calculate_electrostatic_moment(self, sequence):
        """Calcula solo la magnitud del momento electrostático"""
        moment_real = 0
        moment_imag = 0
        
        for i, aa in enumerate(sequence):
            angle = math.radians(self.helix_twist * i)
            charge = self.aa_props.charge[aa]
            moment_real += charge * math.cos(angle)
            moment_imag += charge * math.sin(angle)
        
        return math.sqrt(moment_real**2 + moment_imag**2)

# Ejemplo de uso
def optimize_sequence_greedy(hamiltonian, initial_sequence=None):
    """
    Optimización greedy simple (para demostración)
    En la práctica se usaría quantum annealing como en el paper
    """
    amino_acids = list(AminoAcid_Properties().helix_propensity.keys())
    
    if initial_sequence is None:
        # Secuencia inicial aleatoria
        sequence = [np.random.choice(amino_acids) for _ in range(hamiltonian.L)]
    else:
        sequence = list(initial_sequence)
    
    best_energy = hamiltonian.total_hamiltonian(sequence)
    best_sequence = sequence.copy()
    
    # Optimización greedy
    for iteration in range(1000):
        # Selecciona posición aleatoria
        pos = np.random.randint(hamiltonian.L)
        old_aa = sequence[pos]
        
        # Prueba aminoácido aleatorio
        new_aa = np.random.choice(amino_acids)
        sequence[pos] = new_aa
        
        new_energy = hamiltonian.total_hamiltonian(sequence)
        
        # Acepta si mejora (o con probabilidad si es peor - simulated annealing)
        temperature = 1.0 * np.exp(-iteration/200)
        if new_energy < best_energy or np.random.random() < np.exp(-(new_energy - best_energy)/temperature):
            if new_energy < best_energy:
                best_energy = new_energy
                best_sequence = sequence.copy()
        else:
            sequence[pos] = old_aa  # Revierte cambio
    
    return best_sequence, best_energy

# Ejemplo de aplicación
if __name__ == "__main__":
    # Diseño de péptido para membrana con carga negativa - interfacial
    hamiltonian = MembraneHelixHamiltonian(
        sequence_length=12,
        membrane_charge=-1.0,  # Membrana cargada negativamente
        membrane_depth_profile='interfacial',  # Péptido en la interfaz
        w_helix=2.0,
        w_mj=1.0,
        w_hydrophobic=3.0,
        w_electrostatic=2.0,
        w_membrane_interaction=4.0  # Peso alto para interacción con membrana
    )
    
    # Optimización
    print("Optimizando secuencia para péptido interfacial...")
    best_seq, best_energy = optimize_sequence_greedy(hamiltonian)
    
    print(f"\nMejor secuencia: {''.join(best_seq)}")
    print(f"Energía total: {best_energy:.2f}")
    
    # Análisis detallado
    analysis = hamiltonian.analyze_sequence(best_seq)
    print(f"\nAnálisis detallado:")
    print(f"Estabilidad hélice: {analysis['helix_stability']:.2f}")
    print(f"Energía MJ: {analysis['mj_energy']:.2f}")
    print(f"Momento hidrofóbico: {analysis['hydrophobic_moment']:.2f}")
    print(f"Momento electrostático: {analysis['electrostatic_moment']:.2f}")
    print(f"Interacción con membrana: {analysis['membrane_interaction']:.2f}")
    print(f"Penalización de membrana: {analysis['membrane_penalty']:.2f}")
    print(f"Carga neta: {analysis['net_charge']:.1f}")
    
    print(f"\nAmbiente de membrana por posición:")
    for i, (aa, env) in enumerate(zip(best_seq, analysis['membrane_environment'])):
        print(f"  Pos {i+1}: {aa} -> {env}")
    
    # Comparación con péptido transmembrana
    print("\n" + "="*50)
    print("Comparando con péptido transmembrana...")
    
    hamiltonian_tm = MembraneHelixHamiltonian(
        sequence_length=20,  # Más largo para atravesar membrana
        membrane_charge=0,   # Membrana neutra
        membrane_depth_profile='transmembrane',
        w_helix=2.0,
        w_mj=1.0,
        w_hydrophobic=2.0,  # Menos importante para TM
        w_electrostatic=1.0,
        w_membrane_interaction=5.0  # Muy importante para TM
    )
    
    best_seq_tm, best_energy_tm = optimize_sequence_greedy(hamiltonian_tm)
    analysis_tm = hamiltonian_tm.analyze_sequence(best_seq_tm)
    
    print(f"\nSecuencia transmembrana: {''.join(best_seq_tm)}")
    print(f"Energía total: {best_energy_tm:.2f}")
    print(f"Interacción con membrana: {analysis_tm['membrane_interaction']:.2f}")
    
    print(f"\nAmbiente de membrana por posición (TM):")
    for i, (aa, env) in enumerate(zip(best_seq_tm, analysis_tm['membrane_environment'])):
        if i % 5 == 0:  # Imprime cada 5 posiciones para brevedad
            print(f"  Pos {i+1}: {aa} -> {env}")
            
            
print("\n" + "="*50)
print("Comparando con péptido surface...")

hamiltonian_tm = MembraneHelixHamiltonian(
    sequence_length=20,  # Más largo para atravesar membrana
    membrane_charge=40,   # Membrana neutra
    membrane_depth_profile='surface',
    w_helix=20.0,
    w_mj=1.0,
    w_hydrophobic=2.0,  # Menos importante para TM
    w_electrostatic=10.0,
    w_membrane_interaction=5.0  # Muy importante para TM
)

best_seq_tm, best_energy_tm = optimize_sequence_greedy(hamiltonian_tm)
analysis_tm = hamiltonian_tm.analyze_sequence(best_seq_tm)

print(f"\nSecuencia transmembrana: {''.join(best_seq_tm)}")
print(f"Energía total: {best_energy_tm:.2f}")
print(f"Interacción con membrana: {analysis_tm['membrane_interaction']:.2f}")

print(f"\nAmbiente de membrana por posición (TM):")
for i, (aa, env) in enumerate(zip(best_seq_tm, analysis_tm['membrane_environment'])):
    if i % 5 == 0:  # Imprime cada 5 posiciones para brevedad
        print(f"  Pos {i+1}: {aa} -> {env}")
        
        
        

# Crear una figura
plt.figure(figsize=(len(best_seq_tm) * 0.5, 2))
for i, aa in enumerate(best_seq_tm):
    color = AminoAcid_Properties().aa_color_dict.get(aa, 'black')  # Por si hay caracteres no reconocidos
    plt.text(i, 0, aa, fontsize=16, ha='center', va='center', color='white',
             bbox=dict(facecolor=color, boxstyle='round,pad=0.3'))

# Eliminar ejes
plt.axis('off')
plt.xlim(-1, len(best_seq_tm))
plt.ylim(-1, 1)

plt.title("Secuencia coloreada por tipo de aminoácido", fontsize=14)
plt.tight_layout()
plt.show()