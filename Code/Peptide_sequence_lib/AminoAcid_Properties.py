class AminoAcid_Properties:
    """Amino acid properties for peptide design"""
    
    def __init__(self):
        # Standard amino acids
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
        # Hydrophobicity (Kyte-Doolittle scale)
        self.hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
    
        # Charge at physiological pH
        self.charge = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
            'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
            'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
            'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
    
        # Volume (Å³)
        self.volume = {
            'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
            'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
            'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
            'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
        }
    
        # Secondary structure propensity
        self.helix_propensity = {
            'A': 0.0,  'L': 0.21, 'R': 0.21, 'M': 0.24, 'K': 0.26,
            'Q': 0.39, 'E': 0.40, 'I': 0.41, 'W': 0.49, 'S': 0.50,
            'Y': 0.53, 'F': 0.54, 'V': 0.61, 'H': 0.61, 'N': 0.65,
            'T': 0.66, 'C': 0.68, 'D': 0.69, 'G': 1.0
        }
 
        # Preferencias de interacción con cabezas polares de lípidos (kcal/mol)
        # Valores más negativos = más favorable
        self.polar_head_affinity = {
            'R': -3.0, 'K': -2.8, 'H': -2.0,  # Básicos: favorables con cabezas aniónicas
            'D': -1.5, 'E': -1.8,             # Ácidos: pueden formar puentes H
            'N': -2.2, 'Q': -2.5,             # Amidas: buenos donadores/aceptores H
            'S': -1.8, 'T': -1.9, 'Y': -2.1,  # Con OH: pueden formar puentes H
            'W': -1.2,                        # Indol puede interactuar con fosfatos
            'C': -0.5,                        # Puede formar enlaces débiles
            'A': 0.0, 'G': 0.0,               # Neutros, sin preferencia específica
            'L': 1.0, 'I': 1.2, 'V': 0.8,     # Hidrofóbicos: desfavorables
            'F': 0.5, 'M': 0.2                # Aromático/azufre: ligeramente desfavorables
        }

        # Preferencias de interacción con colas hidrofóbicas de lípidos (kcal/mol)
        # Valores más negativos = más favorable para inserción en región hidrofóbica
        self.hydrophobic_tail_affinity = {
            'L': -2.8, 'I': -2.6, 'V': -2.1,  # Muy hidrofóbicos: muy favorables
            'F': -2.5, 'W': -2.9, 'Y': -1.8,  # Aromáticos: favorables
            'M': -2.0, 'A': -1.8,             # Alifáticos: favorables
            'C': -1.5,                        # Azufre: moderadamente favorable
            'G': -0.5,                        # Pequeño: algo favorable
            'S': 0.8, 'T': 1.0,               # Polares pequeños: desfavorables
            'N': 2.0, 'Q': 2.2,               # Amidas: muy desfavorables
            'H': 2.5,                         # Histidina: desfavorable
            'R': 4.0, 'K': 3.8,               # Básicos: muy desfavorables
            'D': 3.5, 'E': 3.2                # Ácidos: muy desfavorables
        }
        
        
        # Matriz de Miyazawa-Jernigan simplificada (kcal/mol)
        # Solo algunos valores representativos - se necesitaría la matriz completa
        self.mj_matrix_sample = {
            ('A', 'A'): -2.72,
            ('A', 'C'): -3.57,
            ('A', 'D'): -1.70,
            ('A', 'E'): -1.51,
            ('A', 'F'): -4.81,
            ('A', 'G'): -2.72,
            ('A', 'H'): -2.41,
            ('A', 'I'): -4.58,
            ('A', 'K'): -1.31,
            ('A', 'L'): -4.91,
            ('A', 'M'): -3.94,
            ('A', 'N'): -1.84,
            ('A', 'P'): -2.03,
            ('A', 'Q'): -1.89,
            ('A', 'R'): -1.83,
            ('A', 'S'): -2.01,
            ('A', 'T'): -2.32,
            ('A', 'V'): -4.04,
            ('A', 'W'): -3.82,
            ('A', 'Y'): -3.36,
            ('C', 'C'): -5.44,
            ('C', 'D'): -2.41,
            ('C', 'E'): -2.27,
            ('C', 'F'): -5.80,
            ('C', 'G'): -3.16,
            ('C', 'H'): -3.60,
            ('C', 'I'): -5.50,
            ('C', 'K'): -1.95,
            ('C', 'L'): -5.83,
            ('C', 'M'): -4.99,
            ('C', 'N'): -2.59,
            ('C', 'P'): -3.07,
            ('C', 'Q'): -2.85,
            ('C', 'R'): -2.57,
            ('C', 'S'): -2.86,
            ('C', 'T'): -3.11,
            ('C', 'V'): -4.96,
            ('C', 'W'): -4.95,
            ('C', 'Y'): -4.16,
            ('D', 'D'): -1.21,
            ('D', 'E'): -0.91,
            ('D', 'F'): -3.48,
            ('D', 'G'): -1.70,
            ('D', 'H'): -2.32,
            ('D', 'I'): -3.17,
            ('D', 'K'): -1.68,
            ('D', 'L'): -3.40,
            ('D', 'M'): -2.57,
            ('D', 'N'): -1.63,
            ('D', 'P'): -1.33,
            ('D', 'Q'): -1.46,
            ('D', 'R'): -2.29,
            ('D', 'S'): -1.88,
            ('D', 'T'): -1.80,
            ('D', 'V'): -2.48,
            ('D', 'W'): -2.84,
            ('D', 'Y'): -2.76,
            ('E', 'E'): -0.91,
            ('E', 'F'): -3.56,
            ('E', 'G'): -1.51,
            ('E', 'H'): -2.15,
            ('E', 'I'): -3.27,
            ('E', 'K'): -1.26,
            ('E', 'L'): -3.59,
            ('E', 'M'): -2.89,
            ('E', 'N'): -1.42,
            ('E', 'P'): -1.26,
            ('E', 'Q'): -1.02,
            ('E', 'R'): -2.27,
            ('E', 'S'): -1.74,
            ('E', 'T'): -1.74,
            ('E', 'V'): -2.67,
            ('E', 'W'): -2.99,
            ('E', 'Y'): -2.79,
            ('F', 'F'): -7.26,
            ('F', 'G'): -4.13,
            ('F', 'H'): -4.77,
            ('F', 'I'): -6.84,
            ('F', 'K'): -3.36,
            ('F', 'L'): -7.28,
            ('F', 'M'): -6.56,
            ('F', 'N'): -3.75,
            ('F', 'P'): -4.25,
            ('F', 'Q'): -4.10,
            ('F', 'R'): -3.98,
            ('F', 'S'): -4.02,
            ('F', 'T'): -4.28,
            ('F', 'V'): -6.29,
            ('F', 'W'): -6.16,
            ('F', 'Y'): -5.66,
            ('G', 'G'): -2.24,
            ('G', 'H'): -2.15,
            ('G', 'I'): -3.78,
            ('G', 'K'): -1.31,
            ('G', 'L'): -4.16,
            ('G', 'M'): -3.39,
            ('G', 'N'): -1.84,
            ('G', 'P'): -2.03,
            ('G', 'Q'): -1.89,
            ('G', 'R'): -1.83,
            ('G', 'S'): -2.01,
            ('G', 'T'): -2.32,
            ('G', 'V'): -3.38,
            ('G', 'W'): -3.42,
            ('G', 'Y'): -3.01,
            ('H', 'H'): -3.05,
            ('H', 'I'): -4.14,
            ('H', 'K'): -1.35,
            ('H', 'L'): -4.54,
            ('H', 'M'): -3.98,
            ('H', 'N'): -2.08,
            ('H', 'P'): -2.25,
            ('H', 'Q'): -1.98,
            ('H', 'R'): -2.16,
            ('H', 'S'): -2.11,
            ('H', 'T'): -2.42,
            ('H', 'V'): -3.58,
            ('H', 'W'): -3.98,
            ('H', 'Y'): -3.52,
            ('I', 'I'): -6.54,
            ('I', 'K'): -3.01,
            ('I', 'L'): -7.04,
            ('I', 'M'): -6.02,
            ('I', 'N'): -3.24,
            ('I', 'P'): -3.76,
            ('I', 'Q'): -3.67,
            ('I', 'R'): -3.63,
            ('I', 'S'): -3.52,
            ('I', 'T'): -4.03,
            ('I', 'V'): -6.05,
            ('I', 'W'): -5.78,
            ('I', 'Y'): -5.25,
            ('K', 'K'): -0.12,
            ('K', 'L'): -3.37,
            ('K', 'M'): -2.48,
            ('K', 'N'): -1.21,
            ('K', 'P'): -1.68,
            ('K', 'Q'): -1.29,
            ('K', 'R'): -0.59,
            ('K', 'S'): -1.15,
            ('K', 'T'): -1.31,
            ('K', 'V'): -2.49,
            ('K', 'W'): -2.69,
            ('K', 'Y'): -2.60,
            ('L', 'L'): -7.37,
            ('L', 'M'): -6.41,
            ('L', 'N'): -3.74,
            ('L', 'P'): -4.20,
            ('L', 'Q'): -4.04,
            ('L', 'R'): -4.03,
            ('L', 'S'): -3.92,
            ('L', 'T'): -4.34,
            ('L', 'V'): -6.48,
            ('L', 'W'): -6.14,
            ('L', 'Y'): -5.67,
            ('M', 'M'): -5.46,
            ('M', 'N'): -2.95,
            ('M', 'P'): -3.45,
            ('M', 'Q'): -3.30,
            ('M', 'R'): -3.12,
            ('M', 'S'): -3.03,
            ('M', 'T'): -3.51,
            ('M', 'V'): -5.32,
            ('M', 'W'): -5.55,
            ('M', 'Y'): -4.91,
            ('N', 'N'): -1.68,
            ('N', 'P'): -1.53,
            ('N', 'Q'): -1.71,
            ('N', 'R'): -1.64,
            ('N', 'S'): -1.58,
            ('N', 'T'): -1.88,
            ('N', 'V'): -2.83,
            ('N', 'W'): -3.07,
            ('N', 'Y'): -2.76,
            ('P', 'P'): -1.75,
            ('P', 'Q'): -1.73,
            ('P', 'R'): -1.70,
            ('P', 'S'): -1.57,
            ('P', 'T'): -1.90,
            ('P', 'V'): -3.32,
            ('P', 'W'): -3.73,
            ('P', 'Y'): -3.19,
            ('Q', 'Q'): -1.54,
            ('Q', 'R'): -1.80,
            ('Q', 'S'): -1.49,
            ('Q', 'T'): -1.90,
            ('Q', 'V'): -3.07,
            ('Q', 'W'): -3.11,
            ('Q', 'Y'): -2.97,
            ('R', 'R'): -1.55,
            ('R', 'S'): -1.62,
            ('R', 'T'): -1.90,
            ('R', 'V'): -3.07,
            ('R', 'W'): -3.41,
            ('R', 'Y'): -3.16,
            ('S', 'S'): -1.67,
            ('S', 'T'): -1.96,
            ('S', 'V'): -3.05,
            ('S', 'W'): -2.99,
            ('S', 'Y'): -2.78,
            ('T', 'T'): -2.12,
            ('T', 'V'): -3.46,
            ('T', 'W'): -3.22,
            ('T', 'Y'): -3.01,
            ('V', 'V'): -5.52,
            ('V', 'W'): -5.18,
            ('V', 'Y'): -4.62,
            ('W', 'W'): -5.06,
            ('W', 'Y'): -4.66,
            ('Y', 'Y'): -4.17
        }
        
        self.aa_color_dict = {
            # Carga negativa (rojo)
            'D': 'red',
            'E': 'red',
            
            # Carga positiva (azul)
            'K': 'blue',
            'R': 'blue',
            'H': 'blue',
            
            # Apolares (gris)
            'A': 'gray',
            'V': 'gray',
            'L': 'gray',
            'I': 'gray',
            'M': 'gray',
            'F': 'gray',
            'W': 'gray',
            'P': 'gray',
            'G': 'gray',
            
            # Polares sin carga (amarillo)
            'S': 'yellow',
            'T': 'yellow',
            'Y': 'yellow',
            'N': 'yellow',
            'Q': 'yellow',
            'C': 'yellow'
        }










