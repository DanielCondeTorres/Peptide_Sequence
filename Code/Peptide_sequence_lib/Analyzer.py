from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from Peptide_sequence_lib.AminoAcid_Properties import AminoAcid_Properties
from Peptide_sequence_lib.Peptide_Target_Candidate import PeptideCandidate
from collections import defaultdict

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

from Peptide_sequence_lib.AminoAcid_Properties import AminoAcid_Properties
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
