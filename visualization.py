"""
Visualization module for DNA sequence evolution results.
Handles plotting and analysis of semantic similarity, entropy, and sequence length.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
from sequence_generation import load_sequences_compressed, load_from_parts


def cosine_series_from_embeddings(embeddings):
    """
    Compute cosine similarity series relative to the first embedding.
    
    Args:
        embeddings (list): List of embedding arrays
        
    Returns:
        list: List of cosine similarity values
    """
    if not embeddings:
        return []
    
    base = embeddings[0].reshape(1, -1)
    sims = []
    for emb in embeddings:
        vec = emb.reshape(1, -1)
        sims.append(float(cosine_similarity(base, vec)[0, 0]))
    return sims


def calculate_shannon_entropy(sequence):
    """
    Calculate Shannon entropy for a DNA sequence.
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        float: Shannon entropy
    """
    if not sequence:
        return 0.0
    
    sequence_length = len(sequence)
    nucleotide_counts = Counter(sequence.upper())
    entropy = 0.0
    
    for count in nucleotide_counts.values():
        p = count / sequence_length
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


class ResultsLoader:
    """Handle loading of generated sequences and embeddings."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the results loader.
        
        Args:
            output_dir (Path or str): Output directory containing results
        """
        if output_dir is None:
            output_dir = Path('output')
        else:
            output_dir = Path(output_dir)
        
        self.output_dir = output_dir
        self.embeddings = {}
        self.sequences = {}
    
    def load_embeddings(self):
        """Load embeddings from disk."""
        embedding_file = self.output_dir / 'gene_embeddings.pkl.gz'
        
        if embedding_file.exists():
            print(f"Loading embeddings from {embedding_file}...")
            self.embeddings = load_sequences_compressed(embedding_file)
        
        # Try loading from parts if not found
        if not self.embeddings:
            parts_dir = self.output_dir / 'parts' / 'embeddings'
            if parts_dir.exists():
                print(f"Loading embeddings from parts directory...")
                self.embeddings = load_from_parts(parts_dir)
        
        if self.embeddings:
            print(f"✅ Loaded embeddings for genes: {list(self.embeddings.keys())}")
        else:
            print("⚠️ No embeddings found")
        
        return self.embeddings
    
    def load_sequences(self):
        """Load sequences from disk."""
        sequence_file = self.output_dir / 'generated_sequences.json.gz'
        
        if sequence_file.exists():
            print(f"Loading sequences from {sequence_file}...")
            self.sequences = load_sequences_compressed(sequence_file)
        
        # Try loading from parts if not found
        if not self.sequences:
            parts_dir = self.output_dir / 'parts' / 'sequences'
            if parts_dir.exists():
                print(f"Loading sequences from parts directory...")
                self.sequences = load_from_parts(parts_dir)
        
        if self.sequences:
            print(f"✅ Loaded sequences for genes: {list(self.sequences.keys())}")
        else:
            print("⚠️ No sequences found")
        
        return self.sequences


def plot_semantic_similarity_overview(embeddings, model_label="DNABERT-2"):
    """
    Plot semantic similarity overview for all genes and strategies.
    
    Args:
        embeddings (dict): Embeddings dictionary
        model_label (str): Model label to visualize
    """
    # Determine available strategy keys
    preferred_strategies = ["greedy", "sampling_t0.5", "sampling_t1.0", "sampling_t1.5"]
    available_strategies = []
    
    for strategy_key in preferred_strategies:
        if any(
            model_label in embeddings.get(g, {}) and 
            strategy_key in embeddings[g][model_label]
            for g in embeddings.keys()
        ):
            available_strategies.append(strategy_key)
    
    if not available_strategies:
        print(f"⚠️ No strategies found for {model_label}")
        return
    
    # Build similarity series
    similarities = {}
    for gene_name, model_data in embeddings.items():
        if model_label not in model_data:
            continue
        
        similarities[gene_name] = {}
        for strategy_key in available_strategies:
            embs = model_data[model_label].get(strategy_key, [])
            similarities[gene_name][strategy_key] = cosine_series_from_embeddings(embs)
    
    # Plot 2x2 grid
    num_rows, num_cols = 2, 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 10), squeeze=False)
    fig.suptitle(f"{model_label}: Semantic Similarity Overview", fontsize=16, y=1.02)
    
    gene_names = list(similarities.keys())
    
    for idx, strategy_key in enumerate(available_strategies):
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        ax = axes[row_idx, col_idx]
        
        for gene_name in gene_names:
            series = similarities[gene_name].get(strategy_key, [])
            if not series:
                continue
            
            x_axis = list(range(len(series)))
            ax.plot(x_axis, series, marker='o', linestyle='-', markersize=3, label=gene_name)
        
        ax.set_title(f"Strategy: {strategy_key}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1)
        
        if gene_names:
            ax.legend(loc="lower left", fontsize="small")
    
    # Hide unused axes
    for j in range(len(available_strategies), num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
    print(f"✅ {model_label} overview plot generated.")


def plot_gene_pair_comparison(embeddings, sequences, gene_a, gene_b, title, 
                               model_label="DNABERT-2"):
    """
    Plot detailed comparison between two genes with similarity, entropy, and length.
    
    Args:
        embeddings (dict): Embeddings dictionary
        sequences (dict): Sequences dictionary
        gene_a (str): First gene name
        gene_b (str): Second gene name
        title (str): Plot title
        model_label (str): Model label to visualize
    """
    # Determine available strategies
    preferred_strategies = ["greedy", "sampling_t0.5", "sampling_t1.0", "sampling_t1.5"]
    available_strategies = []
    
    for strategy_key in preferred_strategies:
        if (model_label in embeddings.get(gene_a, {}) and 
            strategy_key in embeddings[gene_a][model_label]):
            available_strategies.append(strategy_key)
    
    if not available_strategies:
        print(f"⚠️ No strategies found for {model_label}")
        return
    
    num_rows = len(available_strategies)
    fig, axes = plt.subplots(num_rows, 2, figsize=(18, 4 * num_rows), squeeze=False)
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Compute entropy range
    all_entropies = []
    for strategy_key in available_strategies:
        for gene_name in (gene_a, gene_b):
            seqs = sequences.get(gene_name, {}).get(model_label, {}).get(strategy_key, [])
            all_entropies.extend(calculate_shannon_entropy(seq) for seq in seqs)
    
    entropy_min = min(all_entropies) if all_entropies else 0.0
    entropy_max = max(all_entropies) if all_entropies else 1.0
    
    # Gene colors
    gene_colors = {
        'STAT3': '#1b9e77',
        'NORAD': '#d95f02',
        'GAPDH': '#7570b3',
        'GAPDHP1': '#e7298a',
        'TP53': '#e41a1c',
        'H4C1': '#377eb8',
    }
    
    for row_idx, strategy_key in enumerate(available_strategies):
        ax_sim = axes[row_idx, 0]
        ax_ent = ax_sim.twinx()
        ax_len = axes[row_idx, 1]
        
        for gene_name in (gene_a, gene_b):
            color = gene_colors.get(gene_name, '#999999')
            
            # Get data
            embs = embeddings.get(gene_name, {}).get(model_label, {}).get(strategy_key, [])
            seqs = sequences.get(gene_name, {}).get(model_label, {}).get(strategy_key, [])
            
            # Compute metrics
            sims = cosine_series_from_embeddings(embs)
            entropies = [calculate_shannon_entropy(seq) for seq in seqs]
            lengths = [len(seq) for seq in seqs]
            
            # Plot
            x_sim = list(range(len(sims)))
            x_seq = list(range(len(seqs)))
            
            if sims:
                ax_sim.plot(
                    x_sim, sims, marker='o', linestyle='-', markersize=3,
                    color=color, label=f"{gene_name} Similarity"
                )
            if entropies:
                ax_ent.plot(
                    x_seq, entropies, marker='x', linestyle='--', markersize=3,
                    color=color, alpha=0.7, label=f"{gene_name} Entropy"
                )
            if lengths:
                ax_len.plot(
                    x_seq, lengths, marker='^', linestyle='-', markersize=3,
                    color=color, alpha=0.7, label=f"{gene_name} Length"
                )
        
        # Configure left plot (similarity + entropy)
        ax_sim.set_title(f"{strategy_key} | Similarity + Entropy")
        ax_sim.set_xlabel('Iteration')
        ax_sim.set_ylabel('Cosine Similarity', color='black')
        ax_sim.grid(True, linestyle='--', alpha=0.6)
        ax_sim.set_ylim(0, 1)
        ax_ent.set_ylabel('Shannon Entropy', color='black')
        ax_ent.set_ylim(entropy_min, entropy_max)
        
        # Configure right plot (length)
        ax_len.set_title(f"{strategy_key} | Sequence Length")
        ax_len.set_xlabel('Iteration')
        ax_len.set_ylabel('Sequence Length')
        ax_len.grid(True, linestyle='--', alpha=0.6)
        
        # Combine legends
        lines1, labels1 = ax_sim.get_legend_handles_labels()
        lines2, labels2 = ax_ent.get_legend_handles_labels()
        ax_sim.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize='small')
        ax_len.legend(loc='lower left', fontsize='small')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
    print(f"✅ Gene pair comparison plot generated.")


if __name__ == "__main__":
    # Example usage
    loader = ResultsLoader(output_dir=Path('output'))
    embeddings = loader.load_embeddings()
    sequences = loader.load_sequences()
    
    if embeddings:
        plot_semantic_similarity_overview(embeddings, model_label="DNABERT-2")
    
    if embeddings and sequences:
        plot_gene_pair_comparison(
            embeddings, sequences, 'STAT3', 'NORAD',
            'DNABERT-2: Coding (STAT3) vs Non-coding (NORAD)'
        )
