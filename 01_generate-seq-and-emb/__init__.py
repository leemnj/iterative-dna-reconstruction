"""
DNA Sequence Evolution Pipeline - Modular Version

A modular pipeline for simulating DNA sequence evolution using 
foundation models (DNABERT-2, Nucleotide Transformer) and analyzing
semantic representation changes.

Modules:
    - preparation: Environment setup and model loading
    - sequence_generation: Gene collection and sequence evolution
    - visualization: Results visualization and analysis
"""

__version__ = "1.0.0"
__author__ = "DNA Model Research Team"

from preparation import get_device, load_models, SequenceEvolver
from sequence_generation import (
    fetch_gene_sequences,
    sort_genes_by_length,
    generate_and_embed_sequences,
    save_sequences_compressed,
    load_sequences_compressed,
    DEFAULT_DECODING_STRATEGIES,
    DEFAULT_GENE_UIDS,
    DEFAULT_GENES_TO_SEARCH
)
from visualization import (
    ResultsLoader,
    cosine_series_from_embeddings,
    calculate_shannon_entropy,
    plot_semantic_similarity_overview,
    plot_gene_pair_comparison
)

__all__ = [
    # preparation
    'get_device',
    'load_models',
    'SequenceEvolver',
    # sequence_generation
    'fetch_gene_sequences',
    'sort_genes_by_length',
    'generate_and_embed_sequences',
    'save_sequences_compressed',
    'load_sequences_compressed',
    'DEFAULT_DECODING_STRATEGIES',
    'DEFAULT_GENE_UIDS',
    'DEFAULT_GENES_TO_SEARCH',
    # visualization
    'ResultsLoader',
    'cosine_series_from_embeddings',
    'calculate_shannon_entropy',
    'plot_semantic_similarity_overview',
    'plot_gene_pair_comparison',
]
