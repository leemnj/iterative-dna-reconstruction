"""
Metric utilities for sequence analysis and embeddings.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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


def prepare_sequence_metrics(df):
    """
    Add sequence length and Shannon entropy columns to a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with a "Sequence" column

    Returns:
        pd.DataFrame: Copy with "SequenceLength" and "ShannonEntropy" columns
    """
    df = df.copy()
    if "Sequence" in df.columns:
        df["SequenceLength"] = df["Sequence"].astype(str).str.len()
        df["ShannonEntropy"] = df["Sequence"].apply(calculate_shannon_entropy)
    return df


def compute_gc_content(sequence):
    """
    Compute GC content ratio for a DNA sequence.

    Args:
        sequence (str): DNA sequence

    Returns:
        float: GC content ratio (0-1)
    """
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / max(len(sequence), 1)


def infer_gene_lengths(sequences_dict, prefer_strategy="greedy"):
    """
    Infer gene lengths from the first iteration sequence.

    Args:
        sequences_dict (dict): {gene_id: {strategy: [sequences]}}
        prefer_strategy (str): Strategy key to prefer when available

    Returns:
        dict: {gene_id: length}
    """
    lengths = {}
    for gene_id, strategies in sequences_dict.items():
        seq = None
        if prefer_strategy and prefer_strategy in strategies:
            seqs = strategies.get(prefer_strategy, [])
            if seqs:
                seq = seqs[0]
        if seq is None:
            for seqs in strategies.values():
                if seqs:
                    seq = seqs[0]
                    break
        if seq is not None:
            lengths[gene_id] = len(seq)
    return lengths


def compute_kmer_frequencies(sequence, k=6):
    """
    Compute k-mer frequency distribution for a sequence.

    Args:
        sequence (str): DNA sequence
        k (int): k-mer length

    Returns:
        dict: {kmer: count}
    """
    if not sequence or k <= 0 or len(sequence) < k:
        return {}
    sequence = sequence.upper()
    counts = Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))
    return dict(counts)


def aggregate_kmer_frequencies(sequences, k=6):
    """
    Aggregate k-mer frequencies across multiple sequences.

    Args:
        sequences (list): List of sequences
        k (int): k-mer length

    Returns:
        dict: {kmer: mean_count}
    """
    if not sequences:
        return {}
    aggregate = Counter()
    for seq in sequences:
        aggregate.update(compute_kmer_frequencies(seq, k=k))
    return dict(aggregate)


def compute_vendi_score(embeddings, eps=1e-12):
    """
    Compute Vendi score from a set of embeddings using cosine similarity.

    Args:
        embeddings (list or np.ndarray): List/array of embedding vectors
        eps (float): Small value for numerical stability

    Returns:
        float: Vendi score
    """
    if embeddings is None:
        return 0.0

    matrix = np.asarray(embeddings)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.shape[0] == 0:
        return 0.0
    if matrix.shape[0] == 1:
        return 1.0

    matrix = matrix.reshape(matrix.shape[0], -1)
    finite_mask = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[finite_mask]
    if matrix.shape[0] == 0:
        return 0.0
    if matrix.shape[0] == 1:
        return 1.0

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    matrix = matrix / norms
    kernel = matrix @ matrix.T

    trace = float(np.trace(kernel))
    if trace <= eps:
        return 0.0
    eigvals = np.linalg.eigvalsh(kernel)
    eigvals = np.clip(eigvals, 0.0, None)
    probs = eigvals / trace
    probs = probs[probs > eps]
    if probs.size == 0:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))
