"""
Visualization module for DNA sequence evolution results.
Handles plotting and analysis of semantic similarity, entropy, and sequence length.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
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


def set_publication_style(font_family="Times New Roman", dpi=300):
    """
    Set Matplotlib/Seaborn style for publication-quality figures.
    
    Args:
        font_family (str): Preferred font family ("Times New Roman" or "Arial")
        dpi (int): Figure DPI for on-screen rendering and saving
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


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


def plot_similarity_over_iterations(
    df,
    model_label=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot similarity trends over iterations, faceted by gene.
    
    Args:
        df (pd.DataFrame): Long-form dataframe with Iteration, Gene, Strategy, Similarity
        model_label (str or None): Optional title suffix for model label
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    data = df.copy()
    data["Iteration"] = pd.to_numeric(data["Iteration"], errors="coerce")
    data = data.dropna(subset=["Iteration", "Similarity", "Gene", "Strategy"])
    
    grid = sns.relplot(
        data=data,
        x="Iteration",
        y="Similarity",
        hue="Strategy",
        col="Gene",
        col_wrap=3,
        kind="line",
        marker="o",
        linewidth=1.4,
        height=3.2,
        aspect=1.2,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Iteration", "Cosine Similarity")
    grid.set(ylim=(0, 1))
    for ax in grid.axes.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    title = "Semantic Similarity Over Iterations"
    if model_label:
        title = f"{title} ({model_label})"
    grid.fig.suptitle(title, y=1.03)
    
    if save_path:
        grid.fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return grid


def plot_similarity_distribution(
    df,
    model_label=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot similarity distribution by strategy and gene.
    
    Args:
        df (pd.DataFrame): Long-form dataframe with Similarity, Strategy, Gene
        model_label (str or None): Optional title suffix for model label
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    data = df.dropna(subset=["Similarity", "Strategy", "Gene"]).copy()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.boxplot(
        data=data,
        x="Strategy",
        y="Similarity",
        hue="Gene",
        ax=ax,
        fliersize=2,
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)
    ax.legend(title="Gene", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    title = "Similarity Distribution by Strategy"
    if model_label:
        title = f"{title} ({model_label})"
    ax.set_title(title)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_entropy_and_length(
    df,
    model_label=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot sequence entropy and length over iterations.
    
    Args:
        df (pd.DataFrame): Long-form dataframe with Sequence, Iteration, Strategy, Gene
        model_label (str or None): Optional title suffix for model label
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    data = prepare_sequence_metrics(df)
    data["Iteration"] = pd.to_numeric(data["Iteration"], errors="coerce")
    data = data.dropna(subset=["Iteration", "Gene", "Strategy"])
    
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    sns.lineplot(
        data=data,
        x="Iteration",
        y="ShannonEntropy",
        hue="Strategy",
        style="Gene",
        markers=True,
        dashes=False,
        linewidth=1.2,
        ax=axes[0],
    )
    axes[0].set_title("Shannon Entropy")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    sns.lineplot(
        data=data,
        x="Iteration",
        y="SequenceLength",
        hue="Strategy",
        style="Gene",
        markers=True,
        dashes=False,
        linewidth=1.2,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Sequence Length")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Length (bp)")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    title = "Sequence Complexity Over Iterations"
    if model_label:
        title = f"{title} ({model_label})"
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def build_similarity_records(embeddings_dict, strategies=None):
    """
    Build long-form similarity records from embeddings.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        strategies (list or None): Optional strategy filter
    
    Returns:
        pd.DataFrame: Columns [Gene, Strategy, Iteration, Similarity]
    """
    records = []
    for gene_id, strategies_map in embeddings_dict.items():
        for strategy_key, embs in strategies_map.items():
            if strategies and strategy_key not in strategies:
                continue
            sims = cosine_series_from_embeddings(embs)
            for idx, value in enumerate(sims):
                records.append({
                    "Gene": gene_id,
                    "Strategy": strategy_key,
                    "Iteration": idx,
                    "Similarity": value,
                })
    return pd.DataFrame(records)


def plot_strategy_collapse_with_ci(
    embeddings_dict,
    model_label=None,
    strategies=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot mean similarity with std shading per decoding strategy.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        model_label (str or None): Optional title suffix
        strategies (list or None): Optional strategy filter/order
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    df = build_similarity_records(embeddings_dict, strategies=strategies)
    if df.empty:
        print("⚠️ No similarity data available for strategy collapse plot.")
        return None
    
    summary = (
        df.groupby(["Strategy", "Iteration"])["Similarity"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    
    strategy_order = strategies or list(summary["Strategy"].unique())
    palette = sns.color_palette("tab10", n_colors=len(strategy_order))
    
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for idx, strategy_key in enumerate(strategy_order):
        subset = summary[summary["Strategy"] == strategy_key]
        if subset.empty:
            continue
        color = palette[idx]
        ax.plot(
            subset["Iteration"],
            subset["mean"],
            label=strategy_key,
            color=color,
            linewidth=2.0,
        )
        ax.fill_between(
            subset["Iteration"],
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=color,
            alpha=0.2,
        )
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    title = "Model Collapse by Decoding Strategy"
    if model_label:
        title = f"{title} ({model_label})"
    ax.set_title(title)
    ax.legend(title="Strategy")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_similarity_overview_all_models(
    all_embeddings,
    model_labels=None,
    strategy_order=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot semantic similarity overview for all models and strategies.
    
    Args:
        all_embeddings (dict): {model_label: {gene_id: {strategy: [embeddings]}}}
        model_labels (list or None): Optional model order
        strategy_order (list or None): Optional strategy order
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    if not all_embeddings:
        print("⚠️ No embeddings available for overview plot.")
        return None
    
    model_labels = model_labels or list(all_embeddings.keys())
    if not model_labels:
        print("⚠️ No models available for overview plot.")
        return None
    
    # Determine strategies from first available model
    if strategy_order is None:
        for model_label in model_labels:
            model_data = all_embeddings.get(model_label, {})
            if model_data:
                first_gene = next(iter(model_data.values()), {})
                strategy_order = list(first_gene.keys())
                break
    strategy_order = strategy_order or []
    
    num_rows = len(model_labels)
    num_cols = len(strategy_order)
    if num_rows == 0 or num_cols == 0:
        print("⚠️ No strategies available for overview plot.")
        return None
    
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(4.2 * num_cols, 3.1 * num_rows), squeeze=False
    )
    legend_handles = []
    legend_labels = []
    
    for row_idx, model_label in enumerate(model_labels):
        model_data = all_embeddings.get(model_label, {})
        gene_names = list(model_data.keys())
        for col_idx, strategy_key in enumerate(strategy_order):
            ax = axes[row_idx, col_idx]
            for gene_name in gene_names:
                embeddings = model_data.get(gene_name, {}).get(strategy_key, [])
                sims = cosine_series_from_embeddings(embeddings)
                if not sims:
                    continue
                x_axis = list(range(len(sims)))
                line, = ax.plot(
                    x_axis,
                    sims,
                    marker="o",
                    linestyle="-",
                    markersize=2.6,
                    linewidth=1.1,
                    label=gene_name,
                )
                if gene_name not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(gene_name)
            if row_idx == 0:
                ax.set_title(f"{strategy_key}")
            if col_idx == 0:
                ax.set_ylabel(f"{model_label}\nCosine Similarity")
            else:
                ax.set_ylabel("Cosine Similarity")
            ax.set_xlabel("Iteration")
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", alpha=0.4)
    
    fig.suptitle("Semantic Similarity Overview", y=1.02)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Gene",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


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


def plot_kmer_distribution_pair(
    sequences_dict,
    gene_a,
    gene_b,
    k=6,
    iteration_index=50,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot top-k k-mer distributions for real vs pseudogene comparison.
    
    Args:
        sequences_dict (dict): {gene_id: {strategy: [sequences]}}
        gene_a (str): Real gene name
        gene_b (str): Pseudogene name
        k (int): k-mer length
        iteration_index (int): Iteration index to compare
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    
    def collect_sequences(gene_id, index):
        gene_strategies = sequences_dict.get(gene_id, {})
        sequences = []
        for seqs in gene_strategies.values():
            if not seqs:
                continue
            if index < 0:
                idx = len(seqs) + index
            else:
                idx = index
            if idx < 0 or idx >= len(seqs):
                continue
            sequences.append(seqs[idx])
        return sequences
    
    base_seqs = collect_sequences(gene_a, 0)
    gen_seqs = collect_sequences(gene_b, iteration_index)
    if not base_seqs or not gen_seqs:
        print("⚠️ Insufficient sequences for k-mer distribution plot.")
        return None
    
    base_kmers = aggregate_kmer_frequencies(base_seqs, k=k)
    gen_kmers = aggregate_kmer_frequencies(gen_seqs, k=k)
    if not base_kmers:
        print("⚠️ No k-mer data available for baseline.")
        return None
    
    top_kmers = [kmer for kmer, _ in Counter(base_kmers).most_common(10)]
    base_counts = [base_kmers.get(kmer, 0) for kmer in top_kmers]
    gen_counts = [gen_kmers.get(kmer, 0) for kmer in top_kmers]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    sns.barplot(x=top_kmers, y=base_counts, ax=axes[0], color="#4C78A8")
    axes[0].set_title(f"{gene_a} (Original)")
    axes[0].set_xlabel(f"Top {len(top_kmers)} {k}-mers")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis="x", rotation=45)
    
    sns.barplot(x=top_kmers, y=gen_counts, ax=axes[1], color="#F58518")
    axes[1].set_title(f"{gene_b} (Iter {iteration_index})")
    axes[1].set_xlabel(f"Top {len(top_kmers)} {k}-mers")
    axes[1].tick_params(axis="x", rotation=45)
    
    fig.suptitle("Top k-mer Distributions (Real vs Pseudogene)", y=1.03)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def compute_collapse_severity(
    embeddings_dict,
    metric="final_similarity",
    threshold=0.8,
):
    """
    Compute collapse severity per gene and strategy.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        metric (str): "final_similarity" or "iteration_below_threshold"
        threshold (float): Threshold for collapse iteration metric
    
    Returns:
        pd.DataFrame: Columns [Gene, Strategy, CollapseSeverity]
    """
    records = []
    for gene_id, strategies_map in embeddings_dict.items():
        for strategy_key, embs in strategies_map.items():
            sims = cosine_series_from_embeddings(embs)
            if not sims:
                continue
            if metric == "final_similarity":
                value = sims[-1]
            elif metric == "iteration_below_threshold":
                idx = next((i for i, v in enumerate(sims) if v < threshold), None)
                value = len(sims) if idx is None else idx
            else:
                raise ValueError(f"Unknown collapse metric: {metric}")
            records.append({
                "Gene": gene_id,
                "Strategy": strategy_key,
                "CollapseSeverity": value,
            })
    return pd.DataFrame(records)


def plot_collapse_vs_gene_property(
    embeddings_dict,
    sequences_dict,
    gene_metadata=None,
    property_key="length",
    metric="final_similarity",
    threshold=0.8,
    prefer_strategy="greedy",
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot collapse severity versus a gene property with regression line.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        sequences_dict (dict): {gene_id: {strategy: [sequences]}}
        gene_metadata (dict or None): {gene_id: {"length": int, "exon_count": int}}
        property_key (str): "length" or "exon_count"
        metric (str): Collapse severity metric
        threshold (float): Threshold for collapse iteration metric
        prefer_strategy (str): Preferred strategy for length inference
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    gene_metadata = gene_metadata or {}
    
    if property_key == "length":
        property_map = infer_gene_lengths(sequences_dict, prefer_strategy=prefer_strategy)
        for gene_id, meta in gene_metadata.items():
            if isinstance(meta, dict) and meta.get("length") is not None:
                property_map[gene_id] = meta["length"]
    elif property_key == "exon_count":
        property_map = {}
        for gene_id, meta in gene_metadata.items():
            if isinstance(meta, dict) and meta.get("exon_count") is not None:
                property_map[gene_id] = meta["exon_count"]
    else:
        raise ValueError(f"Unknown property key: {property_key}")
    
    if not property_map:
        print(f"⚠️ No gene property data available for {property_key}.")
        return None
    
    severity_df = compute_collapse_severity(
        embeddings_dict, metric=metric, threshold=threshold
    )
    if severity_df.empty:
        print("⚠️ No collapse severity data available.")
        return None
    
    severity_df["Property"] = severity_df["Gene"].map(property_map)
    severity_df = severity_df.dropna(subset=["Property"])
    if severity_df.empty:
        print("⚠️ No matching genes found for property mapping.")
        return None
    
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.regplot(
        data=severity_df,
        x="Property",
        y="CollapseSeverity",
        scatter=False,
        ax=ax,
        color="#333333",
        line_kws={"linewidth": 1.4},
    )
    sns.scatterplot(
        data=severity_df,
        x="Property",
        y="CollapseSeverity",
        hue="Strategy",
        ax=ax,
        s=60,
        edgecolor="white",
        linewidth=0.4,
    )
    
    xlabel = "Gene Length (bp)" if property_key == "length" else "Exon Count"
    ylabel = "Final Similarity" if metric == "final_similarity" else f"Iter < {threshold}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = "Collapse Severity vs Gene Property"
    ax.set_title(title)
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_gene_entropy_pair(
    sequences_dict,
    gene_a,
    gene_b,
    model_label=None,
    strategies=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot paired entropy time-series for two genes with mean and std shading.
    
    Args:
        sequences_dict (dict): {gene_id: {strategy: [sequences]}}
        gene_a (str): First gene name
        gene_b (str): Second gene name
        model_label (str or None): Optional title suffix
        strategies (list or None): Optional strategy filter
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    
    def aggregate_entropy(gene_id):
        if gene_id not in sequences_dict:
            return [], []
        series_list = []
        for strategy_key, seqs in sequences_dict[gene_id].items():
            if strategies and strategy_key not in strategies:
                continue
            if not seqs:
                continue
            series_list.append([calculate_shannon_entropy(seq) for seq in seqs])
        if not series_list:
            return [], []
        max_len = max(len(series) for series in series_list)
        means = []
        stds = []
        for idx in range(max_len):
            values = [series[idx] for series in series_list if idx < len(series)]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        return means, stds
    
    mean_a, std_a = aggregate_entropy(gene_a)
    mean_b, std_b = aggregate_entropy(gene_b)
    
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, gene_id, means, stds in [
        (axes[0], gene_a, mean_a, std_a),
        (axes[1], gene_b, mean_b, std_b),
    ]:
        if not means:
            ax.set_title(f"{gene_id} (no data)")
            ax.set_xlabel("Iteration")
            continue
        x_axis = np.arange(len(means))
        ax.plot(x_axis, means, color="#1f77b4", linewidth=2)
        ax.fill_between(x_axis, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color="#1f77b4", alpha=0.2)
        ax.set_title(gene_id)
        ax.set_xlabel("Iteration")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axes[0].set_ylabel("Shannon Entropy")
    title = "Entropy Over Iterations"
    if model_label:
        title = f"{title} ({model_label})"
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_gc_content_drift_boxplot(
    sequences_dict,
    gene_groups,
    iteration_index=-1,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot GC content drift for coding vs non-coding groups.
    
    Args:
        sequences_dict (dict): {gene_id: {strategy: [sequences]}}
        gene_groups (dict): {gene_id: "Coding" or "Non-coding"}
        iteration_index (int): Which iteration to compare against initial
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    
    records = []
    for gene_id, strategies in sequences_dict.items():
        group = gene_groups.get(gene_id)
        if not group:
            continue
        for strategy_key, seqs in strategies.items():
            if not seqs:
                continue
            base_seq = seqs[0]
            if iteration_index < 0:
                target_idx = len(seqs) + iteration_index
            else:
                target_idx = iteration_index
            if target_idx < 0 or target_idx >= len(seqs):
                continue
            drift = abs(compute_gc_content(seqs[target_idx]) - compute_gc_content(base_seq))
            records.append({
                "Group": group,
                "Gene": gene_id,
                "Strategy": strategy_key,
                "GCContentDrift": drift,
            })
    
    if not records:
        print("⚠️ No GC content drift data available.")
        return None
    
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sns.boxplot(
        data=df,
        x="Group",
        y="GCContentDrift",
        ax=ax,
        palette="Set2",
    )
    sns.stripplot(
        data=df,
        x="Group",
        y="GCContentDrift",
        ax=ax,
        color="black",
        alpha=0.35,
        jitter=0.18,
        size=4,
    )
    ax.set_xlabel("Group")
    ax.set_ylabel("GC Content Drift")
    ax.set_title("GC Content Drift by Gene Group")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_cross_validation_heatmap(
    cross_embeddings,
    generator_labels,
    evaluator_labels,
    iteration_range=(0, 50),
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot cross-validation heatmap of semantic similarity.
    
    Args:
        cross_embeddings (dict): {(generator, evaluator): {gene_id: {strategy: [embeddings]}}}
        generator_labels (list): Generator labels (rows)
        evaluator_labels (list): Evaluator labels (cols)
        iteration_range (tuple): Inclusive iteration range to average (start, end)
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    set_publication_style(font_family=font_family, dpi=dpi)
    
    matrix = np.full((len(generator_labels), len(evaluator_labels)), np.nan)
    for i, gen_label in enumerate(generator_labels):
        if isinstance(gen_label, (tuple, list)) and len(gen_label) == 2:
            gen_model, gen_strategy = gen_label
        else:
            gen_model = gen_label
            gen_strategy = None
        for j, eval_label in enumerate(evaluator_labels):
            emb_map = cross_embeddings.get((gen_model, eval_label))
            if not emb_map:
                continue
            values = []
            for gene_id, strategies in emb_map.items():
                if gen_strategy is None:
                    strategy_items = strategies.items()
                else:
                    strategy_items = [(gen_strategy, strategies.get(gen_strategy, []))]
                for _, embs in strategy_items:
                    sims = cosine_series_from_embeddings(embs)
                    if not sims:
                        continue
                    start, end = iteration_range
                    start = max(start, 0)
                    end = min(end, len(sims) - 1)
                    if start > end:
                        continue
                    values.append(float(np.mean(sims[start:end + 1])))
            if values:
                matrix[i, j] = float(np.mean(values))
    
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=evaluator_labels,
        yticklabels=[f"{label[0]}\n{label[1]}" if isinstance(label, (tuple, list)) else str(label)
                     for label in generator_labels],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mean Similarity"},
        ax=ax,
    )
    ax.set_xlabel("Evaluator")
    ax.set_ylabel("Generator")
    ax.set_title("Cross-Validation Semantic Similarity")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


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
