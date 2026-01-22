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
from sklearn.decomposition import PCA
from collections import Counter
from io_utils import load_sequences_compressed, load_from_parts
from metrics_utils import (
    cosine_series_from_embeddings,
    calculate_shannon_entropy,
    prepare_sequence_metrics,
    compute_gc_content,
    infer_gene_lengths,
    compute_kmer_frequencies,
    aggregate_kmer_frequencies,
    compute_vendi_score,
)


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

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    _plot_strategy_collapse_with_ci_on_ax(
        ax,
        embeddings_dict,
        model_label=model_label,
        strategies=strategies,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def _plot_strategy_collapse_with_ci_on_ax(
    ax,
    embeddings_dict,
    model_label=None,
    strategies=None,
    show_ylabel=True,
    show_legend=True,
):
    df = build_similarity_records(embeddings_dict, strategies=strategies)
    if df.empty:
        ax.set_title("No similarity data")
        ax.set_axis_off()
        return

    summary = (
        df.groupby(["Strategy", "Iteration"])["Similarity"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    strategy_order = strategies or list(summary["Strategy"].unique())
    palette = sns.color_palette("tab10", n_colors=len(strategy_order))

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
    if show_ylabel:
        ax.set_ylabel("Cosine Similarity")
    else:
        ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if model_label:
        ax.set_title(model_label)
    if show_legend:
        ax.legend(title="Strategy")


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


def build_final_similarity_records(all_embeddings, strategies=None):
    """
    Build long-form records of final-iteration similarity per model/gene/strategy.
    
    Args:
        all_embeddings (dict): {model_label: {gene_id: {strategy: [embeddings]}}}
        strategies (list or None): Optional strategy filter
    
    Returns:
        pd.DataFrame: Columns [Model, Gene, Strategy, FinalSimilarity]
    """
    records = []
    for model_label, embeddings_dict in all_embeddings.items():
        for gene_id, strategies_map in embeddings_dict.items():
            for strategy_key, embs in strategies_map.items():
                if strategies and strategy_key not in strategies:
                    continue
                sims = cosine_series_from_embeddings(embs)
                if not sims:
                    continue
                records.append({
                    "Model": model_label,
                    "Gene": gene_id,
                    "Strategy": strategy_key,
                    "FinalSimilarity": sims[-1],
                })
    return pd.DataFrame(records)


def plot_final_similarity_vs_gene_property_by_model(
    all_embeddings,
    sequences_by_model=None,
    gene_metadata=None,
    property_key="length",
    strategies=None,
    prefer_strategy="greedy",
    prefer_length_model=None,
    aggregate="mean",
    log_x=False,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot final-iteration similarity vs gene property, colored by model.
    
    Args:
        all_embeddings (dict): {model_label: {gene_id: {strategy: [embeddings]}}}
        sequences_by_model (dict or None): {model_label: {gene_id: {strategy: [sequences]}}}
        gene_metadata (dict or None): {gene_id: {"length": int, "exon_count": int}}
        property_key (str): "length" or "exon_count"
        strategies (list or None): Optional strategy filter
        prefer_strategy (str): Strategy key to use for length inference
        prefer_length_model (str or None): Model key to use for length inference
        aggregate (str or None): "mean", "median", or None for per-strategy points
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    gene_metadata = gene_metadata or {}
    sequences_by_model = sequences_by_model or {}
    
    if property_key not in {"length", "exon_count"}:
        raise ValueError(f"Unknown property key: {property_key}")
    
    property_map = {}
    if property_key == "length":
        model_key = prefer_length_model or (next(iter(sequences_by_model), None))
        if model_key and model_key in sequences_by_model:
            property_map = infer_gene_lengths(
                sequences_by_model[model_key], prefer_strategy=prefer_strategy
            )
        for gene_id, meta in gene_metadata.items():
            if isinstance(meta, dict) and meta.get("length") is not None:
                property_map[gene_id] = meta["length"]
    else:
        for gene_id, meta in gene_metadata.items():
            if isinstance(meta, dict) and meta.get("exon_count") is not None:
                property_map[gene_id] = meta["exon_count"]
    
    if not property_map:
        print(f"⚠️ No gene property data available for {property_key}.")
        return None
    
    records = build_final_similarity_records(all_embeddings, strategies=strategies)
    if records.empty:
        print("⚠️ No final similarity records available.")
        return None
    
    records["Property"] = records["Gene"].map(property_map)
    records = records.dropna(subset=["Property"])
    if records.empty:
        print("⚠️ No matching genes found for property mapping.")
        return None
    
    plot_df = records
    if aggregate in {"mean", "median"}:
        plot_df = (
            records.groupby(["Model", "Gene", "Property"])["FinalSimilarity"]
            .agg(aggregate)
            .reset_index()
        )
    
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.scatterplot(
        data=plot_df,
        x="Property",
        y="FinalSimilarity",
        hue="Model",
        style=None if aggregate in {"mean", "median"} else "Strategy",
        ax=ax,
        s=70,
        edgecolor="white",
        linewidth=0.4,
    )
    
    xlabel = "Gene Length (bp)" if property_key == "length" else "Exon Count"
    ax.set_xlabel(xlabel)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylabel("Final Cosine Similarity")
    ax.set_title("Final Similarity vs Gene Property")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_final_similarity_vs_gene_property_by_strategy(
    embeddings_dict,
    sequences_dict=None,
    gene_metadata=None,
    property_key="length",
    strategies=None,
    prefer_strategy="greedy",
    model_label=None,
    log_x=False,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot final-iteration similarity vs gene property, one subplot per strategy.
    Use `plot_final_similarity_vs_gene_property` for a shorter name.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        sequences_dict (dict or None): {gene_id: {strategy: [sequences]}}
        gene_metadata (dict or None): {gene_id: {"length": int, "exon_count": int}}
        property_key (str): "length" or "exon_count"
        strategies (list or None): Optional strategy filter/order
        prefer_strategy (str): Strategy key to use for length inference
        model_label (str or None): Optional title prefix
        log_x (bool): Use log scale for x-axis
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    df, property_map = _build_final_similarity_property_df(
        embeddings_dict,
        sequences_dict=sequences_dict,
        gene_metadata=gene_metadata,
        property_key=property_key,
        strategies=strategies,
        prefer_strategy=prefer_strategy,
    )
    if not property_map:
        print(f"⚠️ No gene property data available for {property_key}.")
        return None
    if df.empty:
        print("⚠️ No matching genes found for property mapping.")
        return None
    
    strategy_order = strategies or list(df["Strategy"].unique())
    n_strategies = len(strategy_order)
    if n_strategies == 0:
        print("⚠️ No strategies available for plot.")
        return None
    
    ncols = 3
    nrows = (n_strategies + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.2 * ncols, 4.6 * nrows), squeeze=False
    )
    palette = sns.color_palette("tab10", n_colors=n_strategies)
    
    for idx, strategy_key in enumerate(strategy_order):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
        subset = df[df["Strategy"] == strategy_key]
        sns.scatterplot(
            data=subset,
            x="Property",
            y="FinalSimilarity",
            ax=ax,
            s=70,
            color=palette[idx],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(f"{strategy_key}")
        ax.set_ylabel("Final Cosine Similarity")
        ax.set_ylim(0, 1)
        if log_x:
            ax.set_xscale("log")
        xlabel = "Gene Length (bp)" if property_key == "length" else "Exon Count"
        ax.set_xlabel(xlabel)
    
    for idx in range(n_strategies, nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx][col_idx].axis("off")
    
    title = "Final Similarity vs Gene Property"
    if model_label:
        title = f"{model_label} - {title}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_final_similarity_vs_gene_property(
    embeddings_dict,
    sequences_dict=None,
    gene_metadata=None,
    property_key="length",
    strategy=None,
    prefer_strategy="greedy",
    model_label=None,
    log_x=False,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot final-iteration similarity vs gene property for a single strategy.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        sequences_dict (dict or None): {gene_id: {strategy: [sequences]}}
        gene_metadata (dict or None): {gene_id: {"length": int, "exon_count": int}}
        property_key (str): "length" or "exon_count"
        strategy (str or None): Strategy key to plot (default: first available)
        prefer_strategy (str): Strategy key to use for length inference
        model_label (str or None): Optional title prefix
        log_x (bool): Use log scale for x-axis
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    df, property_map = _build_final_similarity_property_df(
        embeddings_dict,
        sequences_dict=sequences_dict,
        gene_metadata=gene_metadata,
        property_key=property_key,
        strategies=[strategy] if strategy else None,
        prefer_strategy=prefer_strategy,
    )
    if not property_map:
        print(f"⚠️ No gene property data available for {property_key}.")
        return None
    if df.empty:
        print("⚠️ No matching genes found for property mapping.")
        return None
    
    if strategy is None:
        strategy = df["Strategy"].iloc[0]
    subset = df[df["Strategy"] == strategy]
    if subset.empty:
        print(f"⚠️ Strategy '{strategy}' not found in data.")
        return None
    
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.scatterplot(
        data=subset,
        x="Property",
        y="FinalSimilarity",
        ax=ax,
        s=70,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_ylabel("Final Cosine Similarity")
    ax.set_ylim(0, 1)
    if log_x:
        ax.set_xscale("log")
    xlabel = "Gene Length (bp)" if property_key == "length" else "Exon Count"
    ax.set_xlabel(xlabel)
    title = f"{strategy} - Final Similarity vs Gene Property"
    if model_label:
        title = f"{model_label} - {title}"
    ax.set_title(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def _build_final_similarity_property_df(
    embeddings_dict,
    sequences_dict=None,
    gene_metadata=None,
    property_key="length",
    strategies=None,
    prefer_strategy=None,
):
    gene_metadata = gene_metadata or {}
    sequences_dict = sequences_dict or {}

    if property_key not in {"length", "exon_count"}:
        raise ValueError(f"Unknown property key: {property_key}")

    property_map = {}
    if property_key == "length":
        property_map = infer_gene_lengths(
            sequences_dict, prefer_strategy=prefer_strategy
        )
        for gene_id, meta in gene_metadata.items():
            if isinstance(meta, dict) and meta.get("length") is not None:
                property_map[gene_id] = meta["length"]
    else:
        for gene_id, meta in gene_metadata.items():
            if isinstance(meta, dict) and meta.get("exon_count") is not None:
                property_map[gene_id] = meta["exon_count"]

    if not property_map:
        return pd.DataFrame(), property_map

    records = []
    for gene_id, strategies_map in embeddings_dict.items():
        for strategy_key, embs in strategies_map.items():
            if strategies and strategy_key not in strategies:
                continue
            sims = cosine_series_from_embeddings(embs)
            if not sims:
                continue
            records.append({
                "Gene": gene_id,
                "Strategy": strategy_key,
                "FinalSimilarity": sims[-1],
                "Property": property_map.get(gene_id),
            })

    df = pd.DataFrame(records)
    return df, property_map



def plot_final_similarity_raincloud_by_gene_type(
    embeddings_dict,
    gene_type_map,
    strategies=None,
    model_label=None,
    palette=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot raincloud-style distribution of final similarity by gene type.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        gene_type_map (dict): {gene_id: "Coding"/"Non-coding"}
        strategies (list or None): Optional strategy filter
        model_label (str or None): Optional title prefix
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    if not gene_type_map:
        print("⚠️ No gene type metadata provided.")
        return None
    
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = ["Coding", "Non-coding"]
    _plot_final_similarity_raincloud_on_ax(
        ax,
        embeddings_dict,
        category_map=gene_type_map,
        category_label="Gene Type",
        order=order,
        strategies=strategies,
        model_label=None,
        palette=palette or {"Coding": "#4C78A8", "Non-coding": "#F58518"},
        show_ylabel=True,
    )
    title = "Final Similarity by Gene Type"
    if model_label:
        title = f"{model_label} - {title}"
    ax.set_title(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def _plot_final_similarity_raincloud_on_ax(
    ax,
    embeddings_dict,
    category_map,
    category_label,
    order,
    strategies=None,
    model_label=None,
    palette=None,
    show_ylabel=True,
    gene_type_map=None,
    type_filter=None,
):
    if not category_map:
        ax.set_axis_off()
        return

    records = []
    for gene_id, strategies_map in embeddings_dict.items():
        category_value = category_map.get(gene_id)
        if category_value is None:
            continue
        if type_filter and gene_type_map is not None:
            if gene_type_map.get(gene_id) != type_filter:
                continue
        for strategy_key, embs in strategies_map.items():
            if strategies and strategy_key not in strategies:
                continue
            sims = cosine_series_from_embeddings(embs)
            if not sims:
                continue
            records.append({
                "Gene": gene_id,
                "Category": category_value,
                "Strategy": strategy_key,
                "FinalSimilarity": sims[-1],
            })
    df = pd.DataFrame(records)
    if df.empty:
        ax.set_axis_off()
        return

    if palette is None:
        palette = {}

    before = len(ax.collections)
    sns.violinplot(
        data=df,
        x="FinalSimilarity",
        y="Category",
        hue="Category",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
        legend=False,
    )
    for artist in ax.collections[before:]:
        artist.set_alpha(0.4)
    sns.boxplot(
        data=df,
        x="FinalSimilarity",
        y="Category",
        order=order,
        width=0.18,
        showcaps=False,
        boxprops={"facecolor": "white", "alpha": 0.9},
        showfliers=False,
        whiskerprops={"linewidth": 1.0},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="FinalSimilarity",
        y="Category",
        hue="Category",
        order=order,
        size=4,
        jitter=0.25,
        alpha=0.6,
        palette=palette,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Final Cosine Similarity")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylabel(category_label if show_ylabel else "")

    if model_label:
        ax.set_title(model_label)



def plot_final_similarity_raincloud_by_gene_status(
    embeddings_dict,
    gene_status_map,
    gene_type_map=None,
    type_filter=None,
    strategies=None,
    model_label=None,
    palette=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot raincloud-style distribution of final similarity by gene status.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        gene_status_map (dict): {gene_id: "Real"/"Pseudogene"}
        gene_type_map (dict or None): {gene_id: "Coding"/"Non-coding"}
        type_filter (str or None): Optional gene type to filter ("Non-coding")
        strategies (list or None): Optional strategy filter
        model_label (str or None): Optional title prefix
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    if not gene_status_map:
        print("⚠️ No gene status metadata provided.")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = ["Real", "Pseudogene"]
    _plot_final_similarity_raincloud_on_ax(
        ax,
        embeddings_dict,
        category_map=gene_status_map,
        category_label="Gene Status",
        order=order,
        strategies=strategies,
        model_label=None,
        palette=palette or {"Real": "#54A24B", "Pseudogene": "#E45756"},
        show_ylabel=True,
        gene_type_map=gene_type_map,
        type_filter=type_filter,
    )
    title = "Final Similarity by Gene Status"
    if type_filter:
        title = f"{title} ({type_filter})"
    if model_label:
        title = f"{model_label} - {title}"
    ax.set_title(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def _plot_final_similarity_raincloud_by_gene_status_on_ax(
    ax,
    embeddings_dict,
    gene_status_map,
    gene_type_map=None,
    type_filter=None,
    strategies=None,
    model_label=None,
    palette=None,
    show_ylabel=True,
):
    _plot_final_similarity_raincloud_on_ax(
        ax,
        embeddings_dict,
        category_map=gene_status_map,
        category_label="Gene Status",
        order=["Real", "Pseudogene"],
        strategies=strategies,
        model_label=model_label,
        palette=palette or {"Real": "#54A24B", "Pseudogene": "#E45756"},
        show_ylabel=show_ylabel,
        gene_type_map=gene_type_map,
        type_filter=type_filter,
    )


def plot_pca_trajectory_gene_pairs(
    embeddings_dict,
    gene_pairs,
    strategy="greedy",
    include_random=True,
    random_seed=42,
    random_repeats=1,
    real_point_only=True,
    show_real_legend=False,
    model_label=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot PCA trajectories for real/pseudogene pairs with optional random control.
    
    Args:
        embeddings_dict (dict): {gene_id: {strategy: [embeddings]}}
        gene_pairs (list): List of (real_gene, pseudogene) tuples
        strategy (str): Strategy key to use
        include_random (bool): Add random-walk control from pseudogene start
        random_seed (int): RNG seed for random control
        model_label (str or None): Optional title prefix
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    
    def flatten_embeddings(emb_list):
        flat = []
        for emb in emb_list:
            if emb is None:
                continue
            vec = np.asarray(emb, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(vec)):
                continue
            flat.append(vec)
        return flat
    
    def random_walk_from_steps(start_vec, step_lengths, rng):
        if start_vec is None:
            return []
        if not step_lengths:
            return [start_vec.copy()]
        dim = start_vec.shape[0]
        points = [start_vec.copy()]
        current = start_vec.copy()
        for step_len in step_lengths:
            direction = rng.normal(size=dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = rng.normal(size=dim)
                norm = np.linalg.norm(direction)
            direction = direction / max(norm, 1e-12)
            current = current + direction * step_len
            points.append(current.copy())
        return points
    
    if not gene_pairs:
        print("⚠️ No gene pairs provided for PCA trajectory plot.")
        return None
    
    n_pairs = len(gene_pairs)
    ncols = 3
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False
    )
    rng = np.random.default_rng(random_seed)
    
    for idx, (real_gene, pseudo_gene) in enumerate(gene_pairs):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
        real_embs = flatten_embeddings(
            embeddings_dict.get(real_gene, {}).get(strategy, [])
        )
        if real_point_only and real_embs:
            real_embs = [real_embs[0]]
        pseudo_embs = flatten_embeddings(
            embeddings_dict.get(pseudo_gene, {}).get(strategy, [])
        )
        if not real_embs or not pseudo_embs:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"Missing embeddings for {real_gene} / {pseudo_gene}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        
        step_lengths = [
            float(np.linalg.norm(pseudo_embs[i] - pseudo_embs[i - 1]))
            for i in range(1, len(pseudo_embs))
        ]
        random_walks = []
        if include_random:
            repeats = max(1, int(random_repeats))
            for _ in range(repeats):
                random_walks.append(
                    random_walk_from_steps(pseudo_embs[0], step_lengths, rng)
                )
        
        combined = real_embs + pseudo_embs + [pt for walk in random_walks for pt in walk]
        combined_matrix = np.vstack(combined).astype(np.float64)
        if not np.all(np.isfinite(combined_matrix)):
            combined_matrix = combined_matrix[np.all(np.isfinite(combined_matrix), axis=1)]
        if combined_matrix.shape[0] < 2:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"Insufficient finite data for {real_gene} / {pseudo_gene}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        mean = combined_matrix.mean(axis=0)
        std = combined_matrix.std(axis=0)
        std[std == 0] = 1.0
        combined_matrix = (combined_matrix - mean) / std
        pca = PCA(n_components=2, svd_solver="full")
        coords = pca.fit_transform(combined_matrix)
        real_coords = coords[:len(real_embs)]
        pseudo_coords = coords[len(real_embs):len(real_embs) + len(pseudo_embs)]
        random_coords = coords[len(real_embs) + len(pseudo_embs):]
        
        real_label = "Real" if show_real_legend else "_nolegend_"
        ax.plot(real_coords[:, 0], real_coords[:, 1], color="#4C78A8", label=real_label)
        ax.plot(pseudo_coords[:, 0], pseudo_coords[:, 1], color="#E45756", label="Pseudogene")
        if include_random and random_walks:
            offset = 0
            for walk_idx, walk in enumerate(random_walks):
                if not walk:
                    continue
                walk_len = len(walk)
                walk_coords = random_coords[offset:offset + walk_len]
                offset += walk_len
                label = "Random" if walk_idx == 0 else "_nolegend_"
                ax.plot(
                    walk_coords[:, 0],
                    walk_coords[:, 1],
                    color="#7F7F7F",
                    linestyle="--",
                    alpha=0.6,
                    label=label,
                )
        
        ax.scatter(
            real_coords[0, 0],
            real_coords[0, 1],
            color="#4C78A8",
            marker="*",
            s=70,
            label="Real (iter0)",
        )
        ax.scatter(
            pseudo_coords[0, 0],
            pseudo_coords[0, 1],
            color="#E45756",
            marker="o",
            s=50,
            label="Pseudo start",
        )
        ax.scatter(
            pseudo_coords[-1, 0],
            pseudo_coords[-1, 1],
            color="#E45756",
            marker="s",
            s=50,
            label="Pseudo end",
        )
        ax.annotate(
            "",
            xy=(pseudo_coords[-1, 0], pseudo_coords[-1, 1]),
            xytext=(pseudo_coords[0, 0], pseudo_coords[0, 1]),
            arrowprops={"arrowstyle": "->", "color": "#E45756", "alpha": 0.6},
        )
        
        ax.set_title(f"{real_gene} vs {pseudo_gene}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
    
    for idx in range(n_pairs, nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx][col_idx].axis("off")
    
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    title = "PCA Trajectories (Real vs Pseudogene)"
    if model_label:
        title = f"{model_label} - {title}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def _flatten_embeddings_list(emb_list):
    flat = []
    for emb in emb_list:
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(vec)):
            continue
        flat.append(vec)
    return flat


def compute_vendi_series_across_genes(embeddings_dict, strategy):
    """
    Compute Vendi score across genes for each iteration.
    """
    gene_series = []
    for gene_id, strategies_map in embeddings_dict.items():
        embs = strategies_map.get(strategy, [])
        if not embs:
            continue
        gene_series.append(_flatten_embeddings_list(embs))

    if not gene_series:
        return []

    min_len = min(len(series) for series in gene_series)
    vendi_series = []
    for idx in range(min_len):
        step_embeddings = [series[idx] for series in gene_series if idx < len(series)]
        vendi_series.append(compute_vendi_score(step_embeddings))
    return vendi_series


def compute_gene_vendi_over_time(embeddings_dict, strategy):
    """
    Compute per-gene Vendi score over the full trajectory.
    """
    vendi_by_gene = {}
    for gene_id, strategies_map in embeddings_dict.items():
        embs = strategies_map.get(strategy, [])
        flat = _flatten_embeddings_list(embs)
        if not flat:
            continue
        vendi_by_gene[gene_id] = compute_vendi_score(flat)
    return vendi_by_gene


def plot_vendi_model_comparison(
    all_embeddings,
    strategy="sampling_t1.0",
    model_labels=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Compare models using Vendi score at a fixed decoding strategy.

    Top: Vendi across genes over iterations.
    Bottom: Per-gene Vendi over trajectory (model comparison).
    """
    if not all_embeddings:
        print("⚠️ No embeddings available for Vendi comparison.")
        return None

    model_labels = model_labels or list(all_embeddings.keys())
    if not model_labels:
        print("⚠️ No models available for Vendi comparison.")
        return None

    series_records = []
    gene_records = []
    for model_label in model_labels:
        embeddings_dict = all_embeddings.get(model_label, {})
        series = compute_vendi_series_across_genes(embeddings_dict, strategy)
        for idx, value in enumerate(series):
            series_records.append({
                "Model": model_label,
                "Iteration": idx,
                "Vendi": value,
            })
        vendi_by_gene = compute_gene_vendi_over_time(embeddings_dict, strategy)
        for gene_id, value in vendi_by_gene.items():
            gene_records.append({
                "Model": model_label,
                "Gene": gene_id,
                "Vendi": value,
            })

    series_df = pd.DataFrame(series_records)
    gene_df = pd.DataFrame(gene_records)

    fig = plt.figure(figsize=(8.6, 6.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.3)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_bottom = fig.add_subplot(grid[1, 0])
    ax_bottom.grid(True, linestyle="--", alpha=0.4)

    if not series_df.empty:
        sns.lineplot(
            data=series_df,
            x="Iteration",
            y="Vendi",
            hue="Model",
            marker="o",
            ax=ax_top,
        )
        ax_top.set_title(f"Vendi Over Iterations ({strategy})")
        ax_top.set_xlabel("Iteration")
        ax_top.set_ylabel("Vendi Score")
        ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax_top.set_axis_off()

    if not gene_df.empty:
        sns.boxplot(
            data=gene_df,
            x="Model",
            y="Vendi",
            ax=ax_bottom,
        )
        sns.stripplot(
            data=gene_df,
            x="Model",
            y="Vendi",
            color="black",
            alpha=0.35,
            jitter=0.18,
            size=4,
            ax=ax_bottom,
        )
        ax_bottom.set_xlabel("Model")
        ax_bottom.set_ylabel("Gene Vendi Score")
    else:
        ax_bottom.set_axis_off()

    fig.suptitle("Model Comparison with Vendi Score", y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_pca_trajectory_models_for_pair(
    all_embeddings,
    gene_pair,
    strategy="greedy",
    real_point_only=True,
    model_label=None,
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot PCA trajectories for a gene pair across multiple models on one plot.
    
    Args:
        all_embeddings (dict): {model_label: {gene_id: {strategy: [embeddings]}}}
        gene_pair (tuple): (real_gene, pseudogene)
        strategy (str): Strategy key to use
        real_point_only (bool): Use only iteration 0 for real gene
        model_label (str or None): Optional title prefix
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    
    def flatten_embeddings(emb_list):
        flat = []
        for emb in emb_list:
            if emb is None:
                continue
            vec = np.asarray(emb, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(vec)):
                continue
            flat.append(vec)
        return flat
    
    real_gene, pseudo_gene = gene_pair
    combined = []
    model_slices = []
    for model_name, embeddings_dict in all_embeddings.items():
        real_embs = flatten_embeddings(
            embeddings_dict.get(real_gene, {}).get(strategy, [])
        )
        if real_point_only and real_embs:
            real_embs = [real_embs[0]]
        pseudo_embs = flatten_embeddings(
            embeddings_dict.get(pseudo_gene, {}).get(strategy, [])
        )
        if not real_embs or not pseudo_embs:
            continue
        start = len(combined)
        combined.extend(real_embs + pseudo_embs)
        model_slices.append({
            "model": model_name,
            "start": start,
            "real_len": len(real_embs),
            "pseudo_len": len(pseudo_embs),
        })
    
    if not combined or not model_slices:
        print(f"⚠️ No embeddings available for {real_gene} / {pseudo_gene}.")
        return None
    
    combined_matrix = np.vstack(combined).astype(np.float64)
    if not np.all(np.isfinite(combined_matrix)):
        combined_matrix = combined_matrix[np.all(np.isfinite(combined_matrix), axis=1)]
    if combined_matrix.shape[0] < 2:
        print(f"⚠️ Insufficient data for PCA on {real_gene} / {pseudo_gene}.")
        return None
    
    mean = combined_matrix.mean(axis=0)
    std = combined_matrix.std(axis=0)
    std[std == 0] = 1.0
    combined_matrix = (combined_matrix - mean) / std
    pca = PCA(n_components=2, svd_solver="full")
    coords = pca.fit_transform(combined_matrix)
    
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    palette = sns.color_palette("tab10", n_colors=len(model_slices))
    
    for idx, model_info in enumerate(model_slices):
        start = model_info["start"]
        end = start + model_info["real_len"] + model_info["pseudo_len"]
        model_coords = coords[start:end]
        real_coords = model_coords[:model_info["real_len"]]
        pseudo_coords = model_coords[model_info["real_len"]:]
        color = palette[idx]
        
        ax.plot(
            pseudo_coords[:, 0],
            pseudo_coords[:, 1],
            color=color,
            label=model_info["model"],
            linewidth=1.8,
        )
        ax.scatter(
            pseudo_coords[0, 0],
            pseudo_coords[0, 1],
            color=color,
            marker="o",
            s=40,
        )
        ax.scatter(
            pseudo_coords[-1, 0],
            pseudo_coords[-1, 1],
            color=color,
            marker="s",
            s=40,
        )
        if real_coords.size > 0:
            ax.scatter(
                real_coords[0, 0],
                real_coords[0, 1],
                color=color,
                marker="*",
                s=70,
                edgecolor="white",
                linewidth=0.4,
            )
    
    ax.set_title(f"{real_gene} vs {pseudo_gene} (Models)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    title = "PCA Trajectories Across Models"
    if model_label:
        title = f"{model_label} - {title}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_similarity_to_real_baseline_across_models(
    all_embeddings,
    gene_pair,
    strategy="greedy",
    font_family="Times New Roman",
    dpi=300,
    save_path=None,
):
    """
    Plot pseudogene similarity to real gene baseline across models.
    
    Args:
        all_embeddings (dict): {model_label: {gene_id: {strategy: [embeddings]}}}
        gene_pair (tuple): (real_gene, pseudogene)
        strategy (str): Strategy key to use
        font_family (str): Preferred font family
        dpi (int): Output DPI
        save_path (str or Path or None): Save path for figure (optional)
    """
    real_gene, pseudo_gene = gene_pair
    
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    palette = sns.color_palette("tab10", n_colors=len(all_embeddings))
    
    plotted = 0
    baseline_labeled = False
    for idx, (model_name, embeddings_dict) in enumerate(all_embeddings.items()):
        real_embs = embeddings_dict.get(real_gene, {}).get(strategy, [])
        pseudo_embs = embeddings_dict.get(pseudo_gene, {}).get(strategy, [])
        if not real_embs or not pseudo_embs:
            continue
        base = np.asarray(real_embs[0]).reshape(1, -1)
        pseudo_base = np.asarray(pseudo_embs[0]).reshape(1, -1)
        pseudo_baseline_sim = float(cosine_similarity(base, pseudo_base)[0, 0])
        sims = []
        for emb in pseudo_embs:
            vec = np.asarray(emb).reshape(1, -1)
            sims.append(float(cosine_similarity(base, vec)[0, 0]))
        if not sims:
            continue
        ax.plot(
            list(range(len(sims))),
            sims,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=palette[idx],
            label=model_name,
        )
        baseline_label = "Pseudogene baseline" if not baseline_labeled else "_nolegend_"
        ax.scatter(
            [0],
            [pseudo_baseline_sim],
            color=palette[idx],
            marker="D",
            s=50,
            edgecolor="white",
            linewidth=0.4,
            label=baseline_label,
        )
        baseline_labeled = True
        plotted += 1
    
    if plotted == 0:
        print(f"⚠️ No similarity data for {real_gene} / {pseudo_gene}.")
        return None
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cosine Similarity to Real Baseline")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{real_gene} vs {pseudo_gene} - Similarity to Real Baseline")
    ax.axhline(
        1.0,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="Real baseline",
    )
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


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
