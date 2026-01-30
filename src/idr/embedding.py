"""Embedding helpers."""

import gc
from pathlib import Path
import pickle

from .utils import load_model


def _resolve_results_dir(results_dir):
    return Path(results_dir) if results_dir is not None else Path("results")


def build_cross_embeddings(source_sequences, target_model_instance):
    """
    Build embeddings for a set of sequences using a target model.

    Args:
        source_sequences (dict): {gene_id: {strategy: [seqs]}}
        target_model_instance (SequenceEvolver)

    Returns:
        dict: embeddings with same structure
    """
    cross_emb = {}
    for gene_id, strategies in source_sequences.items():
        cross_emb[gene_id] = {}
        for strategy, sequences in strategies.items():
            if not sequences:
                continue
            embeddings = []
            for seq in sequences:
                if seq is None or seq == "":
                    continue
                emb = target_model_instance.get_embedding(str(seq))
                embeddings.append(emb)
            if embeddings:
                cross_emb[gene_id][strategy] = embeddings
            gc.collect()
            if target_model_instance.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            elif target_model_instance.device == "mps":
                import torch
                torch.mps.empty_cache()
    return cross_emb


def embed_sequences(
    all_sequences,
    model_labels,
    model_configs,
    device,
    torch_dtype="auto",
    results_dir=None,
):
    """
    Compute embeddings for generated sequences across model pairs.

    Args:
        all_sequences (dict): {model_name: {gene_id: {strategy: [seqs]}}}
        model_labels (list): list of model labels used for generation/evaluation
        model_configs (dict): {label: model_path}
        device (str): device string
        torch_dtype (str|torch.dtype): dtype for model loading
        results_dir (str|Path): base directory for outputs (default: results/)

    Returns:
        dict: {(gen_label, eval_label): embeddings}
    """
    results_dir = _resolve_results_dir(results_dir)
    emb_dir = results_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    generator_labels = list(model_labels)
    evaluator_labels = list(model_labels)
    cross_embeddings = {}

    for eval_label in evaluator_labels:
        model_instance = load_model(
            device,
            eval_label,
            model_configs[eval_label],
            torch_dtype=torch_dtype,
        )

        for gen_label in generator_labels:
            cache_name = f"embeddings_{gen_label}__by__{eval_label}.pkl"
            cache_path = emb_dir / cache_name

            source_key = gen_label.replace("/", "-")
            cross_emb = build_cross_embeddings(
                all_sequences[source_key],
                model_instance,
            )
            cross_embeddings[(gen_label, eval_label)] = cross_emb
            with open(cache_path, "wb") as f:
                pickle.dump(cross_emb, f, protocol=4)

        del model_instance
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        elif device == "mps":
            import torch
            torch.mps.empty_cache()

    return cross_embeddings
