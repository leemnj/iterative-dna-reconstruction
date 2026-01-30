"""Sequence generation helpers."""

import gc
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from tqdm import tqdm as tqdm_std


def _resolve_results_dir(results_dir):
    return Path(results_dir) if results_dir is not None else Path("results")


def generate_sequences(
    gene_selection,
    models,
    decoding_strategies,
    iterations=50,
    mask_ratio=0.15,
    save_all=True,
    save_interval=1,
    results_dir=None,
    store_in_memory=False,
    use_notebook_tqdm=True,
):
    """
    Generate evolved sequences and optionally save as CSV per gene/model.

    Args:
        gene_selection (dict): {gene_name: sequence}
        models (dict): {model_label: SequenceEvolver}
        decoding_strategies (dict): strategy configs
        iterations (int): number of evolution steps
        mask_ratio (float): masking ratio per step
        save_all (bool): save all intermediate sequences
        save_interval (int): interval for saving when save_all=False
        results_dir (str|Path): base directory for outputs (default: results/)
        store_in_memory (bool): return full results in memory
        use_notebook_tqdm (bool): use notebook-friendly tqdm

    Returns:
        dict: {model_label: {gene_name: {strategy_key: [seqs]}}} when store_in_memory else {}
    """
    results_dir = _resolve_results_dir(results_dir)
    seq_dir = results_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    tqdm_func = tqdm if use_notebook_tqdm else tqdm_std
    all_sequences = {}

    for model_label, model_instance in models.items():
        model_name = model_label.replace("/", "-")
        model_dir = seq_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        if store_in_memory:
            all_sequences[model_name] = {}

        gene_iter = tqdm_func(
            gene_selection.items(),
            desc=f"{model_label} genes",
            leave=False,
        )

        for gene_id, original_sequence in gene_iter:
            if original_sequence is None or original_sequence == "":
                print(f"⚠️  Skipping {gene_id}: empty sequence")
                continue
            if not isinstance(original_sequence, str):
                try:
                    original_sequence = str(original_sequence)
                except Exception:
                    print(f"⚠️  Skipping {gene_id}: sequence is not string-like")
                    continue
            if original_sequence.lower() == "nan":
                print(f"⚠️  Skipping {gene_id}: sequence is NaN")
                continue

            output_csv = model_dir / f"{gene_id}.csv"
            results_data = {}

            for strategy_base_key, strategy_cfg in decoding_strategies.items():
                strategy_type = strategy_cfg["type"]
                temperatures = strategy_cfg.get("temperatures", [1.0])
                top_k = strategy_cfg.get("top_k", 50)

                for temp in temperatures:
                    if strategy_type == "greedy":
                        strategy_key = strategy_base_key
                    else:
                        strategy_key = f"{strategy_base_key}_t{temp}"

                    generated_sequences = model_instance.run(
                        sequence=original_sequence,
                        steps=iterations,
                        mask_ratio=mask_ratio,
                        strategy=strategy_type,
                        temperature=temp,
                        top_k=top_k,
                        save_all=save_all,
                        save_interval=save_interval,
                    )

                    results_data[strategy_key] = generated_sequences

                    del generated_sequences
                    gc.collect()
                    if model_instance.device == "cuda":
                        import torch
                        torch.cuda.empty_cache()
                    elif model_instance.device == "mps":
                        import torch
                        torch.mps.empty_cache()

            df = pd.DataFrame(results_data).T
            df.columns = [f"iteration_{i}" for i in range(df.shape[1])]
            df.to_csv(output_csv)

            if store_in_memory:
                all_sequences[model_name][gene_id] = results_data

    return all_sequences if store_in_memory else {}
