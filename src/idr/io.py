"""I/O utilities for saving and loading sequences/embeddings."""

import gzip
import json
import pickle
from pathlib import Path


def save_sequences_compressed(data, filepath, compression="gzip"):
    """Save sequences with compression."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if compression == "gzip":
        with gzip.open(filepath.with_suffix(".json.gz"), "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {filepath.with_suffix('.json.gz')} (compressed)")
    elif compression == "json":
        with open(filepath.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {filepath.with_suffix('.json')}")
    elif compression == "pickle":
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved to {filepath.with_suffix('.pkl')}")
    elif compression == "pickle_gzip":
        with gzip.open(filepath.with_suffix(".pkl.gz"), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved to {filepath.with_suffix('.pkl.gz')} (compressed)")


def load_sequences_compressed(filepath):
    """Load sequences from compressed files."""
    filepath = Path(filepath)

    if filepath.suffix == ".gz":
        if len(filepath.suffixes) > 1 and filepath.suffixes[-2] == ".json":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                return json.load(f)
        if len(filepath.suffixes) > 1 and filepath.suffixes[-2] == ".pkl":
            with gzip.open(filepath, "rb") as f:
                return pickle.load(f)
    elif filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unsupported file format: {filepath}")


def load_from_parts(base_dir):
    """Load data from parts directory structure."""
    data = {}
    base_dir = Path(base_dir)

    if not base_dir.exists():
        return data

    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(base_dir)
        if len(rel.parts) < 3:
            continue

        gene_name, model_label = rel.parts[0], rel.parts[1]
        filename = rel.name

        if filename.endswith(".json.gz"):
            strategy_key = filename[:-8]
        elif filename.endswith(".pkl.gz"):
            strategy_key = filename[:-7]
        else:
            continue

        data.setdefault(gene_name, {}).setdefault(model_label, {})[strategy_key] = (
            load_sequences_compressed(path)
        )

    return data
