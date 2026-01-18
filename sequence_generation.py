"""
Sequence generation module for DNA sequence evolution and embedding extraction.
Handles gene fetching from NCBI, sequence evolution, and data saving.
"""

import warnings
import json
import gzip
import pickle
import gc
import io
from pathlib import Path
from Bio import Entrez, SeqIO
from tqdm.notebook import tqdm
from tqdm import tqdm as tqdm_std


# Gene collection configurations
DEFAULT_GENE_UIDS = {
    'GAPDH': 'NM_002046.7',
    'STAT3': 'NM_139276.3',
    'GAPDHP1': 'NG_001123.6'
}

DEFAULT_GENES_TO_SEARCH = ['H4C1', 'TP53', 'NORAD']

DEFAULT_DECODING_STRATEGIES = {
    "greedy": {
        "name": "Greedy Search",
        "type": "greedy",
        "temperatures": [1.0],
        "top_k": 50,
    },
    "sampling": {
        "name": "Sampling",
        "type": "sampling",
        "temperatures": [0.5, 1.0, 1.5],
        "top_k": 50,
    },
}


def fetch_gene_sequences(email, gene_uids=None, genes_to_search=None):
    """
    Fetch gene sequences from NCBI Entrez.
    
    Args:
        email (str): Email for NCBI Entrez queries
        gene_uids (dict): Dictionary of {gene_name: UID}. Uses DEFAULT_GENE_UIDS if None.
        genes_to_search (list): List of gene names to search. Uses DEFAULT_GENES_TO_SEARCH if None.
        
    Returns:
        dict: Dictionary of {gene_name: sequence}
    """
    if gene_uids is None:
        gene_uids = DEFAULT_GENE_UIDS
    if genes_to_search is None:
        genes_to_search = DEFAULT_GENES_TO_SEARCH
    
    # Suppress Biopython warning
    warnings.filterwarnings(
        "ignore",
        message="[Entrez.read] WARNING: Empty sequence description"
    )
    
    Entrez.email = email
    gene_selection = {}
    
    print("Fetching gene sequences from NCBI...")
    
    # Fetch genes by specific UID
    for gene_name, uid in gene_uids.items():
        print(f"Fetching {gene_name} by UID {uid}...")
        try:
            handle = Entrez.efetch(db="nucleotide", id=uid, rettype="fasta", retmode="text")
            fasta_record = handle.read()
            handle.close()
            
            seq_obj = SeqIO.read(io.StringIO(fasta_record), "fasta")
            gene_selection[gene_name] = str(seq_obj.seq)
            print(f"  ✅ Found and added sequence for {gene_name} (Length: {len(str(seq_obj.seq))}bp)")
        except Exception as e:
            print(f"  ❌ Error fetching {gene_name} by UID {uid}: {e}")
    
    # Fetch genes by name with filters
    for gene_name in genes_to_search:
        print(f"Searching for {gene_name}...")
        try:
            search_term = (
                f"{gene_name}[Gene Name] AND Homo sapiens[Organism] AND "
                f"(mRNA[filter] OR ncRNA[filter]) AND 1:10000[slen]"
            )
            handle = Entrez.esearch(
                db="nucleotide", term=search_term, retmax="10"
            )
            record = Entrez.read(handle)
            handle.close()
            
            if record["IdList"]:
                uid = record["IdList"][0]
                handle = Entrez.efetch(db="nucleotide", id=uid, rettype="fasta", retmode="text")
                fasta_record = handle.read()
                handle.close()
                
                seq_obj = SeqIO.read(io.StringIO(fasta_record), "fasta")
                gene_selection[gene_name] = str(seq_obj.seq)
                print(f"  ✅ Found and added sequence for {gene_name} (Length: {len(str(seq_obj.seq))}bp)")
            else:
                print(f"  ❌ No sequence found for {gene_name}")
        except Exception as e:
            print(f"  ❌ Error fetching {gene_name}: {e}")
    
    print(f"\n✅ Gene fetching complete!")
    print(f"Genes successfully loaded: {list(gene_selection.keys())}")
    
    return gene_selection


def sort_genes_by_length(gene_selection):
    """
    Sort genes by sequence length (ascending).
    
    Args:
        gene_selection (dict): Dictionary of {gene_name: sequence}
        
    Returns:
        dict: Sorted dictionary
    """
    sorted_items = sorted(gene_selection.items(), key=lambda item: len(item[1]))
    return {gene: sequence for gene, sequence in sorted_items}


def save_strategy_sequences(gene_name, model_label, strategy_key, sequences, output_dir):
    """
    Save generated sequences for a specific strategy.
    
    Args:
        gene_name (str): Gene name
        model_label (str): Model label
        strategy_key (str): Strategy key
        sequences (list): List of sequences
        output_dir (Path): Output directory
    """
    seq_path = output_dir / 'parts' / 'sequences' / gene_name / model_label / f"{strategy_key}.json.gz"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(seq_path, 'wt', encoding='utf-8') as f:
        json.dump(sequences, f)


def save_strategy_embeddings(gene_name, model_label, strategy_key, embeddings, output_dir):
    """
    Save embeddings for a specific strategy.
    
    Args:
        gene_name (str): Gene name
        model_label (str): Model label
        strategy_key (str): Strategy key
        embeddings (list): List of embedding arrays
        output_dir (Path): Output directory
    """
    emb_path = output_dir / 'parts' / 'embeddings' / gene_name / model_label / f"{strategy_key}.pkl.gz"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_and_embed_sequences(
    gene_selection,
    models,
    decoding_strategies,
    iterations=50,
    mask_ratio=0.15,
    save_all_sequences=True,
    save_interval=5,
    output_dir=None,
    store_in_memory=False,
    save_each_strategy=True,
    device="cuda",
    use_notebook_tqdm=True
):
    """
    Generate evolved sequences and extract embeddings.
    
    Args:
        gene_selection (dict): Dictionary of {gene_name: sequence}
        models (dict): Dictionary of {model_label: SequenceEvolver}
        decoding_strategies (dict): Strategy configurations
        iterations (int): Number of evolution steps
        mask_ratio (float): Masking ratio per step
        save_all_sequences (bool): Whether to save all intermediate sequences
        save_interval (int): Save interval when save_all_sequences=False
        output_dir (Path): Output directory for results
        store_in_memory (bool): Whether to keep all data in memory
        save_each_strategy (bool): Whether to save per-strategy files
        device (str): Device for memory cleanup
        use_notebook_tqdm (bool): Use notebook tqdm (set False for terminal scripts)
        
    Returns:
        tuple: (all_generated_sequences, all_gene_embeddings)
    """
    if output_dir is None:
        output_dir = Path('output')
    else:
        output_dir = Path(output_dir)
    
    all_gene_embeddings = {}
    all_generated_sequences = {}
    
    # Choose tqdm based on environment
    tqdm_func = tqdm if use_notebook_tqdm else tqdm_std
    
    # Main generation loop
    for gene_name, original_sequence in tqdm_func(
        gene_selection.items(), desc="Processing Genes"
    ):
        if store_in_memory:
            all_gene_embeddings[gene_name] = {}
            all_generated_sequences[gene_name] = {}
        
        for model_label, model_instance in tqdm_func(
            models.items(), desc=f"  Model for {gene_name}", leave=False
        ):
            if store_in_memory:
                all_gene_embeddings[gene_name][model_label] = {}
                all_generated_sequences[gene_name][model_label] = {}
            
            for strategy_base_key, strategy_cfg in decoding_strategies.items():
                strategy_type = strategy_cfg["type"]
                temperatures = strategy_cfg.get("temperatures", [1.0])
                top_k = strategy_cfg.get("top_k", 50)
                
                for temp in temperatures:
                    # Create strategy key
                    if strategy_type == "greedy":
                        strategy_key = strategy_base_key
                    else:
                        strategy_key = f"{strategy_base_key}_t{temp}"
                    
                    # Generate sequences
                    generated_sequences = model_instance.run(
                        sequence=original_sequence,
                        steps=iterations,
                        mask_ratio=mask_ratio,
                        strategy=strategy_type,
                        temperature=temp,
                        top_k=top_k,
                        save_all=save_all_sequences,
                        save_interval=save_interval
                    )
                    
                    # Extract embeddings
                    embeddings = []
                    for idx, seq in enumerate(generated_sequences):
                        embedding = model_instance.get_embedding(seq)
                        embeddings.append(embedding)
                        
                        # Periodic memory cleanup
                        if (idx + 1) % 10 == 0:
                            gc.collect()
                            if device == "cuda":
                                import torch
                                torch.cuda.empty_cache()
                            elif device == "mps":
                                import torch
                                torch.mps.empty_cache()
                    
                    # Store in memory if requested
                    if store_in_memory:
                        all_generated_sequences[gene_name][model_label][strategy_key] = generated_sequences
                        all_gene_embeddings[gene_name][model_label][strategy_key] = embeddings
                    
                    # Save to disk
                    if save_each_strategy:
                        save_strategy_sequences(
                            gene_name, model_label, strategy_key, generated_sequences, output_dir
                        )
                        save_strategy_embeddings(
                            gene_name, model_label, strategy_key, embeddings, output_dir
                        )
                    
                    # Memory cleanup
                    del generated_sequences, embeddings
                    gc.collect()
                    if device == "cuda":
                        import torch
                        torch.cuda.empty_cache()
                    elif device == "mps":
                        import torch
                        torch.mps.empty_cache()
        
        # Memory cleanup after each gene
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        elif device == "mps":
            import torch
            torch.mps.empty_cache()
    
    print("\n✅ All sequence generation and embedding complete!")
    
    return all_generated_sequences, all_gene_embeddings


def save_sequences_compressed(data, filepath, compression='gzip'):
    """
    Save sequences with compression.
    
    Args:
        data (dict): Data to save
        filepath (Path or str): Output file path
        compression (str): Compression method ('gzip', 'json', 'pickle', 'pickle_gzip')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compression == 'gzip':
        with gzip.open(filepath.with_suffix('.json.gz'), 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {filepath.with_suffix('.json.gz')} (compressed)")
    elif compression == 'json':
        with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {filepath.with_suffix('.json')}")
    elif compression == 'pickle':
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved to {filepath.with_suffix('.pkl')}")
    elif compression == 'pickle_gzip':
        with gzip.open(filepath.with_suffix('.pkl.gz'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved to {filepath.with_suffix('.pkl.gz')} (compressed)")


def load_sequences_compressed(filepath):
    """
    Load sequences from compressed files.
    
    Args:
        filepath (Path or str): File path
        
    Returns:
        dict or list: Loaded data
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.gz':
        if len(filepath.suffixes) > 1 and filepath.suffixes[-2] == '.json':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        elif len(filepath.suffixes) > 1 and filepath.suffixes[-2] == '.pkl':
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    raise ValueError(f"Unsupported file format: {filepath}")


def load_from_parts(base_dir):
    """
    Load data from parts directory structure.
    
    Args:
        base_dir (Path or str): Base directory
        
    Returns:
        dict: Loaded data
    """
    data = {}
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return data
    
    for path in base_dir.rglob('*'):
        if not path.is_file():
            continue
        
        rel = path.relative_to(base_dir)
        if len(rel.parts) < 3:
            continue
        
        gene_name, model_label = rel.parts[0], rel.parts[1]
        filename = rel.name
        
        if filename.endswith('.json.gz'):
            strategy_key = filename[:-8]
        elif filename.endswith('.pkl.gz'):
            strategy_key = filename[:-7]
        else:
            continue
        
        data.setdefault(gene_name, {}).setdefault(model_label, {})[strategy_key] = (
            load_sequences_compressed(path)
        )
    
    return data


if __name__ == "__main__":
    # Example usage
    from preparation import get_device, load_models
    
    device = get_device()
    models = load_models(device)
    
    gene_selection = fetch_gene_sequences("your_email@example.com")
    gene_selection = sort_genes_by_length(gene_selection)
    
    all_seqs, all_embs = generate_and_embed_sequences(
        gene_selection,
        models,
        DEFAULT_DECODING_STRATEGIES,
        iterations=50,
        output_dir=Path('output'),
        store_in_memory=False
    )
