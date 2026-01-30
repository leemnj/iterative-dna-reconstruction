"""Preparation utilities for gene selection and sequence fetching."""

import warnings
import io
from Bio import Entrez, SeqIO
import requests

# Gene collection configurations
DEFAULT_GENES = {
    "PTEN": {"id": "ENSG00000171862", "type": "Coding", "status": "Real", "sequence": None},
    "PTENP1": {"id": "ENSG00000237938", "type": "Noncoding", "status": "Pseudo", "sequence": None},
    "GAPDH": {"id": "ENSG00000111640", "type": "Coding", "status": "Real", "sequence": None},
    "GAPDHP1": {"id": "ENSG00000228232", "type": "Noncoding", "status": "Pseudo", "sequence": None},
    "HBB": {"id": "ENSG00000244734", "type": "Coding", "status": "Real", "sequence": None},
    "H19": {"id": "ENSG00000130600", "type": "Noncoding", "status": "Real", "sequence": None},
    "RPS29": {"id": "ENSG00000145592", "type": "Coding", "status": "Real", "sequence": None},
    "GAS5": {"id": "ENSG00000234741", "type": "Noncoding", "status": "Real", "sequence": None},
    "TP53": {"id": "ENSG00000141510", "type": "Coding", "status": "Real", "sequence": None},
    "SNHG1": {"id": "ENSG00000237699", "type": "Noncoding", "status": "Real", "sequence": None},
}

DEFAULT_GENES_TO_SEARCH = []

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


def get_sequence_from_ensembl(ensembl_id, seq_type="genomic"):
    """
    Fetch sequence from Ensembl REST API.

    Args:
        ensembl_id (str): ENSG ID string
        seq_type (str): 'genomic' (intron 포함 전체) or 'cds' (coding sequence only)

    Returns:
        str: DNA sequence
    """
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{ensembl_id}?type={seq_type}"

    headers = {"Content-Type": "text/plain"}
    r = requests.get(server + ext, headers=headers)
    if not r.ok:
        r.raise_for_status()
        return None

    return r.text


def fetch_gene_sequences(email, gene_uids=None, genes_to_search=None):
    """
    Fetch gene sequences from NCBI Entrez.

    Args:
        email (str): Email for NCBI Entrez queries
        gene_uids (dict): {gene_name: UID} or {gene_name: metadata dict with "id"}.
        genes_to_search (list): List of gene names to search.

    Returns:
        dict: Dictionary of {gene_name: sequence}
    """
    if gene_uids is None:
        gene_uids = DEFAULT_GENES
    if genes_to_search is None:
        genes_to_search = DEFAULT_GENES_TO_SEARCH

    warnings.filterwarnings(
        "ignore",
        message="[Entrez.read] WARNING: Empty sequence description",
    )

    Entrez.email = email
    gene_selection = {}

    print("Fetching gene sequences from NCBI...")

    for gene_name, uid_or_meta in gene_uids.items():
        is_meta = isinstance(uid_or_meta, dict)
        uid = uid_or_meta.get("id") if is_meta else uid_or_meta
        print(f"Fetching {gene_name} by UID {uid}...")
        try:
            handle = Entrez.efetch(db="nucleotide", id=uid, rettype="fasta", retmode="text")
            fasta_record = handle.read()
            handle.close()

            seq_obj = SeqIO.read(io.StringIO(fasta_record), "fasta")
            sequence = str(seq_obj.seq)
            gene_selection[gene_name] = sequence
            if is_meta:
                uid_or_meta["sequence"] = sequence
            print(f"  ✅ Found and added sequence for {gene_name} (Length: {len(str(seq_obj.seq))}bp)")
        except Exception as e:
            print(f"  ❌ Error fetching {gene_name} by UID {uid}: {e}")

    for gene_name in genes_to_search:
        print(f"Searching for {gene_name}...")
        try:
            search_term = (
                f"{gene_name}[Gene Name] AND Homo sapiens[Organism] AND "
                f"(mRNA[filter] OR ncRNA[filter]) AND 1:10000[slen]"
            )
            handle = Entrez.esearch(db="nucleotide", term=search_term, retmax="10")
            record = Entrez.read(handle)
            handle.close()

            if record["IdList"]:
                uid = record["IdList"][0]
                handle = Entrez.efetch(db="nucleotide", id=uid, rettype="fasta", retmode="text")
                fasta_record = handle.read()
                handle.close()

                seq_obj = SeqIO.read(io.StringIO(fasta_record), "fasta")
                sequence = str(seq_obj.seq)
                gene_selection[gene_name] = sequence
                meta = gene_uids.get(gene_name)
                if isinstance(meta, dict):
                    meta["sequence"] = sequence
                print(f"  ✅ Found and added sequence for {gene_name} (Length: {len(str(seq_obj.seq))}bp)")
            else:
                print(f"  ❌ No sequence found for {gene_name}")
        except Exception as e:
            print(f"  ❌ Error fetching {gene_name}: {e}")

    print("\n✅ Gene fetching complete!")
    print(f"Genes successfully loaded: {list(gene_selection.keys())}")

    return gene_selection


def sort_genes_by_length(gene_selection):
    """Sort genes by sequence length (ascending)."""
    sorted_items = sorted(gene_selection.items(), key=lambda item: len(item[1]))
    return {gene: sequence for gene, sequence in sorted_items}
