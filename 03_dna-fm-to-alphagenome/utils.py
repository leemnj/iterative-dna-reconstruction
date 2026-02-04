from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from alphagenome.data import gene_annotation, transcript
from alphagenome.models import dna_client

DEFAULT_GTF_URL = (
    "https://storage.googleapis.com/alphagenome/reference/"
    "gencode/hg38/gencode.v46.annotation.gtf.gz.feather"
)
DEFAULT_GTF_FILENAME = "gencode.v46.annotation.gtf.gz.feather"
DEFAULT_DATA_DIR = Path("../../data")


def get_dna_model(api_key: str | None = None) -> dna_client.DnaClient:
    """Create Alphagenome DNA client using env var if api_key not provided."""
    if api_key is None:
        load_dotenv()
        api_key = _get_env_api_key()
    return dna_client.create(api_key)


def get_output_metadata(
    dna_model: dna_client.DnaClient,
) -> dna_client.OutputMetadata:
    """Return output metadata for human organism."""
    return dna_model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)


def load_gtf_feather(
    data_dir: Path | None = None,
    filename: str = DEFAULT_GTF_FILENAME,
    url: str = DEFAULT_GTF_URL,
) -> pd.DataFrame:
    """Load GTF feather file, downloading if missing."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    local_path = Path(data_dir) / filename
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(url, local_path)
    else: print(f"{filename} already exists!")
    return pd.read_feather(local_path)


def prepare_gtf_views(
    gtf: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return filtered transcript and longest transcript views."""
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ["1"]
    )
    gtf_longest_transcript = gene_annotation.filter_to_longest_transcript(
        gtf_transcript
    )
    return gtf_transcript, gtf_longest_transcript


def build_transcript_extractors(
    gtf_transcript: pd.DataFrame,
    gtf_longest_transcript: pd.DataFrame,
) -> Tuple[transcript.TranscriptExtractor, transcript.TranscriptExtractor]:
    """Build transcript extractors from filtered views."""
    transcript_extractor = transcript.TranscriptExtractor(gtf_transcript)
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gtf_longest_transcript
    )
    return transcript_extractor, longest_transcript_extractor


def _download_file(url: str, local_path: Path) -> None:
    print(f"Downloading to {local_path}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")


def _get_env_api_key() -> str:
    import os

    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ALPHAGENOME_API_KEY is not set. "
            "Set it in the environment or provide api_key."
        )
    return api_key


def get_sequence_from_ensembl_region(
    species: str,
    chrom: str,
    start: int,
    end: int,
    strand: str | int,
    coord_system_version: str | None = None,
) -> str:
    """Fetch raw sequence for half-open interval [start, end) in 1-based coordinates.

    Ensembl REST expects inclusive end, so we convert end -> end-1.
    """
    if end <= start:
        raise ValueError(f"Invalid half-open interval: start={start}, end={end}")

    server = "https://rest.ensembl.org"
    clean_chrom = str(chrom).replace("chr", "")
    strand_val = 1 if str(strand) in ["+", "1"] else -1
    inclusive_end = end - 1
    region = f"{clean_chrom}:{start}..{inclusive_end}:{strand_val}"
    ext = f"/sequence/region/{species}/{region}"
    if coord_system_version:
        ext = f"{ext}?coord_system_version={coord_system_version}"

    headers = {"Content-Type": "text/plain"}
    r = requests.get(server + ext, headers=headers)
    r.raise_for_status()
    return r.text.strip()


def normalize_sequence(seq: str) -> str:
    """Normalize sequence to A/C/G/T/N only."""
    seq = seq.upper()
    normalized = []
    for ch in seq:
        if ch in "ACGT":
            normalized.append(ch)
        else:
            normalized.append("N")
    return "".join(normalized)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement_map = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(complement_map[base] for base in reversed(seq))


def get_sequence(
    gene: str,
    gtf: pd.DataFrame | None = None,
    species: str = "human",
    coord_system_version: str = "GRCh38",
    feature: str = "transcript",
    start: int | None = None,
    end: int | None = None,
) -> tuple[str, dict]:
    """Return normalized sequence and metadata for a gene symbol."""
    if gtf is None:
        gtf = load_gtf_feather()

    chrom_col = "Chromosome" if "Chromosome" in gtf.columns else "chromosome"
    start_col = "Start" if "Start" in gtf.columns else "start"
    end_col = "End" if "End" in gtf.columns else "end"
    strand_col = "Strand" if "Strand" in gtf.columns else "strand"

    gene_rows = gtf[(gtf["gene_name"] == gene) & (gtf["Feature"] == feature)]
    if gene_rows.empty:
        raise KeyError(f"Gene not found in GTF: {gene} (Feature={feature})")
    gene_gtf = gene_rows.iloc[0]

    gene_id = gene_gtf["gene_id"]
    chrom = gene_gtf[chrom_col]
    gtf_start = int(gene_gtf[start_col])
    gtf_end = int(gene_gtf[end_col])
    strand = gene_gtf[strand_col]

    if (start is None) != (end is None):
        raise ValueError("Both start and end must be provided together.")
    if start is None:
        start = gtf_start
        end = gtf_end

    seq_raw = get_sequence_from_ensembl_region(
        species,
        chrom,
        start,
        end,
        strand,
        coord_system_version=coord_system_version,
    )
    seq_normalized = normalize_sequence(seq_raw)

    metadata = {
        "gene": gene,
        "gene_id": gene_id,
        "chrom": chrom,
        "start": start,
        "end": end,
        "strand": strand,
        "feature": feature,
        "species": species,
        "coord_system_version": coord_system_version,
    }
    return seq_normalized, metadata
