"""
Preparation module for DNA sequence evolution models.
Handles device setup, model patching, and model initialization.
"""

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import snapshot_download


def get_device():
    """
    Detect and return the best available device (CUDA, MPS, or CPU).
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {device}")
    return device


def force_patch_triton_config(model_path):
    """
    Patch DNABERT-2 Triton configuration for MPS compatibility.
    
    Args:
        model_path (str): Path to the model directory
    """
    target_file = os.path.join(model_path, "flash_attn_triton.py")
    if not os.path.exists(target_file):
        print(f"⚠️  Triton config file not found at {target_file}")
        return
    
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Apply patches
    if "trans_b=True" in content:
        content = content.replace("qk += tl.dot(q, k, trans_b=True)", "qk += tl.dot(q, tl.trans(k))")
    content = re.sub(r"'BLOCK_M':\s*\d+", "'BLOCK_M': 32", content)
    content = re.sub(r"'BLOCK_N':\s*\d+", "'BLOCK_N': 32", content)
    content = re.sub(r"num_stages=\d+", "num_stages=2", content)
    content = re.sub(r"num_warps=\d+", "num_warps=4", content)
    
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ DNABERT-2 Triton patch applied successfully.")


class SequenceEvolver:
    """
    DNA sequence evolution model using masked language models.
    Supports DNABERT-2, Nucleotide Transformer, and other HuggingFace models.
    """
    
    def __init__(self, model_path, model_label, device):
        """
        Initialize the SequenceEvolver with a pretrained model.
        
        Args:
            model_path (str): Path to the pretrained model or HuggingFace model ID
            model_label (str): Label for the model (e.g., "DNABERT-2")
            device (str): Device to load the model on ('cuda', 'mps', 'cpu')
        """
        self.label = model_label
        self.device = device
        print(f"[{model_label}] Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()
        print(f"[{model_label}] Model loaded successfully.")
    
    def get_embedding(self, sequence):
        """
        Extract mean pooling embedding for a DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            np.ndarray: Embedding vector (mean pooled)
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            embedding_numpy = mean_embedding.cpu().numpy()
            
            # Memory cleanup
            del outputs, hidden_states, attention_mask, sum_embeddings, sum_mask, mean_embedding
            self._clear_cache()
        
        return embedding_numpy
    
    def _clear_cache(self):
        """Clear GPU/MPS cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
    
    def decode(self, logits, strategy="greedy", temperature=1.0, top_k=50):
        """
        Decode logits using the specified strategy.
        
        Args:
            logits (torch.Tensor): Model output logits
            strategy (str): Decoding strategy ('greedy', 'sampling', 'top_k')
            temperature (float): Sampling temperature
            top_k (int): Number of top tokens to consider
            
        Returns:
            torch.Tensor: Decoded token indices
        """
        if strategy == "greedy":
            return torch.argmax(logits, dim=-1)
        elif strategy == "sampling" or strategy == "top_k":
            logits = logits / temperature
            if strategy == "top_k":
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def evolve_step(self, current_sequence, mask_ratio, strategy, temperature, top_k):
        """
        Evolve a sequence by one step using masked prediction.
        
        Args:
            current_sequence (str): Current DNA sequence
            mask_ratio (float): Ratio of nucleotides to mask (0-1)
            strategy (str): Decoding strategy
            temperature (float): Sampling temperature
            top_k (int): Number of top tokens to consider
            
        Returns:
            str: Evolved sequence
        """
        inputs = self.tokenizer(
            current_sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get special tokens to exclude from masking
        special_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
        ]
        special_tokens = [x for x in special_tokens if x is not None]
        
        # Select random positions to mask
        seq_len = input_ids.shape[1]
        candidate_indices = [
            i for i in range(seq_len)
            if input_ids[0, i].item() not in special_tokens
        ]
        num_mask = max(1, int(len(candidate_indices) * mask_ratio))
        mask_indices = np.random.choice(candidate_indices, num_mask, replace=False)
        
        # Mask selected positions
        input_ids[0, mask_indices] = self.tokenizer.mask_token_id
        
        # Predict masked tokens
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        mask_logits = logits[0, mask_indices, :]
        predicted_tokens = self.decode(mask_logits, strategy, temperature, top_k)
        
        # Replace masked tokens with predictions
        final_ids = input_ids.clone()
        final_ids[0, mask_indices] = predicted_tokens
        
        restored_sequence = self.tokenizer.decode(final_ids[0], skip_special_tokens=True)
        
        # Memory cleanup
        del inputs, input_ids, outputs, logits, mask_logits, predicted_tokens, final_ids
        self._clear_cache()
        
        return restored_sequence.replace(" ", "")
    
    def run(self, sequence, steps, mask_ratio, strategy, temperature, top_k, 
            save_all=True, save_interval=1):
        """
        Run iterative sequence evolution.
        
        Args:
            sequence (str): Original DNA sequence
            steps (int): Number of evolution steps
            mask_ratio (float): Masking ratio per step
            strategy (str): Decoding strategy
            temperature (float): Sampling temperature
            top_k (int): Number of top tokens to consider
            save_all (bool): Whether to save all intermediate sequences
            save_interval (int): Interval for saving when save_all=False
            
        Returns:
            list: List of evolved sequences
        """
        current_seq = sequence
        sequence_history = [current_seq] if save_all or save_interval == 1 else []
        
        for step in range(steps):
            current_seq = self.evolve_step(
                current_seq, mask_ratio, strategy, temperature, top_k
            )
            
            if save_all or (step + 1) % save_interval == 0 or step == steps - 1:
                sequence_history.append(current_seq)
            
            # Periodic memory cleanup
            if (step + 1) % 10 == 0:
                gc.collect()
                self._clear_cache()
        
        return sequence_history


def load_models(device, model_configs=None):
    """
    Load pretrained models.
    
    Args:
        device (str): Device to load models on
        model_configs (dict): Model configurations. If None, uses default configs.
                            Keys are model labels, values are model paths/IDs.
        
    Returns:
        dict: Dictionary of model instances {label: SequenceEvolver}
    """
    if model_configs is None:
        model_configs = {
            "DNABERT-2": "zhihan1996/DNABERT-2-117M",
            "NT-v2-50m": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            "NT-v2-500m": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        }
    
    models = {}
    
    for label, model_path in model_configs.items():
        try:
            # Special handling for DNABERT-2
            if label == "DNABERT-2":
                print(f"📥 Downloading {label}...")
                local_path = snapshot_download(repo_id=model_path)
                force_patch_triton_config(local_path)
                model_path = local_path
            
            models[label] = SequenceEvolver(model_path, label, device)
            print(f"✅ {label} loaded successfully.")
        except Exception as e:
            print(f"❌ {label} load failed: {e}")
    
    if not models:
        raise RuntimeError("No models loaded successfully.")
    
    print(f"\n🚀 {len(models)} model(s) ready!")
    return models


if __name__ == "__main__":
    # Example usage
    device = get_device()
    models = load_models(device)
    print(f"Loaded models: {list(models.keys())}")
