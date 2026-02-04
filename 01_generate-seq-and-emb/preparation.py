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


def resolve_torch_dtype(device, torch_dtype):
    """
    Resolve desired torch dtype based on device and preference.
    
    Args:
        device (str): 'cuda', 'mps', or 'cpu'
        torch_dtype (str or torch.dtype): Desired dtype or "auto"
    
    Returns:
        torch.dtype: Resolved dtype
    """
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype is None or torch_dtype == "auto":
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if torch_dtype in dtype_map:
        return dtype_map[torch_dtype]
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


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
    
    def __init__(self, model_path, model_label, device, torch_dtype="auto"):
        """
        Initialize the SequenceEvolver with a pretrained model.
        
        Args:
            model_path (str): Path to the pretrained model or HuggingFace model ID
            model_label (str): Label for the model (e.g., "DNABERT-2")
            device (str): Device to load the model on ('cuda', 'mps', 'cpu')
            torch_dtype (str or torch.dtype): Desired torch dtype for model weights
        """
        self.label = model_label
        self.device = device
        print(f"[{model_label}] Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        resolved_dtype = resolve_torch_dtype(device, torch_dtype)
        try:
            # Prefer new HF argument name to avoid deprecation warnings.
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=resolved_dtype,
            ).to(device)
        except TypeError:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=resolved_dtype,
            ).to(device)
        self.model.eval()
        print(f"[{model_label}] Model loaded successfully.")

        if model_label == "DNABERT-2":
            # Warm up alibi cache to reduce repeated resize warnings.
            max_len = 1024
            dummy_seq = "A" * max_len
            inputs = self.tokenizer(
                dummy_seq,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                _ = self.model(**inputs)
            self._clear_cache()
        # DNABERT-2: alibi 캐시를 1024로 한번에 키우기
        if model_label == "DNABERT-2":
            max_len = 1024
            dummy = "A" * max_len
            inputs = self.tokenizer(
                dummy,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                _ = self.model(**inputs)
            self._clear_cache()
    
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
        
        with torch.inference_mode():
            base_model = self.model.base_model if hasattr(self.model, "base_model") else self.model
            outputs = base_model(**inputs, return_dict=True)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                # Some models ignore return_dict=True and return a tuple.
                hidden_states = outputs[0]
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
        max_length = 1024
        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        window_size = max(1, max_length - num_special_tokens)
        stride = max(1, window_size // 2)

        raw_ids = self.tokenizer(
            current_sequence,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]
        raw_ids = torch.tensor(raw_ids, device=self.device)

        for start in range(0, raw_ids.numel(), stride):
            end = min(start + window_size, raw_ids.numel())
            window_raw = raw_ids[start:end].tolist()
            window_ids = self.tokenizer.build_inputs_with_special_tokens(window_raw)
            input_ids = torch.tensor([window_ids], device=self.device)

            special_mask = self.tokenizer.get_special_tokens_mask(
                window_ids, already_has_special_tokens=True
            )
            candidate_indices = [i for i, m in enumerate(special_mask) if m == 0]
            if not candidate_indices:
                continue

            num_mask = max(1, int(len(candidate_indices) * mask_ratio))
            mask_indices = np.random.choice(candidate_indices, num_mask, replace=False)
            input_ids[0, mask_indices] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            mask_logits = logits[0, mask_indices, :]
            predicted_tokens = self.decode(mask_logits, strategy, temperature, top_k)

            # Map window token positions back to raw indices.
            window_raw_positions = [i for i, m in enumerate(special_mask) if m == 0]
            local_to_raw = {
                pos: start + idx for idx, pos in enumerate(window_raw_positions)
            }
            for local_pos, predicted_id in zip(mask_indices, predicted_tokens):
                raw_ids[local_to_raw[int(local_pos)]] = predicted_id

            del input_ids, outputs, logits, mask_logits, predicted_tokens
            self._clear_cache()

            if end == raw_ids.numel():
                break

        restored_sequence = self.tokenizer.decode(raw_ids, skip_special_tokens=True)
        del raw_ids
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


def load_model(device, model_label, model_path, torch_dtype="auto"):
    """
    Load a single pretrained model.
    
    Args:
        device (str): Device to load model on
        model_label (str): Model label
        model_path (str): Model path/ID
        torch_dtype (str or torch.dtype): Desired torch dtype
    
    Returns:
        SequenceEvolver: Loaded model instance
    """
    # Special handling for DNABERT-2
    if model_label == "DNABERT-2":
        print(f"📥 Downloading {model_label}...")
        local_path = snapshot_download(repo_id=model_path)
        force_patch_triton_config(local_path)
        model_path = local_path
    
    return SequenceEvolver(model_path, model_label, device, torch_dtype=torch_dtype)


def iter_models(device, model_configs, torch_dtype="auto"):
    """
    Iterate over model configs and load one model at a time.
    
    Args:
        device (str): Device to load models on
        model_configs (dict): {label: model_path}
        torch_dtype (str or torch.dtype): Desired torch dtype
    
    Yields:
        tuple: (model_label, SequenceEvolver)
    """
    for label, model_path in model_configs.items():
        try:
            model = load_model(device, label, model_path, torch_dtype=torch_dtype)
            print(f"✅ {label} loaded successfully.")
            yield label, model
        except Exception as e:
            print(f"❌ {label} load failed: {e}")
            continue


def load_models(device, model_configs=None, torch_dtype="auto"):
    """
    Load pretrained models.
    
    Args:
        device (str): Device to load models on
        model_configs (dict): Model configurations. If None, uses default configs.
                            Keys are model labels, values are model paths/IDs.
        torch_dtype (str or torch.dtype): Desired torch dtype
        
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
            models[label] = load_model(device, label, model_path, torch_dtype=torch_dtype)
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
