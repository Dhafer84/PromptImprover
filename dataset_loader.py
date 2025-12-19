from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import pandas as pd
from datasets import load_dataset


@dataclass
class DatasetConfig:
    dataset_name: str = "nomic-ai/gpt4all_prompt_generations_with_p3"
    split: str = "train"
    sample_size: int = 1500  # MVP: 500-5000 est raisonnable
    seed: int = 42


def load_gpt4all_sample(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Loads a *sample* of the dataset (NOT the full train split).
    This avoids RAM issues on Streamlit Cloud and typical laptops.

    Returns a DataFrame with columns: prompt, response, source (if present).
    """
    # Hugging Face datasets slicing: "train[:1500]"
    split_slice = f"{cfg.split}[:{cfg.sample_size}]"
    ds = load_dataset(cfg.dataset_name, split=split_slice)

    df = pd.DataFrame(ds)

    # normalize expected columns
    # dataset typically has: prompt, response, source
    keep_cols = []
    for c in ["prompt", "response", "source"]:
        if c in df.columns:
            keep_cols.append(c)

    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError("Expected columns 'prompt' and 'response' not found in the dataset sample.")

    df = df[keep_cols] if keep_cols else df
    df["prompt"] = df["prompt"].astype(str)
    df["response"] = df["response"].astype(str)

    if "source" not in df.columns:
        df["source"] = ""

    return df
