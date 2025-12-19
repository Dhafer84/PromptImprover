from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from datasets import load_dataset


@dataclass
class DatasetConfig:
    dataset_name: str = "nomic-ai/gpt4all_prompt_generations_with_p3"
    split: str = "train"
    sample_size: int = 1500


def load_gpt4all_sample(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Charge uniquement un échantillon: train[:N] (évite crash RAM).
    Colonnes attendues: prompt, response, (source optionnel).
    """
    split_slice = f"{cfg.split}[:{cfg.sample_size}]"
    ds = load_dataset(cfg.dataset_name, split=split_slice)

    df = pd.DataFrame(ds)

    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError("Expected columns 'prompt' and 'response' not found in dataset sample.")

    if "source" not in df.columns:
        df["source"] = ""

    df["prompt"] = df["prompt"].astype(str)
    df["response"] = df["response"].astype(str)
    df["source"] = df["source"].astype(str)

    return df[["prompt", "response", "source"]]
