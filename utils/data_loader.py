# data_loader.py

import pandas as pd
from pathlib import Path

def find_repo_root(marker="README.md"):
    """Walk up from current path to find the repo root containing `marker`."""
    current = Path().resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find repo root with marker '{marker}'")

def load_parquet(columns=None, marker="README.md"):
    """
    Load the GFOC dataset from the repo.
    
    Parameters
    ----------
    columns : list or None
        List of columns to load. If None, loads all.
    marker : str
        File to identify the repo root.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with requested columns and 'time' as datetime.
    """
    repo_root = find_repo_root(marker)
    path = repo_root / Path("Dataset/Dataset_MSc/GFOC_RDCDFI.parquet")

    try:
        df = pd.read_parquet(path, columns=columns)
        df = df.reset_index()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        print(f"✅ Successfully loaded data with shape: {df.shape}")
        return df
    except MemoryError:
        raise MemoryError("Unable to load dataset — try chunked reading instead.")

def load_csv(columns=None, marker="README.md"):
    """
    Load the GFOC dataset from the repo.
    
    Parameters
    ----------
    columns : list or None
        List of columns to load. If None, loads all.
    marker : str
        File to identify the repo root.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with requested columns and 'time' as datetime.
    """
    repo_root = find_repo_root(marker)
    path = repo_root / Path("Dataset/Dataset_MSc/GFOC_RDCDFI.csv")

    try:
        df = pd.read_csv(path, usecols=columns)
        df = df.reset_index()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        print(f"✅ Successfully loaded data with shape: {df.shape}")
        return df
    except MemoryError:
        print("⚠️ MemoryError: Falling back to chunked loading...")
        chunks = []
        for chunk in pd.read_csv(path, usecols=columns, chunksize=50000, low_memory=True):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        print(f"✅ Loaded CSV in chunks with shape: {df.shape}")
        return df

def load_flags(marker="README.md", **read_csv_kwargs):
    """
    Load the three flag CSV files and return them as (Shock, Helio, RC).

    Parameters
    ----------
    marker : str
        File used to locate the repo root (default "README.md").
    **read_csv_kwargs :
        Optional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    Shock, Helio, RC : pd.DataFrame
    """
    repo_root = find_repo_root(marker)

    # Path to the CSV files for shocks, helio, and RC
    Shock_path = repo_root / Path('Dataset/Dataset_IPshocks/shocks_GFOC.csv')
    Helio_path  = repo_root / Path('Dataset/Dataset_ICMECAT/helio4cast_icmecat_GFOC.csv')
    RC_path     = repo_root / Path('Dataset/Dataset_ICMECAT/RC_icmecat_GFOC.csv')

    Shock = pd.read_csv(Shock_path, **read_csv_kwargs)
    Helio = pd.read_csv(Helio_path,  **read_csv_kwargs)
    RC    = pd.read_csv(RC_path,     **read_csv_kwargs)

    return Shock, Helio, RC