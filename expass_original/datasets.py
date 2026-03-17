#!/usr/bin/env python3
"""
Dataset loaders for GRENADE-EXPASS Pipeline.

This module provides dataset loading functionality specifically for GRENADE outputs.
Old EXPASS dataset loaders (MUTAG, alkane, etc.) have been removed as this pipeline
focuses on integrating GRENADE's contrastive learning outputs with EXPASS explainability.
"""

from pathlib import Path
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_grenade(embeddings_path: str, adjacency_path: str, labels, 
                 train_mask=None, val_mask=None, test_mask=None, batch_size: int = 32):
    """
    Load GRENADE outputs for EXPASS.
    
    This function integrates GRENADE's contrastive learning outputs with EXPASS.
    It creates a single-graph dataset from GRENADE's learned embeddings and adjacency.
    
    Supports all 7 GRENADE datasets:
    1. Toxigen
    2. LGBTEn
    3. MigrantsEn
    4. MultilingualENCorpusFrench
    5. MultilingualENCorpusGerman
    6. MultilingualENCorpusCypriot
    7. MultilingualENCorpusSlovene
    
    Args:
        embeddings_path: Path to GRENADE embeddings (.npy)
        adjacency_path: Path to GRENADE adjacency matrix (.pkl)
        labels: Node labels array
        train_mask: Training mask (optional, will be generated if None)
        val_mask: Validation mask (optional)
        test_mask: Test mask (optional)
        batch_size: Batch size (not used for single graph, kept for API consistency)
    
    Returns:
        train_loader, val_loader, test_loader (all return the same single graph)
    """
    from grenade_dataset import GrenadeOutputDataset
    
    dataset = GrenadeOutputDataset(
        embeddings_path=embeddings_path,
        adjacency_path=adjacency_path,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # For single graph datasets, we return the same data for all loaders
    # The train/val/test split is handled via masks in the Data object
    data = dataset.get(0)
    data.idx = 0  # Add index for consistency with EXPASS batching
    
    # Wrap in DataLoader for API consistency
    train_loader = [data]
    val_loader = [data]
    test_loader = [data]
    
    return train_loader, val_loader, test_loader


DATASET_LOADERS = {
    "grenade": load_grenade,
}
