#!/usr/bin/env python3
"""Save labels from GRENADE datasets for EXPASS training.

This script extracts labels from any of the 7 GRENADE datasets using GRENADE's preprocessing
and saves them in a format that EXPASS can use directly.

Supported datasets:
1. Toxigen
2. LGBTEn
3. MigrantsEn
4. MultilingualENCorpusFrench
5. MultilingualENCorpusGerman
6. MultilingualENCorpusCypriot
7. MultilingualENCorpusSlovene

Usage:
    python grenade_original/Contrastive_Learning_Approach/save_labels.py [dataset_name]
    
    If no dataset_name is provided, defaults to 'toxigen'
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import load_data

# Dataset name mapping
DATASET_NAMES = {
    'toxigen': 'toxigen',
    'lgbten': 'lgbten',
    'migrantsen': 'migrantsen',
    'french': 'multilingualencorpusfrench',
    'german': 'multilingualencorpusgerman',
    'cypriot': 'multilingualencorpuscypriot',
    'slovene': 'multilingualencorpusslovene',
}

def get_labels_from_dataset(dataset):
    """Extract or generate labels from a GRENADE dataset.
    
    Args:
        dataset: A GRENADE dataset object
        
    Returns:
        numpy array of binary labels
    """
    # Try different label strategies based on dataset type
    
    # Strategy 1: Check if dataset already has labels attribute
    if hasattr(dataset, 'labels') and len(dataset.labels) > 0:
        labels = np.array(dataset.labels)
        print(f"  Using existing labels from dataset")
    
    # Strategy 2: Use target_group column (for Toxigen, LGBTEn, MigrantsEn)
    elif 'target_group' in dataset.data.columns:
        target_groups = dataset.data['target_group'].values
        encoder = LabelEncoder()
        labels = encoder.fit_transform(target_groups)
        print(f"  Generated labels from 'target_group' column")
        print(f"  Unique groups: {np.unique(target_groups)}")
    
    # Strategy 3: Use In-Group column (for multilingual datasets)
    elif 'In-Group' in dataset.data.columns:
        in_groups = dataset.data['In-Group'].values
        encoder = LabelEncoder()
        labels = encoder.fit_transform(in_groups)
        print(f"  Generated labels from 'In-Group' column")
        print(f"  Unique groups: {np.unique(in_groups)[:10]}...")  # Show first 10
    
    # Strategy 4: Use Out-group column as fallback
    elif 'Out-group' in dataset.data.columns:
        out_groups = dataset.data['Out-group'].values
        encoder = LabelEncoder()
        labels = encoder.fit_transform(out_groups)
        print(f"  Generated labels from 'Out-group' column")
        print(f"  Unique groups: {np.unique(out_groups)[:10]}...")  # Show first 10
    
    else:
        raise ValueError("Could not find a suitable column for label generation")
    
    # Make binary if needed
    unique_labels = np.unique(labels)
    if len(unique_labels) > 2:
        print(f"  Converting {len(unique_labels)} classes to binary classification...")
        # Group into binary: first class vs all others
        labels = (labels == 0).astype(int)
    
    return labels

def main():
    # Get dataset name from command line or use default
    dataset_name = sys.argv[1].lower() if len(sys.argv) > 1 else 'toxigen'
    
    # Map common names to full names
    if dataset_name in DATASET_NAMES:
        dataset_name = DATASET_NAMES[dataset_name]
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Loading from GRENADE preprocessing...")
    
    # Load dataset using GRENADE's data_loader
    try:
        dataset = load_data(dataset_name=dataset_name, experiment_nb=1)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print(f"\nAvailable datasets: {list(DATASET_NAMES.keys())}")
        sys.exit(1)
    
    print(f"Found {len(dataset.data)} samples")
    
    # Extract labels
    labels = get_labels_from_dataset(dataset)
    
    # Determine output filename based on dataset
    if dataset_name == 'toxigen':
        output_filename = "Toxigen_labels.npy"
    elif dataset_name == 'lgbten':
        output_filename = "LGBTEn_labels.npy"
    elif dataset_name == 'migrantsen':
        output_filename = "MigrantsEn_labels.npy"
    elif 'french' in dataset_name:
        output_filename = "Multilingual_EN_Corpus_FRENCH_labels.npy"
    elif 'german' in dataset_name:
        output_filename = "Multilingual_EN_Corpus_GERMAN_labels.npy"
    elif 'cypriot' in dataset_name:
        output_filename = "Multilingual_EN_Corpus_CYPRIOT_labels.npy"
    elif 'slovene' in dataset_name:
        output_filename = "Multilingual_EN_Corpus_SLOVENE_labels.npy"
    else:
        output_filename = f"{dataset_name}_labels.npy"
    
    # Save
    output_path = Path(__file__).parent / "datasets" / output_filename
    np.save(output_path, labels)
    
    print(f"\n✓ Saved {len(labels)} labels to {output_path}")
    print(f"  Label distribution: {np.bincount(labels)}")
    print(f"  Class 0: {np.sum(labels == 0)} samples")
    print(f"  Class 1: {np.sum(labels == 1)} samples")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
