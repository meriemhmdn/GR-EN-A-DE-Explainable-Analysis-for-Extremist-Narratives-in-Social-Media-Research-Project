#!/usr/bin/env python3
"""
Extract edge importance explanations from trained EXPASS model.
Run this after training completes.
"""

import torch
import numpy as np
from pathlib import Path
from batched_explainer import BatchedGNNExplainer
from model import Model
import pandas as pd

def export_edge_explanations(
    model_path: str,
    embeddings_path: str,
    adjacency_path: str,
    output_path: str = "edge_explanations.csv",
    arch: str = "gcn",
    num_layers: int = 3,
    nhid: int = 64,
    num_classes: int = 13,
):
    """
    Load trained model and generate edge importance scores.

    Args:
        model_path: Path to saved model (.pth file)
        embeddings_path: Path to GRENADE embeddings (.npy)
        adjacency_path: Path to GRENADE adjacency matrix (.pkl)
        output_path: Where to save the edge importance CSV
        arch: Model architecture (gcn, gat, etc.)
        num_layers: Number of GNN layers
        nhid: Hidden feature dimension
        num_classes: Number of output classes
    """
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)

    print(f"Loading adjacency from {adjacency_path}...")
    import pickle
    with open(adjacency_path, 'rb') as f:
        adj_matrix = pickle.load(f)

    # Convert to edge_index format
    edge_index = torch.from_numpy(np.array(adj_matrix.nonzero())).long()
    x = torch.from_numpy(embeddings).float()

    num_features = x.shape[1]

    print(f"Creating model ({arch}, {num_layers} layers, {num_classes} output classes)...")
    model = Model(
        nfeat=num_features,
        nhid=nhid,  # Hidden dimension
        nclass=num_classes,
        dropout=0.0,
        num_layers=num_layers,
        gnn_arch=arch,
        node_classification_mode=True,
    )

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to eval mode for inference

    # Enable gradients for input features (required by GNNExplainer)
    x.requires_grad_(True)

    print("Creating explainer...")
    explainer = BatchedGNNExplainer(
        model=model,
        lr=0.01,
        epochs=100,
        return_type='raw',
        log=False,
    )

    print("Generating edge importance scores (this may take a few minutes)...")
    # Don't use no_grad - explainer needs gradients
    _, edge_mask = explainer.explain_graph(
        x=x,
        edge_index=edge_index,
        edge_weight=None,
    )

    # Create DataFrame with edge information
    print("Creating edge importance table...")
    edges_df = pd.DataFrame({
        'source_node': edge_index[0].numpy(),
        'target_node': edge_index[1].numpy(),
        'importance_score': edge_mask.numpy(),
    })

    # Sort by importance
    edges_df = edges_df.sort_values('importance_score', ascending=False)
    edges_df['rank'] = range(1, len(edges_df) + 1)

    # Save to CSV
    edges_df.to_csv(output_path, index=False)
    print(f"✓ Saved edge explanations to {output_path}")
    print(f"  Total edges: {len(edges_df)}")
    print(f"\nTop 10 most important edges:")
    print(edges_df.head(10).to_string(index=False))

    return edges_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument("--embeddings", required=True, help="Path to GRENADE embeddings .npy")
    parser.add_argument("--adjacency", required=True, help="Path to GRENADE adjacency .pkl")
    parser.add_argument("--output", default="edge_explanations.csv", help="Output CSV file")
    parser.add_argument("--arch", default="gcn", help="Model architecture")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--nhid", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_classes", type=int, default=13, help="Number of output classes. Must match classes in model checkpoint.")
    args = parser.parse_args()

    export_edge_explanations(
        model_path=args.model,
        embeddings_path=args.embeddings,
        adjacency_path=args.adjacency,
        output_path=args.output,
        arch=args.arch,
        num_layers=args.layers,
        nhid=args.nhid,
        num_classes=args.num_classes,
    )
