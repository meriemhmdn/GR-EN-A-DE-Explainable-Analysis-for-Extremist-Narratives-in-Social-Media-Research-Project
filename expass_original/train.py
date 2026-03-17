#!/usr/bin/env python3

import torch
import numpy as np
from time import time
from pathlib import Path
from typing import NamedTuple, Optional, Any
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import normalize, softmax
from tqdm import tqdm
import csv
import re
import sys

from parser import argument_parser
from datasets import DATASET_LOADERS
from model import Model
# from torch_geometric.nn.models.gnn_explainer import GNNExplainer
from batched_explainer import BatchedGNNExplainer as GNNExplainer
from PGMEx import PGMExplainer
# from intgrad import IntegratedGradExplainer

DEVICE = "cpu"
HERE = Path(__file__).parent
CONVERGENCE_DIR = HERE / "convergence_files"
CONVERGENCE_DIR.mkdir(exist_ok=True)

def get_experiment_config(exp_nb: int, grenade_root: Path) -> dict:
    """Read experiment configuration from GRENADE's experiment_params.csv.
    
    Args:
        exp_nb: Experiment number
        grenade_root: Path to GRENADE root directory
        
    Returns:
        Dictionary with 'dataset', 'label_col', and other parameters
    """
    csv_path = grenade_root / "grenade_original/Contrastive_Learning_Approach/src/experiment_params.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"experiment_params.csv not found at {csv_path}")
    
    # Try utf-8, fallback to utf-16 (GRENADE supports both)
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with open(csv_path, 'r', encoding='utf-16') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    # Find row matching exp_nb
    for row in rows:
        if int(row.get('exp_nb', -1)) == exp_nb:
            return {
                'dataset': row.get('dataset', 'Toxigen'),
                'label_col': row.get('label_col', 'target_group'),
                'text_col': row.get('text_col', 'text'),
                'context_cols': [col.strip() for col in row.get('context_cols', 'In-Group,Out-group').split(',')]
            }
    
    raise ValueError(f"Experiment {exp_nb} not found in experiment_params.csv")

def extract_exp_nb_from_path(embeddings_path: str) -> int:
    """Extract experiment number from GRENADE embeddings filename.
    
    Example: 'embeddings__exp1__ntrials_1.npy' -> 1
    """
    match = re.search(r'exp(\d+)', embeddings_path)
    if match:
        return int(match.group(1))
    return 1  # Default to experiment 1

class PerformanceResults(NamedTuple):
    train_acc: float
    val_acc: float
    test_acc: float
    train_auroc: float
    val_auroc: float
    test_auroc: float
    test_f1_score: float

# region main ---------------------------------

def main(
    grenade_embeddings: str,
    grenade_adjacency: str,
    grenade_labels: str = None,
    grenade_exp_nb: int = None,
    dataset_name: str = "toxigen",
    label_col: str = None,
    arch: str = "gcn",
    explainer: str = "gnn_explainer",
    num_layers: int = 3,
    batch_size: int = 200,
    seed: int = 912,
    epochs:int = 150,
    model_saving_lag: int = 25,
    vanilla_mode: bool = False,
    lr_gnn=0.01,
    explainer_iters=5,
    correct_sampling_percent=0.4,
    explanations_lag=20,
    explanation_topk_thresh=0.3,
    lr_gnnex=0.01,
    explainer_epochs=200,
):
    out_dir = HERE / f"grenade-{arch}"
    out_dir.mkdir(exist_ok = True)
    convergence_file_stem = f"loss-lrgnn_{lr_gnn}-seed_{seed}"
    best_model_path = out_dir / f"{convergence_file_stem}-best.pth"
    
    # Initialize all placeholder variables that are updated in the loop
    preds = None
    use_explanations = False
    best_auroc_val = 0
    oversmoothing = 0
    
    # Load GRENADE data
    from grenade_dataset import GrenadeOutputDataset
    
    # Determine experiment number from embeddings filename if not provided
    exp_nb = grenade_exp_nb if grenade_exp_nb else extract_exp_nb_from_path(grenade_embeddings)
    
    if grenade_labels:
        # User provided pre-saved labels
        labels = np.load(grenade_labels)
        print(f"✓ Loaded {len(labels)} labels from {grenade_labels}")
    else:
        # Load labels from GRENADE dataset using experiment config
        print(f"Loading labels for experiment {exp_nb}...")
        
        try:
            # Get experiment configuration
            exp_config = get_experiment_config(exp_nb, HERE.parent)
            dataset_name = exp_config['dataset']
            label_col = label_col or exp_config['label_col']  # Use CLI arg if provided, otherwise use config
            
            print(f"  Dataset: {dataset_name}")
            print(f"  Label column: {label_col}")
            
            # Load dataset using GRENADE's preprocessing
            sys.path.insert(0, str(HERE.parent / "grenade_original/Contrastive_Learning_Approach/src"))
            
            try:
                from preprocessing import (
                    ToxigenDataset,
                    LGBTEnDataset,
                    MigrantsEnDataset,
                    MultilingualENCorpusGermanDataset,
                    MultilingualENCorpusFrenchDataset,
                    MultilingualENCorpusCypriotDataset,
                    MultilingualENCorpusSloveneDataset,
                )
                
                DATASETS = {
                    "Toxigen": ToxigenDataset,
                    "LGBTEn": LGBTEnDataset,
                    "MigrantsEn": MigrantsEnDataset,
                    "MultilingualENCorpusGerman": MultilingualENCorpusGermanDataset,
                    "MultilingualENCorpusFrench": MultilingualENCorpusFrenchDataset,
                    "MultilingualENCorpusCypriot": MultilingualENCorpusCypriotDataset,
                    "MultilingualENCorpusSlovene": MultilingualENCorpusSloveneDataset,
                }
                
                if dataset_name not in DATASETS:
                    raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS.keys())}")
                
                # Pass full path to CSV file
                csv_path = str(HERE.parent / f"grenade_original/Contrastive_Learning_Approach/datasets/{dataset_name}.csv")
                grenade_dataset = DATASETS[dataset_name](
                    experiment_nb=exp_nb,
                    csv_path=csv_path,
                    skip_embeddings=True  # We already have embeddings
                )
                
                # Extract label column
                if label_col not in grenade_dataset.data.columns:
                    raise ValueError(f"Label column '{label_col}' not found in dataset. Available: {grenade_dataset.data.columns.tolist()}")
                
                label_values = grenade_dataset.data[label_col].values
                
                # Encode labels to integers
                encoder = LabelEncoder()
                labels = encoder.fit_transform(label_values)
                
                print(f"✓ Loaded {len(labels)} labels")
                print(f"  Classes: {len(encoder.classes_)} - {encoder.classes_[:5]}{'...' if len(encoder.classes_) > 5 else ''}")
                print(f"  Distribution: {np.bincount(labels)}")
                
            finally:
                # Clean up sys.path
                grenade_src = str(HERE.parent / "grenade_original/Contrastive_Learning_Approach/src")
                if grenade_src in sys.path:
                    sys.path.remove(grenade_src)
            
        except Exception as e:
            print(f"Warning: Could not load labels from GRENADE: {e}")
            print("Falling back to dummy labels")
            # Fallback: create dummy labels based on embeddings size
            embeddings = np.load(grenade_embeddings)
            labels = np.random.randint(0, 2, size=len(embeddings))
            print(f"Warning: Using {len(labels)} random binary labels (0 or 1)")
            print(f"  This is a FALLBACK and may not match the actual dataset structure!")
            print(f"  For proper training, fix the label loading error above.")
    
    # Create dataset
    dataset_obj = GrenadeOutputDataset(
        embeddings_path=grenade_embeddings,
        adjacency_path=grenade_adjacency,
        labels=labels
    )
    
    # Get the data object
    data = dataset_obj._data_obj
    
    # Detect node classification mode
    use_node_classification = False
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        use_node_classification = True
        num_classes = dataset_obj.num_classes
        print("=" * 80)
        print("Detected node classification mode (GRENADE)")
        print(f"  Total nodes: {data.x.shape[0]}")
        print(f"  Train nodes: {data.train_mask.sum().item()} ({100 * data.train_mask.sum().item() / data.x.shape[0]:.1f}%)")
        print(f"  Val nodes: {data.val_mask.sum().item()} ({100 * data.val_mask.sum().item() / data.x.shape[0]:.1f}%)")
        print(f"  Test nodes: {data.test_mask.sum().item()} ({100 * data.test_mask.sum().item() / data.x.shape[0]:.1f}%)")
        print(f"  Number of classes: {num_classes}")
        print("=" * 80)
    else:
        print("Using graph classification mode (original EXPASS)")
        num_classes = 2
    
    # Create data loaders using the PyG Data object
    # For graph-level tasks, wrap in a list
    data.idx = 0  # Add index for consistency with EXPASS batching
    train_loader = [data]
    val_loader = [data]
    test_loader = [data]
    
    # Get number of features
    n_feat = train_loader[0].x.shape[1]
    model = Model(
        nhid=32,
        nfeat=n_feat,
        nclass=num_classes,
        dropout=0.0,
        num_layers=num_layers,
        gnn_arch=arch,
        node_classification_mode=use_node_classification,
    ).to(DEVICE)
    sample_weights = cal_weights_model(train_loader, use_node_classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_gnn)
    criterion = torch.nn.CrossEntropyLoss(weight=sample_weights)
    explainer = get_explainer(
        explainer=explainer, 
        model=model,
        explainer_epochs=explainer_epochs,
        lr_gnnex=lr_gnnex,
        criterion=criterion,
    )

    # Begin train-test-explanation loop
    for epoch in tqdm(range(epochs)):
        epoch_start_time = time()
        if not vanilla_mode and epoch > explanations_lag:
            use_explanations = True
        avg_loss = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            preds=preds,
            explainer=explainer,
            use_explanations=use_explanations,
            explainer_iters=explainer_iters,
            correct_sampling_percent=correct_sampling_percent,
            explanation_topk_thresh=explanation_topk_thresh,
            use_node_classification=use_node_classification,
        )
        output_train, performance = evaluate_performance(
            train_loader, val_loader, test_loader, model, use_node_classification, num_classes
        )
        preds = output_train
        model_saving_lag = 25 if model_saving_lag is None else model_saving_lag
        if epoch >= model_saving_lag and performance.val_auroc >= best_auroc_val:
            best_auroc_val = performance.val_auroc
            torch.save(
                model.state_dict(), 
                best_model_path
            )
        log_progress(
            epoch, avg_loss, performance, oversmoothing, convergence_file_stem, epoch_start_time
        )
    
    # Note: Oversmoothing calculation removed for GRENADE datasets as it requires dataset_loader

# endregion main

# region Functions ---------------------------------

def guess_n_features(train_loader) -> int:
    # TODO test that this works
    first_batch = train_loader.dataset[0]
    # print(
    #     "num_nodes", first_batch.num_nodes, 
    #     "num_edges", first_batch.num_edges, 
    #     "num_node_features", first_batch.num_node_features, 
    #     "num_edge_features", first_batch.num_edge_features
    # )
    return first_batch.num_node_features

def cal_weights_model(dataset, use_node_classification=False):
    "Calculate weights for weighted cross entropy loss to address data imbalance"
    labels = []
    if use_node_classification:
        # For node classification, only use training nodes
        for data in dataset:
            if hasattr(data, 'train_mask'):
                labels += data.y[data.train_mask].tolist()
            else:
                labels += data.y.tolist()
    else:
        # For graph classification, use all graph labels
        for data in dataset:
            labels += data.y.tolist()
    
    labels_tensor = torch.tensor(labels).squeeze()
    
    # Count occurrences of each class
    num_classes = labels_tensor.max().item() + 1
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    
    # Compute weights: inverse frequency
    n_full = labels_tensor.size(0)
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = n_full / (num_classes * class_counts[i])
        else:
            weights[i] = 0.0
    
    return weights

def get_explainer(
    explainer: str,
    model: Model,
    explainer_epochs: Optional[int] = None,
    lr_gnnex: Optional[float] = None,
    criterion: Optional[Any] = None
):
    if explainer == "gnn_explainer":
        return GNNExplainer(
            model, lr=lr_gnnex, return_type="raw", log=False
        )
        return
    if explainer == "pgmexplainer":
        return PGMExplainer(model=model, graph=None)
    if explainer == "intgradexplainer":
        return IntegratedGradExplainer(model, criterion)
    raise ValueError(
        '`explainer` must be one of: ("gnn_explainer", "pgmexplainer", "intgradexplainer")'
    )

def train(
    model,
    train_loader,
    optimizer,
    criterion,
    preds,
    explainer,
    use_explanations: bool,
    explainer_iters: int=5,
    correct_sampling_percent: float=0.05,
    explanation_topk_thresh: float=0.25,
    use_node_classification: bool=False,
):
    losses = []
    for idx, data in enumerate(train_loader):  
        model.eval()
        # NOTE: Use `scores_edges = weights_graphs[idx.item()]` if you want to
        # use the explanations that were obtained in the previous loop
        input_data = data.x
        scores = get_default_scores(data, explainer)
        if use_explanations and preds is not None:
            scores = []
            # Use the explanations that were obtained in the previous loop
            # Uses predictions for previous epoch from a selected batch through 'idx'
            if use_node_classification:
                # For node classification, use training nodes
                sampled_correct_indices = sample_correct_indices(
                    pred=preds[data.train_mask], 
                    gtruth=data.y[data.train_mask],
                    correct_sampling_percent=correct_sampling_percent
                )
            else:
                # For graph classification
                sampled_correct_indices = sample_correct_indices(
                    pred=preds[idx], 
                    gtruth=data.y,
                    correct_sampling_percent=correct_sampling_percent
                )
            scores = get_explainer_scores(
                data=data,
                model=model,
                explainer=explainer,
                sampled_correct_indices=sampled_correct_indices,
                explainer_iters=explainer_iters,
                use_explanations=use_explanations,
                explanation_topk_thresh=explanation_topk_thresh,
                use_node_classification=use_node_classification,
            )
            if isinstance(explainer, PGMExplainer):
                input_data = scores
                scores = None
            

        model.train() # Change to training mode
        optimizer.zero_grad()  # Clear gradients.
        
        # Get model output
        # For node classification, don't pass batch parameter
        batch_param = None if use_node_classification else data.batch
        out = model(input_data, data.edge_index, scores, batch_param)
        
        # Compute loss based on mode
        if use_node_classification:
            # Only compute loss on training nodes
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        else:
            # Original: loss on all graphs in batch
            loss = criterion(out, data.y)
        
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        losses.append(loss)
    
    if use_node_classification:
        # Average loss per training batch (typically 1 graph)
        avg_loss = sum(l.item() for l in losses) / len(losses)
    else:
        # Original: average loss per graph
        avg_loss = sum(losses) / len(train_loader.dataset)
    
    return avg_loss

def get_default_scores(data, explainer):
    if isinstance(explainer, GNNExplainer):
        return torch.ones(data.edge_index.shape[1])
    if isinstance(explainer, PGMExplainer):
        return None
    if isinstance(explainer, IntegratedGradExplainer):
        return None
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")

def sample_correct_indices(pred, gtruth, correct_sampling_percent: float = 0.5) -> np.ndarray:
    """
    Takes predictions from model, returns the indices of a subset of the
    correct predictions
    """
    cor_idx = np.where(pred.cpu() == gtruth)[0]
    samples = int(correct_sampling_percent * cor_idx.size)
    if samples < 1:
        samples = 1
    if cor_idx.shape[0] == 0:
        return np.array([])
    else:
        sampled_idx = np.random.choice(cor_idx, samples)
        return sampled_idx

def get_explainer_scores(
    data,
    model,
    explainer,
    sampled_correct_indices,
    explainer_iters: int,
    use_explanations: bool, 
    explanation_topk_thresh: float,
    use_node_classification: bool = False,
):
    scores = []
    
    if use_node_classification:
        # For node classification, treat the entire graph as one unit
        # Generate explanations based on training nodes
        if len(sampled_correct_indices) > 0:
            graph_scores = _get_sampled_nodes_or_edge_scores(
                data=data,
                idx=0,  # Single graph (not used when use_node_classification=True)
                model=model,
                explainer=explainer,
                explainer_iters=explainer_iters,
                use_explanations=use_explanations,
                explanation_topk_thresh=explanation_topk_thresh,
                use_node_classification=True,
            )
        else:
            # Default weights if no correct predictions
            graph_scores = _get_remaining_nodes_or_edge_scores(
                data=data, idx=0, explainer=explainer, use_node_classification=True,
            )
        scores = graph_scores
    else:
        # Original graph classification logic
        for i in range(data.num_graphs):
            if i in sampled_correct_indices:
                # Generating explanations for sampled graphs from batch
                graph_scores = _get_sampled_nodes_or_edge_scores(
                    data=data,
                    idx=i,
                    model=model,
                    explainer=explainer,
                    explainer_iters=explainer_iters,
                    use_explanations=use_explanations,
                    explanation_topk_thresh=explanation_topk_thresh,
                    use_node_classification=False,
                )
            else:
                # Default weights for non-sampled graphs
                graph_scores = _get_remaining_nodes_or_edge_scores(
                    data=data, idx=i, explainer=explainer, use_node_classification=False,
                )
            scores.extend(graph_scores)

    if isinstance(explainer, GNNExplainer):
        scores = torch.Tensor(scores)
    if isinstance(explainer, PGMExplainer):
        # changing shape to match data.x nodes
        scores = torch.tensor(scores).view(data.x.shape[0], 1)  
        # applying weights to nodes
        scores = scores * data.x
    return scores
    

def _get_sampled_nodes_or_edge_scores(
    data,
    idx,
    model,
    explainer,
    explainer_iters: int,
    use_explanations: bool, 
    explanation_topk_thresh: float,
    use_node_classification: bool = False,
):
    # For node classification, data is the graph object itself (not a batch)
    # For graph classification, data[idx] selects a specific graph from the batch
    graph_data = data if use_node_classification else data[idx]
    
    if isinstance(explainer, GNNExplainer):
        scores_edges = normalized_explanation_median(
            graph_data, explainer_iters, explainer, use_explanations, explanation_topk_thresh
        )
        scores_edges = scores_edges.detach().cpu().numpy()
        return scores_edges
    if isinstance(explainer, PGMExplainer):
        explainer = PGMExplainer(model, graph_data)
        _, p_values, _ = explainer.explain(
            num_samples=1000,
            percentage=10,
            top_node=3,
            p_threshold=0.05,
            pred_threshold=0.1,
        )
        scores_nodes = [1 - j for j in p_values]  
        scores_nodes = torch.tensor(scores_nodes, dtype=graph_data.x.dtype)
        return scores_nodes
    if isinstance(explainer, IntegratedGradExplainer):
        # For node classification, batch should be None
        batch_param = None if use_node_classification else graph_data.batch
        model_kwargs = {"batch": batch_param, "edge_weight": None}
        exp = explainer.get_explanation_graph(
            edge_index=graph_data.edge_index,
            x=graph_data.x,
            y=graph_data.y,
            forward_kwargs=model_kwargs,
        )
        scores_nodes = exp.node_imp
        scores_nodes = normalize(scores_nodes, dim=0)
        scores_nodes = scores_nodes.detach().cpu()
        return scores_nodes
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")

def _get_remaining_nodes_or_edge_scores(data, idx, explainer, use_node_classification: bool = False):
    # For node classification, data is the graph object itself
    # For graph classification, data[idx] selects a specific graph from the batch
    graph_data = data if use_node_classification else data[idx]
    
    if isinstance(explainer, GNNExplainer):
        remaining_edges = torch.ones_like(graph_data.edge_index[1])
        remaining_edges = remaining_edges.detach().cpu().numpy()
        return remaining_edges
    if isinstance(explainer, PGMExplainer):
        return torch.ones(graph_data.x.shape[0], dtype=graph_data.x.dtype)  
    if isinstance(explainer, IntegratedGradExplainer):
        remaining_nodes = torch.ones(graph_data.x.shape[0])
        remaining_nodes = remaining_nodes.detach().cpu()
        return remaining_nodes
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")

def normalized_explanation_median(
    data,
    iters: int,
    explainer: GNNExplainer,
    use_explanations: bool,
    explanation_topk_thresh: float,
):
    "Finds the normalized median of multiple explanations on the same data point"
    weigths_iters = []
    for it in range(iters):
        _, scores_edges = explainer.explain_graph(
            x = data.x, 
            edge_index = data.edge_index, 
            edge_weight=None, 
            use_explanations=use_explanations
        )
        weigths_iters.append(scores_edges)

    scores_edges = torch.stack(weigths_iters).median(0)[0]
    # Normalise weights
    scores_edges = (scores_edges - scores_edges.min()) / (
        scores_edges.max() - scores_edges.min()
    )
    thresh = scores_edges.topk(int(explanation_topk_thresh * data.edge_index.shape[1]))[0][-1]
    scores_edges = torch.where(scores_edges >= thresh, 1.0, 0.0)
    return scores_edges

def test(loader, model, use_node_classification=False):
    model.eval()
    
    if use_node_classification:
        # Evaluate on masked nodes
        preds_list = []
        labels_list = []
        logits_list = []  # Store raw outputs for AUROC
        
        for data in loader:
            with torch.no_grad():
                # Don't pass batch parameter for node classification
                out = model(data.x, data.edge_index, None, None)
                pred = out.argmax(dim=1)
                preds_list.append(pred)
                labels_list.append(data.y)
                logits_list.append(out)  # Store raw outputs
        
        preds = torch.cat(preds_list)
        labels = torch.cat(labels_list)
        logits = torch.cat(logits_list)
        
        # Return full predictions, labels, and logits for different masks
        return preds, labels, None, logits  # Accuracy computed in evaluate_performance
    else:
        # Original graph classification evaluation
        preds = []
        labels = []
        logits = []
        for data in loader:
            out = model(data.x, data.edge_index, None, data.batch)
            pred = out.argmax(dim=1)
            preds.append(pred)
            labels.append(data.y)
            logits.append(out)
        preds  = torch.cat(preds)
        labels = torch.cat(labels)
        logits = torch.cat(logits)
        accuracy = (preds == labels).float().mean()
        return preds, labels, accuracy, logits

def evaluate_performance(train_loader, val_loader, test_loader, model, use_node_classification=False, num_classes=2):
    output_train, labels_train, train_acc, logits_train = test(train_loader, model, use_node_classification)
    output_val, labels_val, val_acc, logits_val = test(val_loader, model, use_node_classification)
    output_test, labels_test, test_acc, logits_test = test(test_loader, model, use_node_classification)
    
    if use_node_classification:
        # Get the data object to access masks
        data = train_loader[0]
        
        # Compute accuracy for each split using masks
        train_pred = output_train[data.train_mask]
        train_labels = labels_train[data.train_mask]
        train_logits = logits_train[data.train_mask]
        train_acc = (train_pred == train_labels).float().mean()
        
        val_pred = output_val[data.val_mask]
        val_labels = labels_val[data.val_mask]
        val_logits = logits_val[data.val_mask]
        val_acc = (val_pred == val_labels).float().mean()
        
        test_pred = output_test[data.test_mask]
        test_labels = labels_test[data.test_mask]
        test_logits = logits_test[data.test_mask]
        test_acc = (test_pred == test_labels).float().mean()
        
        # For AUROC, use softmax probabilities
        if num_classes == 2:
            # Binary classification: use probabilities from logits
            train_probs = softmax(train_logits, dim=1)[:, 1]
            val_probs = softmax(val_logits, dim=1)[:, 1]
            test_probs = softmax(test_logits, dim=1)[:, 1]
            
            train_auroc = roc_auc_score(train_labels.cpu(), train_probs.cpu())
            val_auroc   = roc_auc_score(val_labels.cpu(), val_probs.cpu())
            test_auroc  = roc_auc_score(test_labels.cpu(), test_probs.cpu())
            test_f1_score = f1_score(test_labels.cpu(), test_pred.cpu())
        else:
            # Multi-class: use softmax probabilities for each class
            train_probs = softmax(train_logits, dim=1)
            val_probs = softmax(val_logits, dim=1)
            test_probs = softmax(test_logits, dim=1)
            
            try:
                train_auroc = roc_auc_score(train_labels.cpu(), train_probs.cpu().numpy(), multi_class='ovr', average='macro')
                val_auroc   = roc_auc_score(val_labels.cpu(), val_probs.cpu().numpy(), multi_class='ovr', average='macro')
                test_auroc  = roc_auc_score(test_labels.cpu(), test_probs.cpu().numpy(), multi_class='ovr', average='macro')
                test_f1_score = f1_score(test_labels.cpu(), test_pred.cpu(), average='macro')
            except ValueError as e:
                # Fallback if AUROC fails (e.g., missing classes in split)
                print(f"Warning: AUROC calculation failed ({e}), using accuracy as fallback")
                train_auroc = train_acc.item()
                val_auroc = val_acc.item()
                test_auroc = test_acc.item()
                test_f1_score = f1_score(test_labels.cpu(), test_pred.cpu(), average='macro')
        
        # Return predictions for all nodes for next epoch
        output_for_next_epoch = output_train
    else:
        # Original graph classification - use logits for AUROC
        # logits_train.shape should be [n_samples, num_classes]
        if num_classes == 2:
            # Binary: use probability of positive class
            train_probs = softmax(logits_train, dim=1)[:, 1]
            val_probs = softmax(logits_val, dim=1)[:, 1]
            test_probs = softmax(logits_test, dim=1)[:, 1]
            
            train_auroc = roc_auc_score(labels_train.cpu(), train_probs.cpu())
            val_auroc   = roc_auc_score(labels_val.cpu(), val_probs.cpu())
            test_auroc  = roc_auc_score(labels_test.cpu(), test_probs.cpu())
        else:
            # Multi-class
            train_probs = softmax(logits_train, dim=1)
            val_probs = softmax(logits_val, dim=1)
            test_probs = softmax(logits_test, dim=1)
            
            train_auroc = roc_auc_score(labels_train.cpu(), train_probs.cpu().numpy(), multi_class='ovr', average='macro')
            val_auroc   = roc_auc_score(labels_val.cpu(), val_probs.cpu().numpy(), multi_class='ovr', average='macro')
            test_auroc  = roc_auc_score(labels_test.cpu(), test_probs.cpu().numpy(), multi_class='ovr', average='macro')
        
        test_f1_score = f1_score(labels_test.cpu(), output_test.cpu())
        output_for_next_epoch = output_train
    
    performance = PerformanceResults(
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        train_auroc=train_auroc,
        val_auroc=val_auroc,
        test_auroc=test_auroc,
        test_f1_score=test_f1_score,
    )
    return output_for_next_epoch, performance

def log_progress(
    epoch: int,
    avg_loss: float,
    performance: PerformanceResults,
    oversmoothing: float,
    convergence_file_stem: str,
    epoch_start_time: float,
):
    metrics = {
        "Epoch": epoch,
        "Train Loss": avg_loss,
        "Train Acc": performance.train_acc,
        "Test Acc": performance.test_acc,
        "Train AUROC": performance.train_auroc,
        "Val AUROC": performance.val_auroc,
        "Test AUROC": performance.test_auroc,
        "Test F1": performance.test_f1_score,
        "Val Acc": performance.val_acc,
        "Oversmoothing": oversmoothing,
    }
    metrics_formatted = [
        f"{metric_name}: {metric_value:.4f}"
        for metric_name, metric_value in metrics.items()
    ]
    progress_string = ", ".join(metrics_formatted)
    if epoch % 25 == 0:
        print(progress_string)
    with open(CONVERGENCE_DIR / f"{convergence_file_stem}.csv", "a") as f:
        f.write(progress_string + "\n")
    # print(f"Elapsed: {time() - epoch_start_time:.3f}s")

def calculate_oversmoothing(model, dataset_loader, seed, batch_size, best_model_path):
    graph_embedding = torch.Tensor()
    graph_label     = torch.Tensor()
    model.eval()
    dataset = dataset_loader(
        seed=seed, batch_size=batch_size, split_train_val_test=False
    )
    model.load_state_dict(torch.load(best_model_path))
    for data in dataset:
        embedding = model.embed(data.x, data.edge_index, None, data.batch)
        graph_embedding = torch.cat((graph_embedding, embedding))
        graph_label = torch.cat((graph_label, data.y))
    oversmoothing = calculate_gdr(graph_label, graph_embedding)
    return oversmoothing

def calculate_gdr(label, embedding):
    X_labels = []
    for i in label.unique():
        X_label = embedding[label == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.0] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    dis_intra = 0.0
    for i in label.unique():
        x2 = np.sum(np.square(X_labels[int(i)]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[int(i)], X_labels[int(i)].T)
        dis_intra += np.mean(dists)
    dis_intra /= label.unique().shape[0]

    dis_inter = 0.0
    for i in range(label.unique().shape[0] - 1):
        for j in range(i + 1, label.unique().shape[0]):
            x2_i = np.sum(np.square(X_labels[int(i)]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[int(j)]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(label.unique().shape[0] * (label.unique().shape[0] - 1) / 2)
    dis_inter /= num_inter

    return dis_inter / dis_intra

# endregion

if __name__ == "__main__":
    args = argument_parser.parse_known_args()[0]
    # print(args)
    main(**vars(args))
