#!/usr/bin/env python3

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode, ModelTaskLevel

class ModelWrapper(torch.nn.Module):
    """Wrapper to make model compatible with Explainer API"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, edge_index, **kwargs):
        # Add missing parameters that Model expects
        edge_weight = kwargs.get('edge_attr', None)
        batch = kwargs.get('batch', None)
        return self.model(x, edge_index, edge_weight, batch)

class BatchedGNNExplainer:
    """Wrapper for GNNExplainer that works with PyG 2.x API"""
    
    def __init__(self, model, lr=0.01, epochs=100, return_type='raw', log=False, **kwargs):
        self.model = model
        self.return_type = return_type
        self.log = log
        
        # Wrap the model to fix signature mismatch
        wrapped_model = ModelWrapper(model)
        
        # Create the new-style Explainer with GNNExplainer algorithm
        algorithm = GNNExplainer(epochs=epochs, lr=lr)
        
        model_config = ModelConfig(
            mode=ModelMode.multiclass_classification,
            task_level=ModelTaskLevel.node,
            return_type='raw',
        )
        
        self.explainer = Explainer(
            model=wrapped_model,
            algorithm=algorithm,
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
    
    def explain_graph(self, x, edge_index, edge_weight=None, **kwargs):
        """
        Compatibility method for old API.
        Returns: (node_mask, edge_mask)
        """
        # Call the new API
        explanation = self.explainer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            batch=kwargs.get('batch', None),
        )
        
        # Return in old format: (node_mask, edge_mask)
        node_mask = explanation.node_mask if hasattr(explanation, 'node_mask') else None
        edge_mask = explanation.edge_mask if hasattr(explanation, 'edge_mask') else torch.ones(edge_index.size(1))
        
        return node_mask, edge_mask
    
    @torch.no_grad()
    def get_initial_prediction(self, x, edge_index, batch=None, edge_weight=None, **kwargs):
        out = self.model(x, edge_index, edge_weight, batch, **kwargs)
        if self.return_type == "regression":
            prediction = out
        else:
            log_logits = torch.log_softmax(out, dim=-1)
            prediction = log_logits.argmax(dim=-1)
        return prediction
