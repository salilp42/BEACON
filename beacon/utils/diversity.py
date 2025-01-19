import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

def compute_diversity_metrics(models: List[nn.Module]) -> torch.Tensor:
    """Compute diversity scores for ensemble members.
    
    This implements several diversity measures:
    1. Parameter space diversity
    2. Prediction correlation
    3. Gradient diversity
    
    Args:
        models: List of ensemble models
        
    Returns:
        Tensor of diversity scores for each model
    """
    num_models = len(models)
    if num_models < 2:
        return torch.ones(1)
        
    # Parameter space diversity
    param_vectors = []
    for model in models:
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        param_vectors.append(params)
    param_matrix = torch.stack(param_vectors)
    
    # Compute pairwise cosine similarities
    param_norm = torch.norm(param_matrix, dim=1, keepdim=True)
    param_similarities = torch.mm(param_matrix, param_matrix.t()) / (param_norm * param_norm.t())
    
    # Convert similarities to distances
    param_distances = 1 - param_similarities
    
    # Compute diversity score as mean distance to other models
    diversity_scores = param_distances.sum(dim=1) / (num_models - 1)
    
    return diversity_scores

def compute_gradient_diversity(
    gradients: List[torch.Tensor]
) -> torch.Tensor:
    """Compute gradient diversity measure.
    
    Args:
        gradients: List of gradient tensors
        
    Returns:
        Gradient diversity score
    """
    if len(gradients) < 2:
        return torch.ones(1)
        
    grad_matrix = torch.stack(gradients)
    grad_norm = torch.norm(grad_matrix, dim=1, keepdim=True)
    grad_similarities = torch.mm(grad_matrix, grad_matrix.t()) / (grad_norm * grad_norm.t())
    
    return 1 - grad_similarities.mean()

def compute_prediction_diversity(
    predictions: torch.Tensor
) -> torch.Tensor:
    """Compute prediction-based diversity measure.
    
    Args:
        predictions: Tensor of predictions from different models
        
    Returns:
        Prediction diversity score
    """
    if predictions.shape[0] < 2:
        return torch.ones(1)
        
    pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
    pred_similarities = torch.matmul(pred_centered, pred_centered.transpose(0, 1))
    pred_norm = torch.norm(pred_centered, dim=1, keepdim=True)
    pred_similarities = pred_similarities / (pred_norm * pred_norm.transpose(0, 1))
    
    return 1 - pred_similarities.mean()

def compute_ensemble_stats(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """Compute ensemble statistics.
    
    Args:
        predictions: Tensor of predictions from different models
        targets: Ground truth targets
        
    Returns:
        Dictionary of ensemble statistics
    """
    stats = {}
    
    # Average individual accuracy
    individual_acc = (predictions.argmax(dim=-1) == targets.unsqueeze(0)).float().mean(dim=1)
    stats['avg_individual_acc'] = individual_acc.mean().item()
    
    # Ensemble accuracy
    ensemble_pred = predictions.mean(dim=0).argmax(dim=-1)
    stats['ensemble_acc'] = (ensemble_pred == targets).float().mean().item()
    
    # Diversity measures
    stats['prediction_diversity'] = compute_prediction_diversity(predictions).item()
    
    # Disagreement measure
    pairwise_disagree = (predictions.argmax(dim=-1).unsqueeze(1) != 
                        predictions.argmax(dim=-1).unsqueeze(0)).float().mean()
    stats['disagreement'] = pairwise_disagree.item()
    
    return stats
