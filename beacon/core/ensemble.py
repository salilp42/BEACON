import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from ..utils.diversity import compute_diversity_metrics

class EnsembleModel(nn.Module):
    """Enhanced deep ensemble with diversity promotion and adaptive pruning."""
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        diversity_weight: float = 0.1,
        prune_threshold: float = 0.1
    ):
        """Initialize ensemble model.
        
        Args:
            models: List of member models
            device: Device to run computations on
            diversity_weight: Weight for diversity loss term
            prune_threshold: Threshold for pruning redundant members
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = device
        self.diversity_weight = diversity_weight
        self.prune_threshold = prune_threshold
        self.to(device)
        
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            return_individual: Whether to return individual predictions
            
        Returns:
            Tuple of (mean predictions, uncertainties)
        """
        individual_preds = []
        
        for model in self.models:
            pred = model(x)
            individual_preds.append(pred.unsqueeze(0))
            
        individual_preds = torch.cat(individual_preds, dim=0)
        mean_pred = individual_preds.mean(dim=0)
        
        # Compute both epistemic and aleatoric uncertainty
        epistemic = individual_preds.var(dim=0)  # Model uncertainty
        if individual_preds.shape[-1] > 1:  # For classification
            aleatoric = mean_pred * (1 - mean_pred)  # Prediction uncertainty
        else:  # For regression
            aleatoric = torch.zeros_like(epistemic)
            
        total_uncertainty = epistemic + aleatoric
        
        if return_individual:
            return mean_pred, total_uncertainty, individual_preds
        return mean_pred, total_uncertainty
    
    def compute_diversity_loss(
        self,
        individual_preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute diversity-promoting loss term.
        
        Args:
            individual_preds: Predictions from individual models
            targets: Ground truth targets
            
        Returns:
            Diversity loss term
        """
        # Negative correlation loss
        pred_centered = individual_preds - individual_preds.mean(dim=0, keepdim=True)
        correlation_matrix = torch.matmul(pred_centered, pred_centered.transpose(0, 1))
        diversity_loss = torch.mean(torch.triu(correlation_matrix, diagonal=1))
        
        return self.diversity_weight * diversity_loss
    
    def prune_redundant_members(self) -> None:
        """Remove ensemble members that provide redundant predictions."""
        with torch.no_grad():
            diversity_scores = compute_diversity_metrics(self.models)
            keep_indices = diversity_scores > self.prune_threshold
            self.models = nn.ModuleList([m for i, m in enumerate(self.models) if keep_indices[i]])
    
    def add_member(self, model: nn.Module) -> None:
        """Add new member to ensemble.
        
        Args:
            model: New model to add
        """
        self.models.append(model.to(self.device))
        
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary of model configuration
        """
        return {
            'num_models': len(self.models),
            'diversity_weight': self.diversity_weight,
            'prune_threshold': self.prune_threshold
        }
    
    def set_config(self, config: Dict) -> None:
        """Set model configuration.
        
        Args:
            config: Dictionary of model configuration
        """
        self.diversity_weight = config['diversity_weight']
        self.prune_threshold = config['prune_threshold']
