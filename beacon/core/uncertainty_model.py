import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from ..utils.diagnostics import UncertaintyMetrics
from ..methods.deep_ensemble import DeepEnsemble
from ..methods.swag import SWAG
from ..methods.bbp import BayesByBackprop

class UncertaintyModel:
    """Main interface for uncertainty quantification in neural networks."""
    
    def __init__(
        self,
        base_model: nn.Module,
        method: str = 'deep_ensemble',
        num_models: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize uncertainty quantification model.
        
        Args:
            base_model: Base PyTorch model to wrap
            method: Uncertainty method ['deep_ensemble', 'swag', 'bbp']
            num_models: Number of ensemble members or samples
            device: Device to run computations on
            **kwargs: Additional method-specific parameters
        """
        self.device = device
        self.base_model = base_model.to(device)
        self.method = method.lower()
        
        if self.method == 'deep_ensemble':
            self.model = DeepEnsemble(
                base_model=base_model,
                num_models=num_models,
                device=device,
                **kwargs
            )
        elif self.method == 'swag':
            self.model = SWAG(
                base_model=base_model,
                num_samples=num_models,
                device=device,
                **kwargs
            )
        elif self.method == 'bbp':
            self.model = BayesByBackprop(
                base_model=base_model,
                num_samples=num_models,
                device=device,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.metrics = UncertaintyMetrics()
        
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> Dict:
        """Train the uncertainty model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        return self.model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def predict(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_individual: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with uncertainty estimates.
        
        Args:
            data_loader: Test data loader
            return_individual: Whether to return individual model predictions
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        return self.model.predict(
            data_loader=data_loader,
            return_individual=return_individual
        )
    
    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        method: str = 'temperature'
    ) -> None:
        """Calibrate the uncertainty estimates.
        
        Args:
            val_loader: Validation data for calibration
            method: Calibration method ['temperature', 'isotonic']
        """
        self.model.calibrate(val_loader=val_loader, method=method)
    
    def evaluate_uncertainty(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate uncertainty quality metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of uncertainty metrics
        """
        predictions, uncertainties = self.predict(test_loader)
        targets = torch.cat([y for _, y in test_loader])
        
        return self.metrics.compute_all(
            predictions=predictions,
            uncertainties=uncertainties,
            targets=targets
        )
    
    def save(self, path: str) -> None:
        """Save model state.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state': self.model.state_dict(),
            'method': self.method,
            'config': self.model.get_config()
        }, path)
    
    def load(self, path: str) -> None:
        """Load model state.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.method = checkpoint['method']
        self.model.set_config(checkpoint['config'])
