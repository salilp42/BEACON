import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
import copy
from ..core.ensemble import EnsembleModel
from ..core.calibration import ModelCalibrator

class DeepEnsemble:
    """Enhanced Deep Ensemble with diversity promotion and adaptive pruning."""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_models: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        diversity_weight: float = 0.1,
        prune_threshold: float = 0.1
    ):
        """Initialize deep ensemble.
        
        Args:
            base_model: Base PyTorch model
            num_models: Number of ensemble members
            device: Device to run computations on
            diversity_weight: Weight for diversity loss
            prune_threshold: Threshold for member pruning
        """
        self.device = device
        self.models = [copy.deepcopy(base_model) for _ in range(num_models)]
        self.ensemble = EnsembleModel(
            models=self.models,
            device=device,
            diversity_weight=diversity_weight,
            prune_threshold=prune_threshold
        )
        self.calibrator = ModelCalibrator(device=device)
        
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> Dict:
        """Train ensemble members.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            
        Returns:
            Dictionary of training history
        """
        optimizers = [
            optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for model in self.models
        ]
        
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.ensemble.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass with diversity
                mean_pred, uncertainty, individual_preds = self.ensemble(
                    data,
                    return_individual=True
                )
                
                # Compute losses
                ensemble_loss = criterion(mean_pred, target)
                diversity_loss = self.ensemble.compute_diversity_loss(
                    individual_preds,
                    target
                )
                total_loss = ensemble_loss + diversity_loss
                
                # Backward pass
                for opt in optimizers:
                    opt.zero_grad()
                total_loss.backward()
                for opt in optimizers:
                    opt.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                
                # Adaptive pruning based on validation performance
                if epoch > 0 and epoch % 10 == 0:
                    self.ensemble.prune_redundant_members()
        
        # Final calibration
        if val_loader is not None:
            self.calibrate(val_loader)
            
        return history
    
    def predict(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_individual: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with uncertainty estimates.
        
        Args:
            data_loader: Test data loader
            return_individual: Whether to return individual predictions
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.ensemble.eval()
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                pred, uncertainty = self.ensemble(data)
                
                if self.calibrator.calibrated:
                    pred = self.calibrator.calibrate_predictions(pred)
                    
                predictions.append(pred)
                uncertainties.append(uncertainty)
        
        return (
            torch.cat(predictions),
            torch.cat(uncertainties)
        )
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluate ensemble performance.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Average loss
        """
        self.ensemble.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred, _ = self.ensemble(data)
                loss = criterion(pred, target)
                total_loss += loss.item()
                
        return total_loss / len(data_loader)
    
    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        method: str = 'temperature'
    ) -> None:
        """Calibrate ensemble predictions.
        
        Args:
            val_loader: Validation data loader
            method: Calibration method
        """
        self.ensemble.eval()
        logits = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                pred, _ = self.ensemble(data)
                logits.append(pred)
                targets.append(target)
                
        logits = torch.cat(logits)
        targets = torch.cat(targets).to(self.device)
        
        self.calibrator.calibrate(logits, targets)
    
    def state_dict(self) -> Dict:
        """Get model state.
        
        Returns:
            Dictionary of model state
        """
        return {
            'ensemble': self.ensemble.state_dict(),
            'calibrator': self.calibrator.get_config()
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load model state.
        
        Args:
            state_dict: Dictionary of model state
        """
        self.ensemble.load_state_dict(state_dict['ensemble'])
        self.calibrator.set_config(state_dict['calibrator'])
        
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary of model configuration
        """
        return {
            'num_models': len(self.models),
            'ensemble_config': self.ensemble.get_config(),
            'calibrator_config': self.calibrator.get_config()
        }
    
    def set_config(self, config: Dict) -> None:
        """Set model configuration.
        
        Args:
            config: Dictionary of model configuration
        """
        self.ensemble.set_config(config['ensemble_config'])
        self.calibrator.set_config(config['calibrator_config'])
