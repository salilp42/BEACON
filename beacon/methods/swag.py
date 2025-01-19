import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
import math
from ..core.calibration import ModelCalibrator

class SWAG:
    """Stochastic Weight Averaging Gaussian with efficient low-rank approximation."""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_samples: int = 30,
        max_rank: int = 20,
        swa_start: float = 0.75,
        swa_lr: float = 1e-2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize SWAG model.
        
        Args:
            base_model: Base PyTorch model
            num_samples: Number of samples for prediction
            max_rank: Maximum rank for covariance matrix
            swa_start: When to start SWA (fraction of training)
            swa_lr: Learning rate for SWA
            device: Device to run computations on
        """
        self.base_model = base_model.to(device)
        self.num_samples = num_samples
        self.max_rank = max_rank
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.device = device
        
        self.calibrator = ModelCalibrator(device=device)
        
        # Initialize SWA parameters
        self.swa_model = copy.deepcopy(base_model)
        self.swa_n = 0
        
        # Initialize deviation storage
        self.sq_mean = {}
        self.deviations = {}
        
        for name, param in self.base_model.named_parameters():
            self.sq_mean[name] = torch.zeros_like(param)
            self.deviations[name] = []
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> Dict:
        """Train SWAG model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            
        Returns:
            Dictionary of training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.base_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
        
        history = {'train_loss': [], 'val_loss': []}
        swa_start_epoch = int(epochs * self.swa_start)
        
        for epoch in range(epochs):
            # Training
            self.base_model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.base_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # SWA updates
            if epoch >= swa_start_epoch:
                self._update_swa(epoch - swa_start_epoch)
                
            scheduler.step()
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
        
        # Final calibration
        if val_loader is not None:
            self.calibrate(val_loader)
            
        return history
    
    def _update_swa(self, t: int) -> None:
        """Update SWA statistics.
        
        Args:
            t: Current SWA iteration
        """
        self.swa_n += 1
        momentum = 1.0 / (self.swa_n + 1)
        
        for name, param in self.base_model.named_parameters():
            swa_param = self.swa_model.state_dict()[name]
            param_data = param.data
            
            # Update mean
            swa_param.mul_(1.0 - momentum).add_(param_data, alpha=momentum)
            
            # Update squared mean
            self.sq_mean[name].mul_(1.0 - momentum).add_(
                param_data ** 2,
                alpha=momentum
            )
            
            # Update deviations
            if len(self.deviations[name]) < self.max_rank:
                self.deviations[name].append(param_data - swa_param)
            else:
                self.deviations[name][t % self.max_rank] = param_data - swa_param
    
    def _sample(self) -> None:
        """Sample model parameters from SWAG approximate posterior."""
        scale = 1.0 / math.sqrt(2.0)
        
        for name, param in self.swa_model.named_parameters():
            mean = param.data
            sq_mean = self.sq_mean[name]
            eps = torch.randn_like(mean)
            
            # Diagonal variance
            var = torch.clamp(sq_mean - mean ** 2, 1e-30)
            scaled_diag = scale * torch.sqrt(var) * eps
            
            # Low rank component
            if len(self.deviations[name]) > 0:
                num_dev = len(self.deviations[name])
                eps = torch.randn(num_dev, device=self.device)
                deviation = torch.stack(self.deviations[name])
                scaled_low_rank = (scale / math.sqrt(2 * (num_dev - 1))) * (deviation.t() @ eps)
            else:
                scaled_low_rank = 0
                
            param.data = mean + scaled_diag + scaled_low_rank
    
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
        all_predictions = []
        
        for _ in range(self.num_samples):
            self._sample()
            predictions = []
            
            self.swa_model.eval()
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    pred = self.swa_model(data)
                    
                    if self.calibrator.calibrated:
                        pred = self.calibrator.calibrate_predictions(pred)
                        
                    predictions.append(pred)
                    
            predictions = torch.cat(predictions)
            all_predictions.append(predictions.unsqueeze(0))
            
        all_predictions = torch.cat(all_predictions)
        mean_pred = all_predictions.mean(dim=0)
        uncertainty = all_predictions.var(dim=0)
        
        if return_individual:
            return mean_pred, uncertainty, all_predictions
        return mean_pred, uncertainty
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluate model performance.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Average loss
        """
        self.swa_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.swa_model(data)
                loss = criterion(pred, target)
                total_loss += loss.item()
                
        return total_loss / len(data_loader)
    
    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        method: str = 'temperature'
    ) -> None:
        """Calibrate model predictions.
        
        Args:
            val_loader: Validation data loader
            method: Calibration method
        """
        self.swa_model.eval()
        logits = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                pred = self.swa_model(data)
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
            'swa_model': self.swa_model.state_dict(),
            'sq_mean': self.sq_mean,
            'deviations': self.deviations,
            'swa_n': self.swa_n,
            'calibrator': self.calibrator.get_config()
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load model state.
        
        Args:
            state_dict: Dictionary of model state
        """
        self.swa_model.load_state_dict(state_dict['swa_model'])
        self.sq_mean = state_dict['sq_mean']
        self.deviations = state_dict['deviations']
        self.swa_n = state_dict['swa_n']
        self.calibrator.set_config(state_dict['calibrator'])
        
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary of model configuration
        """
        return {
            'num_samples': self.num_samples,
            'max_rank': self.max_rank,
            'swa_start': self.swa_start,
            'swa_lr': self.swa_lr,
            'calibrator_config': self.calibrator.get_config()
        }
    
    def set_config(self, config: Dict) -> None:
        """Set model configuration.
        
        Args:
            config: Dictionary of model configuration
        """
        self.num_samples = config['num_samples']
        self.max_rank = config['max_rank']
        self.swa_start = config['swa_start']
        self.swa_lr = config['swa_lr']
        self.calibrator.set_config(config['calibrator_config'])
