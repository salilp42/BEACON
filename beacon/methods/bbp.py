import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
import math
from ..core.calibration import ModelCalibrator

class BayesianLayer(nn.Module):
    """Bayesian layer with Gaussian weight posteriors."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0
    ):
        """Initialize Bayesian layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_std: Standard deviation of prior
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize prior
        self.prior_std = prior_std
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize layer parameters."""
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        
        return nn.functional.linear(x, weight, bias)
    
    def kl_loss(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior.
        
        Returns:
            KL divergence loss
        """
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        kl_weight = self._kl_normal(
            self.weight_mu,
            weight_std,
            torch.zeros_like(self.weight_mu),
            self.prior_std
        )
        kl_bias = self._kl_normal(
            self.bias_mu,
            bias_std,
            torch.zeros_like(self.bias_mu),
            self.prior_std
        )
        
        return kl_weight + kl_bias
    
    def _kl_normal(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: float
    ) -> torch.Tensor:
        """Compute KL divergence between two normal distributions.
        
        Args:
            mu1: Mean of first distribution
            sigma1: Standard deviation of first distribution
            mu2: Mean of second distribution
            sigma2: Standard deviation of second distribution
            
        Returns:
            KL divergence
        """
        return (torch.log(sigma2) - torch.log(sigma1) + 
                (sigma1.pow(2) + (mu1 - mu2).pow(2)) / (2 * sigma2.pow(2)) - 0.5).sum()

class BayesByBackprop:
    """Bayes by Backprop implementation with efficient variational inference."""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_samples: int = 30,
        prior_std: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize BBP model.
        
        Args:
            base_model: Base PyTorch model
            num_samples: Number of samples for prediction
            prior_std: Standard deviation of prior
            device: Device to run computations on
        """
        self.num_samples = num_samples
        self.prior_std = prior_std
        self.device = device
        
        # Convert base model to Bayesian
        self.model = self._convert_to_bayesian(base_model).to(device)
        self.calibrator = ModelCalibrator(device=device)
    
    def _convert_to_bayesian(self, model: nn.Module) -> nn.Module:
        """Convert standard neural network to Bayesian neural network.
        
        Args:
            model: Standard PyTorch model
            
        Returns:
            Bayesian neural network
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, BayesianLayer(
                    module.in_features,
                    module.out_features,
                    self.prior_std
                ))
            else:
                self._convert_to_bayesian(module)
        return model
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0
    ) -> Dict:
        """Train BBP model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            kl_weight: Weight for KL divergence term
            
        Returns:
            Dictionary of training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with single sample
                output = self.model(data)
                nll_loss = criterion(output, target)
                
                # Add KL divergence
                kl_loss = 0.0
                for module in self.model.modules():
                    if isinstance(module, BayesianLayer):
                        kl_loss += module.kl_loss()
                
                # Total loss with KL weight
                loss = nll_loss + kl_weight * kl_loss / len(train_loader.dataset)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
        
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
        all_predictions = []
        
        for _ in range(self.num_samples):
            predictions = []
            
            self.model.eval()
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    pred = self.model(data)
                    
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
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)
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
        self.model.eval()
        logits = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                pred = self.model(data)
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
            'model': self.model.state_dict(),
            'calibrator': self.calibrator.get_config()
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load model state.
        
        Args:
            state_dict: Dictionary of model state
        """
        self.model.load_state_dict(state_dict['model'])
        self.calibrator.set_config(state_dict['calibrator'])
        
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary of model configuration
        """
        return {
            'num_samples': self.num_samples,
            'prior_std': self.prior_std,
            'calibrator_config': self.calibrator.get_config()
        }
    
    def set_config(self, config: Dict) -> None:
        """Set model configuration.
        
        Args:
            config: Dictionary of model configuration
        """
        self.num_samples = config['num_samples']
        self.prior_std = config['prior_std']
        self.calibrator.set_config(config['calibrator_config'])
