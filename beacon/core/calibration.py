import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict
import numpy as np
from sklearn.isotonic import IsotonicRegression

class ModelCalibrator:
    """Model calibration using various methods."""
    
    def __init__(
        self,
        method: str = 'temperature',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize calibrator.
        
        Args:
            method: Calibration method ['temperature', 'isotonic']
            device: Device to run computations on
        """
        self.method = method.lower()
        self.device = device
        self.calibrated = False
        
        if self.method == 'temperature':
            self.temperature = nn.Parameter(torch.ones(1).to(device))
        elif self.method == 'isotonic':
            self.calibrators = []
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def calibrate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> None:
        """Calibrate model outputs.
        
        Args:
            logits: Raw model outputs
            targets: Ground truth targets
            val_loader: Optional validation data loader
        """
        if self.method == 'temperature':
            self._temperature_scaling(logits, targets)
        elif self.method == 'isotonic':
            self._isotonic_regression(logits, targets)
        
        self.calibrated = True
    
    def _temperature_scaling(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """Temperature scaling calibration.
        
        Args:
            logits: Raw model outputs
            targets: Ground truth targets
        """
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, targets)
            loss.backward()
            return loss
            
        optimizer.step(eval)
    
    def _isotonic_regression(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """Isotonic regression calibration.
        
        Args:
            logits: Raw model outputs
            targets: Ground truth targets
        """
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        self.calibrators = []
        for c in range(probs.shape[1]):
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs[:, c], (targets == c).astype(float))
            self.calibrators.append(calibrator)
    
    def calibrate_predictions(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply calibration to new predictions.
        
        Args:
            logits: Raw model outputs
            
        Returns:
            Calibrated predictions
        """
        if not self.calibrated:
            raise RuntimeError("Model must be calibrated before making predictions")
            
        if self.method == 'temperature':
            return torch.softmax(logits / self.temperature, dim=1)
        elif self.method == 'isotonic':
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            calibrated = np.zeros_like(probs)
            for c, calibrator in enumerate(self.calibrators):
                calibrated[:, c] = calibrator.predict(probs[:, c])
            return torch.from_numpy(calibrated).to(self.device)
    
    def get_config(self) -> Dict:
        """Get calibrator configuration.
        
        Returns:
            Dictionary of calibrator configuration
        """
        config = {'method': self.method, 'calibrated': self.calibrated}
        if self.method == 'temperature':
            config['temperature'] = self.temperature.item()
        return config
    
    def set_config(self, config: Dict) -> None:
        """Set calibrator configuration.
        
        Args:
            config: Dictionary of calibrator configuration
        """
        self.method = config['method']
        self.calibrated = config['calibrated']
        if self.method == 'temperature':
            self.temperature = nn.Parameter(torch.tensor([config['temperature']]).to(self.device))
