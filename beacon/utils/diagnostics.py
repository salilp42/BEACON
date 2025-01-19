import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import calibration_curve

class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quality."""
    
    def compute_all(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all uncertainty metrics.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy for sklearn metrics
        predictions = predictions.cpu().numpy()
        uncertainties = uncertainties.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Classification metrics
        if predictions.shape[-1] > 1:  # Multi-class
            metrics.update(self._compute_classification_metrics(
                predictions, uncertainties, targets
            ))
        else:  # Regression
            metrics.update(self._compute_regression_metrics(
                predictions, uncertainties, targets
            ))
            
        # General uncertainty metrics
        metrics.update(self._compute_uncertainty_metrics(
            predictions, uncertainties, targets
        ))
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification-specific metrics.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to probabilities if logits
        if len(predictions.shape) > 1:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        
        # One-vs-all ROC AUC
        try:
            metrics['auroc'] = roc_auc_score(
                targets,
                predictions,
                multi_class='ovr'
            )
        except ValueError:
            metrics['auroc'] = np.nan
            
        # Average precision
        try:
            metrics['aupr'] = average_precision_score(
                targets,
                predictions,
                average='macro'
            )
        except ValueError:
            metrics['aupr'] = np.nan
            
        # Expected calibration error
        prob_true, prob_pred = calibration_curve(
            targets,
            np.max(predictions, axis=1),
            n_bins=10,
            strategy='uniform'
        )
        metrics['ece'] = np.mean(np.abs(prob_true - prob_pred))
        
        return metrics
    
    def _compute_regression_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression-specific metrics.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Negative log likelihood
        nll = -np.mean(self._gaussian_log_likelihood(
            targets,
            predictions,
            uncertainties
        ))
        metrics['nll'] = nll
        
        # Root mean squared error
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        metrics['rmse'] = rmse
        
        # Prediction interval coverage
        coverage = self._compute_coverage(predictions, uncertainties, targets)
        metrics['picp'] = coverage
        
        return metrics
    
    def _compute_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute general uncertainty metrics.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Uncertainty calibration
        uncertainty_scores = self._compute_uncertainty_scores(
            predictions,
            uncertainties,
            targets
        )
        metrics['uncertainty_auroc'] = roc_auc_score(
            uncertainty_scores['errors'],
            uncertainty_scores['uncertainties']
        )
        
        # Uncertainty sharpness
        metrics['sharpness'] = np.mean(uncertainties)
        
        # Uncertainty dispersion
        metrics['dispersion'] = np.std(uncertainties)
        
        return metrics
    
    def _gaussian_log_likelihood(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """Compute Gaussian log likelihood.
        
        Args:
            targets: Ground truth targets
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            eps: Small constant for numerical stability
            
        Returns:
            Log likelihood values
        """
        return -0.5 * (np.log(2 * np.pi) + np.log(uncertainties + eps) + 
                      (targets - predictions) ** 2 / (uncertainties + eps))
    
    def _compute_coverage(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Compute prediction interval coverage probability.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            confidence: Confidence level
            
        Returns:
            Coverage probability
        """
        z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), confidence))
        lower = predictions - z_score * np.sqrt(uncertainties)
        upper = predictions + z_score * np.sqrt(uncertainties)
        
        return np.mean((targets >= lower) & (targets <= upper))
    
    def _compute_uncertainty_scores(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute uncertainty scores for evaluation.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Dictionary of uncertainty scores
        """
        if len(predictions.shape) > 1:  # Classification
            errors = (np.argmax(predictions, axis=1) != targets).astype(float)
        else:  # Regression
            errors = np.abs(predictions - targets)
            
        return {
            'errors': errors,
            'uncertainties': uncertainties.mean(axis=1) if len(uncertainties.shape) > 1 else uncertainties
        }
