# BEACON 
## Bayesian Estimation And Calibration Of Neural-networks


BEACON is a lightweight framework for practical uncertainty quantification in deep learning, designed to help ML researchers obtain reliable uncertainty estimates without excessive computational overhead. It combines state-of-the-art ensemble methods with efficient Bayesian approximations.

## Key Features & Theoretical Foundations

### 1. Deep Ensemble with Diversity Promotion
- **Implementation**: `beacon.methods.deep_ensemble.DeepEnsemble`
- **Theory**: Based on the principle that diverse ensemble members capture different aspects of predictive uncertainty. Our implementation extends traditional deep ensembles (Lakshminarayanan et al., 2017) with:
  - Adaptive diversity-promoting loss term that minimizes prediction correlation
  - Efficient parameter-space diversity metrics
  - Dynamic ensemble size optimization through member pruning

### 2. Stochastic Weight Averaging-Gaussian (SWAG)
- **Implementation**: `beacon.methods.swag.SWAG`
- **Theory**: Approximates the posterior distribution over neural network weights using a Gaussian distribution. Our implementation includes:
  - Low-rank plus diagonal covariance structure
  - Online covariance updates for memory efficiency
  - Adaptive rank estimation based on explained variance

### 3. Bayes By Backprop (BBP)
- **Implementation**: `beacon.methods.bbp.BayesByBackprop`
- **Theory**: Performs variational inference by minimizing the KL divergence between the approximate posterior and true posterior. Features:
  - Reparameterization trick for efficient gradient computation
  - Local reparameterization for reduced variance
  - Adaptive prior based on empirical Bayes

### 4. Advanced Calibration Suite
- **Implementation**: `beacon.core.calibration.ModelCalibrator`
- **Theory**: Combines multiple calibration methods to ensure reliable uncertainty estimates:
  - Temperature scaling with automatic optimization
  - Isotonic regression for non-parametric calibration
  - Distribution matching for improved reliability

## Unique Contributions

1. **Adaptive Ensemble Pruning (AEP)**
   - Dynamically optimizes ensemble size while maintaining uncertainty quality
   - Uses theoretically-grounded diversity metrics (implemented in `utils.diversity`)
   - Reduces computational overhead without sacrificing performance

2. **Comprehensive Uncertainty Metrics**
   - Implementation: `beacon.utils.diagnostics.UncertaintyMetrics`
   - Includes:
     - Expected Calibration Error (ECE)
     - Prediction Interval Coverage Probability (PICP)
     - Uncertainty Calibration Score
     - Sharpness and Dispersion metrics

3. **Visualization Tools**
   - Implementation: `beacon.utils.visualization`
   - Provides:
     - Calibration curves
     - Reliability diagrams
     - Uncertainty distribution plots
     - Ensemble diversity visualizations

## Quick Start

```python
import torch
from beacon import UncertaintyModel

# Wrap your existing PyTorch model
model = YourModel()
uncertain_model = UncertaintyModel(
    model, 
    method='deep_ensemble',  # or 'swag', 'bbp'
    num_models=5
)

# Train with uncertainty quantification
uncertain_model.fit(train_loader)

# Get predictions with uncertainty estimates
predictions, uncertainties = uncertain_model.predict(test_loader)
```
## Installation

```bash
pip install beacon-uncertainty
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0 (for visualization)


## License

MIT License - see [LICENSE](LICENSE) for details
