# BEACON 🌟
## Bayesian Estimation And Calibration Of Neural-networks


BEACON is a lightweight framework for practical uncertainty quantification in deep learning, combining the power of ensemble methods with Bayesian approximations. It provides researchers and practitioners with tools to obtain reliable uncertainty estimates without the computational overhead typically associated with Bayesian deep learning.

## 🎯 Key Features

- **Adaptive Ensemble Pruning (AEP)**: Dynamically optimize ensemble size while maintaining uncertainty quality
- **Uncertainty Flow Analysis (UFA)**: Novel method to track uncertainty propagation through network layers
- **Calibration Quality Metrics (CQM)**: State-of-the-art metrics for assessing calibration quality
- **Lightweight Posterior Approximation (LPA)**: Efficient Bayesian posterior approximation

## 🚀 Quick Start

```python
import torch
from beacon import UncertaintyModel

# Wrap your existing PyTorch model
model = YourModel()
uncertain_model = UncertaintyModel(model, method='deep_ensemble')

# Train with uncertainty quantification
uncertain_model.fit(train_loader)

# Get predictions with uncertainty estimates
predictions, uncertainties = uncertain_model.predict(test_loader)
```

## 🔬 Scientific Foundation

BEACON builds on established theoretical frameworks:

1. **Ensemble Diversity**: Leverages theoretical work on ensemble diversity measures (Zhou et al., 2012)
2. **Bayesian Approximation**: Based on variational inference principles (Blundell et al., 2015)
3. **Calibration Theory**: Incorporates modern calibration techniques (Guo et al., 2017)

## 📊 Key Methods

### 1. Deep Ensemble+
Enhanced deep ensembles with:
- Diversity-promoting training
- Adaptive size optimization
- Efficient member pruning

### 2. Lightweight SWAG
Efficient implementation of SWA-Gaussian with:
- Reduced memory footprint
- Adaptive rank estimation
- Online covariance updates

### 3. Calibration Suite
Comprehensive calibration tools:
- Temperature scaling
- Isotonic regression
- Distribution matching

## 🎓 Applications

Particularly suited for:
- Medical imaging uncertainty
- Scientific computing
- Risk-sensitive applications
- Active learning systems

## 🛠 Installation

```bash
pip install beacon-uncertainty
```

## 📜 Citation

If you use BEACON in your research, please cite:

```bibtex
@software{beacon2025,
  title={BEACON: Bayesian Estimation And Calibration Of Neural-networks},
  author={Patel, Salil},
  year={2025},
  url={https://github.com/salilp42/BEACON}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details
