import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import seaborn as sns
from sklearn.calibration import calibration_curve

def plot_uncertainty_calibration(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
    fig_size: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot uncertainty calibration curve.
    
    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: Ground truth targets
        num_bins: Number of bins for calibration
        fig_size: Figure size
        
    Returns:
        Matplotlib figure
    """
    predictions = predictions.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()
    targets = targets.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        targets,
        predictions.max(axis=1) if len(predictions.shape) > 1 else predictions,
        n_bins=num_bins
    )
    
    # Plot calibration curve
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_uncertainty_histogram(
    uncertainties: torch.Tensor,
    errors: torch.Tensor,
    num_bins: int = 30,
    fig_size: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot histogram of uncertainties for correct and incorrect predictions.
    
    Args:
        uncertainties: Uncertainty estimates
        errors: Binary indicator of prediction errors
        num_bins: Number of histogram bins
        fig_size: Figure size
        
    Returns:
        Matplotlib figure
    """
    uncertainties = uncertainties.cpu().numpy()
    errors = errors.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot histograms
    ax.hist(uncertainties[errors == 0], bins=num_bins, alpha=0.5,
            label='Correct predictions', density=True)
    ax.hist(uncertainties[errors == 1], bins=num_bins, alpha=0.5,
            label='Incorrect predictions', density=True)
    
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Density')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_reliability_diagram(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
    fig_size: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot reliability diagram with uncertainty.
    
    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: Ground truth targets
        num_bins: Number of bins
        fig_size: Figure size
        
    Returns:
        Matplotlib figure
    """
    predictions = predictions.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()
    targets = targets.cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [3, 1]})
    
    # Compute reliability curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        targets,
        predictions.max(axis=1) if len(predictions.shape) > 1 else predictions,
        n_bins=num_bins
    )
    
    # Plot reliability curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    
    # Add uncertainty bands
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(predictions.max(axis=1) if len(predictions.shape) > 1 else predictions,
                            bin_edges) - 1
    
    uncertainty_means = np.array([uncertainties[bin_indices == i].mean()
                                for i in range(num_bins)])
    
    ax1.fill_between(mean_predicted_value,
                    fraction_of_positives - uncertainty_means,
                    fraction_of_positives + uncertainty_means,
                    alpha=0.2, label='Uncertainty')
    
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Reliability Diagram with Uncertainty')
    ax1.legend()
    ax1.grid(True)
    
    # Plot sample counts
    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    ax2.bar(mean_predicted_value, bin_counts, width=1/num_bins)
    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Count')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_ensemble_diversity(
    predictions: torch.Tensor,
    fig_size: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot ensemble diversity metrics.
    
    Args:
        predictions: Predictions from ensemble members
        fig_size: Figure size
        
    Returns:
        Matplotlib figure
    """
    predictions = predictions.cpu().numpy()
    num_models = predictions.shape[0]
    
    # Compute pairwise disagreements
    disagreement_matrix = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                disagreement_matrix[i, j] = np.mean(
                    predictions[i].argmax(axis=1) != predictions[j].argmax(axis=1)
                )
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot heatmap
    sns.heatmap(disagreement_matrix, annot=True, cmap='YlOrRd', ax=ax)
    ax.set_title('Ensemble Diversity (Pairwise Disagreement)')
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Model Index')
    
    return fig
