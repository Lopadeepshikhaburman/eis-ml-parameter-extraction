"""
Evaluation Metrics for EIS Parameter Prediction

This module provides comprehensive evaluation metrics for assessing
the performance of ML models on EIS parameter prediction tasks.
"""

import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      target_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values (n_samples, n_targets)
    target_names : list, optional
        Names of target variables
    
    Returns
    -------
    dict
        Dictionary of metrics
    """
    if target_names is None:
        target_names = [f"target_{i}" for i in range(y_true.shape[1])]
    
    metrics = {}
    
    # Overall metrics
    metrics['r2_overall'] = r2_score(y_true, y_pred)
    metrics['rmse_overall'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae_overall'] = mean_absolute_error(y_true, y_pred)
    
    # Per-target metrics
    for i, name in enumerate(target_names):
        metrics[f'r2_{name}'] = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f'rmse_{name}'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        metrics[f'mae_{name}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
    
    return metrics


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   target_names: List[str] = None,
                   scaler=None) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Parameters
    ----------
    model : object
        Trained model with predict() method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        True test targets
    target_names : list, optional
        Names of target variables
    scaler : object, optional
        Feature scaler (if model doesn't handle scaling internally)
    
    Returns
    -------
    dict
        Comprehensive evaluation metrics
    """
    # Predict
    if scaler is not None:
        X_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, target_names)
    
    return metrics


def print_metrics_table(metrics: Dict[str, float], target_names: List[str]):
    """
    Print metrics in a formatted table.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary
    target_names : list
        Target variable names
    """
    print("\n" + "="*70)
    print(f"{'Parameter':<25} {'R²':<15} {'RMSE':<15} {'MAE':<15}")
    print("="*70)
    
    for name in target_names:
        r2 = metrics.get(f'r2_{name}', 0)
        rmse = metrics.get(f'rmse_{name}', 0)
        mae = metrics.get(f'mae_{name}', 0)
        print(f"{name:<25} {r2:<15.4f} {rmse:<15.4f} {mae:<15.4f}")
    
    print("="*70)
    print(f"Overall R²: {metrics['r2_overall']:.4f}")
    print(f"Overall RMSE: {metrics['rmse_overall']:.4f}")
    print(f"Overall MAE: {metrics['mae_overall']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test metrics
    y_true = np.random.randn(100, 3)
    y_pred = y_true + 0.1 * np.random.randn(100, 3)
    
    target_names = ['Rs', 'Rct', 'Cdl']
    metrics = calculate_metrics(y_true, y_pred, target_names)
    print_metrics_table(metrics, target_names)
