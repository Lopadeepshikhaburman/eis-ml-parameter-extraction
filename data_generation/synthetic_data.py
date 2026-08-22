"""
Synthetic EIS Data Generation Module

This module generates synthetic Electrochemical Impedance Spectroscopy (EIS)
data using equivalent circuit models. The generated data serves as training
datasets for machine learning models aimed at predicting electrochemical parameters.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from .equivalent_circuits import RandlesCircuit, RandlesWarburgCircuit


def generate_frequency_grid(n_points: int = 100, 
                            f_min: float = 1e-3, 
                            f_max: float = 1e6) -> np.ndarray:
    """
    Generate logarithmically spaced frequency grid.
    
    Parameters
    ----------
    n_points : int
        Number of frequency points
    f_min : float
        Minimum frequency in Hz
    f_max : float
        Maximum frequency in Hz
    
    Returns
    -------
    np.ndarray
        Array of frequencies in Hz
    """
    return np.logspace(np.log10(f_min), np.log10(f_max), n_points)


def generate_randles_circuit(n_samples: int = 1000,
                             n_freq_points: int = 100,
                             noise_level: float = 0.02,
                             random_state: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic EIS data using Randles circuit model.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_freq_points : int
        Number of frequency points per spectrum
    noise_level : float
        Relative noise amplitude (default 2%)
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        X: impedance spectra (n_samples, n_freq_points * 2)
        y: DataFrame with target parameters (Rs, Rct, Cdl)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    frequencies = generate_frequency_grid(n_freq_points)
    
    # Parameter ranges (physically realistic)
    Rs_range = (0.1, 10.0)  # Ohm
    Rct_range = (1.0, 1000.0)  # Ohm
    Cdl_range = (1e-6, 1e-3)  # Farad
    
    X = np.zeros((n_samples, n_freq_points * 2))
    y_data = {
        'Rs': np.zeros(n_samples),
        'Rct': np.zeros(n_samples),
        'Cdl': np.zeros(n_samples)
    }
    
    for i in range(n_samples):
        # Sample parameters
        Rs = np.random.uniform(*Rs_range)
        Rct = np.random.uniform(*Rct_range)
        Cdl = np.random.uniform(*Cdl_range)
        
        # Generate impedance
        circuit = RandlesCircuit(Rs, Rct, Cdl)
        Z = circuit.impedance(frequencies)
        
        # Add noise
        noise = noise_level * np.abs(Z) * (np.random.randn(len(Z)) + 1j * np.random.randn(len(Z))) / np.sqrt(2)
        Z_noisy = Z + noise
        
        # Store features (Z_real, Z_imag)
        X[i, :n_freq_points] = Z_noisy.real
        X[i, n_freq_points:] = -Z_noisy.imag  # Use -Z'' as convention
        
        # Store targets
        y_data['Rs'][i] = Rs
        y_data['Rct'][i] = Rct
        y_data['Cdl'][i] = Cdl
    
    y = pd.DataFrame(y_data)
    
    return X, y


def generate_eis_dataset(n_samples: int = 10000,
                         circuit_type: str = 'randles',
                         n_freq_points: int = 100,
                         noise_level: float = 0.02,
                         random_state: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate complete EIS dataset for machine learning.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    circuit_type : str
        Type of equivalent circuit ('randles', 'randles_warburg')
    n_freq_points : int
        Number of frequency points
    noise_level : float
        Relative noise level
    random_state : int, optional
        Random seed
    
    Returns
    -------
    tuple
        X: feature matrix (n_samples, n_freq_points * 2)
        y: DataFrame with target parameters
    """
    if circuit_type == 'randles':
        return generate_randles_circuit(n_samples, n_freq_points, noise_level, random_state)
    elif circuit_type == 'randles_warburg':
        # TODO: Implement Randles + Warburg circuit
        raise NotImplementedError("Randles + Warburg circuit not yet implemented")
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


if __name__ == "__main__":
    # Test data generation
    X, y = generate_eis_dataset(n_samples=1000, random_state=42)
    print(f"Generated dataset shape: X={X.shape}, y={y.shape}")
    print(f"Target statistics:\n{y.describe()}")
