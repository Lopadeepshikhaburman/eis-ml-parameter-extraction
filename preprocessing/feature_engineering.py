"""
Feature Engineering for EIS Data

This module provides utilities for extracting domain-specific features
from EIS spectra to improve model performance.
"""

import numpy as np
from typing import Dict, List, Optional


class EISFeatureExtractor:
    """
    Extract physically meaningful features from EIS spectra.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array used for measurements
    """
    
    def __init__(self, frequencies: np.ndarray):
        self.frequencies = frequencies
        self.n_freq = len(frequencies)
    
    def extract_features(self, Z_real: np.ndarray, Z_imag: np.ndarray) -> Dict[str, float]:
        """
        Extract features from impedance spectra.
        
        Parameters
        ----------
        Z_real : np.ndarray
            Real part of impedance (Z')
        Z_imag : np.ndarray
            Imaginary part of impedance (-Z'')
        
        Returns
        -------
        dict
            Dictionary of extracted features
        """
        features = {}
        
        # High-frequency resistance (approximate Rs)
        idx_hf = np.argmax(self.frequencies)
        features['Rs_approx'] = Z_real[idx_hf]
        
        # Low-frequency impedance magnitude
        idx_lf = np.argmin(self.frequencies)
        features['Z_low_freq_mag'] = np.sqrt(Z_real[idx_lf]**2 + Z_imag[idx_lf]**2)
        
        # Impedance at 1 Hz (if in range)
        idx_1hz = np.argmin(np.abs(self.frequencies - 1.0))
        features['Z_at_1Hz'] = np.sqrt(Z_real[idx_1hz]**2 + Z_imag[idx_1hz]**2)
        features['Z_real_at_1Hz'] = Z_real[idx_1hz]
        features['Z_imag_at_1Hz'] = Z_imag[idx_1hz]
        
        # Maximum imaginary impedance (related to time constant)
        features['Z_imag_max'] = np.max(Z_imag)
        features['freq_at_Z_imag_max'] = self.frequencies[np.argmax(Z_imag)]
        
        # Phase angle at characteristic frequencies
        phase = np.arctan2(-Z_imag, Z_real)  # in radians
        features['phase_at_1Hz'] = phase[idx_1hz]
        features['phase_at_100Hz'] = phase[np.argmin(np.abs(self.frequencies - 100.0))]
        
        # Time constant estimate (tau = R * C, from peak frequency)
        freq_peak = features['freq_at_Z_imag_max']
        features['tau_estimate'] = 1.0 / (2 * np.pi * freq_peak) if freq_peak > 0 else 0
        
        # Nyquist plot area (approximate)
        features['nyquist_area'] = np.trapz(Z_imag, Z_real)
        
        # Slope in low-frequency region (Warburg indicator)
        idx_low = self.frequencies < 10.0
        if np.sum(idx_low) > 2:
            slope = np.polyfit(np.log10(self.frequencies[idx_low]), 
                              np.log10(Z_imag[idx_low]), 1)[0]
            features['low_freq_slope'] = slope
        else:
            features['low_freq_slope'] = 0
        
        return features
    
    def extract_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features for a batch of spectra.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_freq * 2)
        
        Returns
        -------
        np.ndarray
            Extracted features (n_samples, n_features)
        """
        n_samples = X.shape[0]
        feature_list = []
        
        for i in range(n_samples):
            Z_real = X[i, :self.n_freq]
            Z_imag = X[i, self.n_freq:]
            
            features = self.extract_features(Z_real, Z_imag)
            feature_list.append(list(features.values()))
        
        return np.array(feature_list)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns
        -------
        list
            List of feature names
        """
        sample_features = self.extract_features(
            np.zeros(self.n_freq), 
            np.zeros(self.n_freq)
        )
        return list(sample_features.keys())


if __name__ == "__main__":
    # Test feature extraction
    from data_generation.synthetic_data import generate_eis_dataset
    
    frequencies = np.logspace(-3, 6, 100)
    X, y = generate_eis_dataset(n_samples=100, n_freq_points=100, random_state=42)
    
    extractor = EISFeatureExtractor(frequencies)
    features = extractor.extract_batch(X)
    
    print(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
    print(f"Feature names: {extractor.get_feature_names()}")
