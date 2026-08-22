"""
Multi-Layer Perceptron (MLP) Model for EIS Parameter Prediction

This module implements a neural network-based regressor using scikit-learn's
MLPRegressor for predicting electrochemical parameters from EIS spectra.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
import joblib


class MLPEISModel:
    """
    Multi-Layer Perceptron model for multi-output regression of EIS parameters.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple
        Sizes of hidden layers (default: (128, 64, 32))
    activation : str
        Activation function ('relu', 'tanh', 'logistic')
    learning_rate : str
        Learning rate schedule ('constant', 'adaptive')
    learning_rate_init : float
        Initial learning rate
    max_iter : int
        Maximum number of iterations
    early_stopping : bool
        Use early stopping
    validation_fraction : float
        Fraction of training data for validation
    random_state : int, optional
        Random seed
    """
    
    def __init__(self, hidden_layer_sizes: Tuple[int] = (128, 64, 32),
                 activation: str = 'relu',
                 learning_rate: str = 'adaptive',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 500,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 random_state: Optional[int] = None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            verbose=False
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPEISModel':
        """
        Fit the MLP model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples, n_targets)
        
        Returns
        -------
        self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict electrochemical parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted parameters (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True target values
        
        Returns
        -------
        float
            R² score
        """
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def save(self, filepath: str):
        """
        Save model and scaler to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MLPEISModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved model
        
        Returns
        -------
        MLPEISModel
            Loaded model instance
        """
        data = joblib.load(filepath)
        model = cls()
        model.model = data['model']
        model.scaler = data['scaler']
        model.is_fitted = True
        return model


def train_mlp_model(X_train: np.ndarray, y_train: np.ndarray,
                    hidden_layers: Tuple[int] = (128, 64, 32),
                    random_state: int = 42) -> Tuple[MLPEISModel, StandardScaler]:
    """
    Convenience function to train an MLP model.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    hidden_layers : tuple
        Hidden layer architecture
    random_state : int
        Random seed
    
    Returns
    -------
    tuple
        Trained model and scaler
    """
    model = MLPEISModel(hidden_layer_sizes=hidden_layers, random_state=random_state)
    model.fit(X_train, y_train)
    return model, model.scaler


if __name__ == "__main__":
    # Test model
    from data_generation.synthetic_data import generate_eis_dataset
    
    X, y = generate_eis_dataset(n_samples=5000, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlp_model = MLPEISModel(hidden_layer_sizes=(128, 64, 32), random_state=42)
    mlp_model.fit(X_train, y_train)
    
    r2 = mlp_model.score(X_test, y_test)
    print(f"Test R² Score: {r2:.4f}")
    
    # Save model
    mlp_model.save('mlp_eis_model.joblib')
