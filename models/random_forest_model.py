"""
Random Forest Model for EIS Parameter Prediction

This module implements a Random Forest-based regressor for predicting
electrochemical parameters from EIS spectra.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import joblib


class RandomForestEISModel:
    """
    Random Forest model for multi-output regression of EIS parameters.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest (default: 100)
    max_depth : int, optional
        Maximum depth of trees
    min_samples_leaf : int
        Minimum samples at leaf node (default: 5)
    random_state : int, optional
        Random seed
    """
    
    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 5,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestEISModel':
        """
        Fit the Random Forest model.
        
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
    
    def feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the Random Forest.
        
        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """
        Save model and scaler to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model (e.g., 'rf_model.joblib')
        """
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestEISModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved model
        
        Returns
        -------
        RandomForestEISModel
            Loaded model instance
        """
        data = joblib.load(filepath)
        model = cls()
        model.model = data['model']
        model.scaler = data['scaler']
        model.is_fitted = True
        return model


if __name__ == "__main__":
    # Test model
    from data_generation.synthetic_data import generate_eis_dataset
    
    X, y = generate_eis_dataset(n_samples=5000, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestEISModel(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    
    r2 = rf_model.score(X_test, y_test)
    print(f"Test R² Score: {r2:.4f}")
    
    # Save model
    rf_model.save('rf_eis_model.joblib')
