#!/usr/bin/env python3
"""
Training Pipeline for EIS Parameter Prediction Model

This script orchestrates the complete training workflow:
1. Generate synthetic EIS dataset
2. Split into train/test sets
3. Train multiple models (RF, MLP)
4. Evaluate and compare performance
5. Save best models
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import joblib

# Import project modules
from data_generation.synthetic_data import generate_eis_dataset
from models.random_forest_model import RandomForestEISModel
from models.mlp_model import MLPEISModel
from evaluation.metrics import evaluate_model, print_metrics_table


def main(n_samples=10000, n_freq_points=100, random_state=42):
    """
    Run complete training pipeline.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic samples to generate
    n_freq_points : int
        Number of frequency points per spectrum
    random_state : int
        Random seed for reproducibility
    """
    print("="*70)
    print("EIS ML Parameter Extraction - Training Pipeline")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic EIS dataset...")
    X, y = generate_eis_dataset(
        n_samples=n_samples,
        n_freq_points=n_freq_points,
        circuit_type='randles',
        random_state=random_state
    )
    print(f"Generated {n_samples} samples with {X.shape[1]} features")
    print(f"Target parameters: {list(y.columns)}")
    
    # Step 2: Train-test split
    print("\n[2/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train Random Forest model
    print("\n[3/5] Training Random Forest model...")
    rf_model = RandomForestEISModel(
        n_estimators=200,
        max_depth=30,
        min_samples_leaf=3,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete")
    
    # Step 4: Train MLP model
    print("\n[4/5] Training MLP model...")
    mlp_model = MLPEISModel(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=random_state
    )
    mlp_model.fit(X_train, y_train)
    print("MLP training complete")
    
    # Step 5: Evaluate both models
    print("\n[5/5] Evaluating models on test set...")
    target_names = list(y.columns)
    
    print("\n--- Random Forest Performance ---")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, target_names)
    print_metrics_table(rf_metrics, target_names)
    
    print("\n--- MLP Performance ---")
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, target_names)
    print_metrics_table(mlp_metrics, target_names)
    
    # Compare models
    print("\n" + "="*70)
    print("Model Comparison (Overall R²)")
    print("="*70)
    print(f"Random Forest: {rf_metrics['r2_overall']:.4f}")
    print(f"MLP: {mlp_metrics['r2_overall']:.4f}")
    
    if rf_metrics['r2_overall'] > mlp_metrics['r2_overall']:
        print("\n✅ Random Forest performs better overall")
    else:
        print("\n✅ MLP performs better overall")
    
    # Save best models
    print("\n" + "="*70)
    print("Saving models...")
    print("="*70)
    
    rf_model.save('rf_eis_model.joblib')
    mlp_model.save('mlp_eis_model.joblib')
    
    # Save training metadata
    metadata = {
        'n_samples': n_samples,
        'n_freq_points': n_freq_points,
        'random_state': random_state,
        'rf_r2_overall': rf_metrics['r2_overall'],
        'mlp_r2_overall': mlp_metrics['r2_overall'],
        'target_names': target_names
    }
    joblib.dump(metadata, 'training_metadata.joblib')
    print("Training metadata saved")
    
    print("\n" + "="*70)
    print("Training pipeline complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EIS parameter prediction model')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of synthetic samples')
    parser.add_argument('--n_freq', type=int, default=100,
                        help='Number of frequency points')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    main(n_samples=args.n_samples, n_freq_points=args.n_freq, random_state=args.seed)
