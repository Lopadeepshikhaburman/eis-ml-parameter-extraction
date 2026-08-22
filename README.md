# Machine Learning Assisted Extraction of Electrochemical Parameters from EIS Data

## Research Motivation

Electrochemical Impedance Spectroscopy (EIS) is one of the most powerful techniques for characterising electrochemical systems such as supercapacitors, batteries, fuel cells, sensors, and electrocatalysts. However, extracting meaningful electrochemical parameters from EIS measurements typically requires equivalent circuit fitting, which can be time-consuming, user-dependent, and often suffers from non-unique solutions.

The objective of this work is to develop a physics-guided machine learning framework capable of directly predicting electrochemical parameters from EIS spectra while maintaining consistency with electrochemical theory.

## Research Objective

To develop a machine learning model that can:

1. Predict electrochemical parameters directly from EIS data
2. Reconstruct impedance behaviour from predicted parameters
3. Reduce dependence on manual equivalent circuit fitting
4. Provide rapid and automated interpretation of EIS measurements
5. Establish a foundation for future validation using experimental electrochemical data

## Target Parameters

The model aims to predict:

- **Solution resistance (Rs)**
- **Charge transfer resistance (Rct)**
- **Double layer capacitance (Cdl)**
- **Constant phase element (CPE) parameters**
- **Warburg diffusion parameters**
- **Characteristic time constants**

**Derived outputs include:**
- Specific capacitance
- Energy density
- Power density
- Nyquist plot reconstruction
- Bode response reconstruction

## Methodology

### Dataset Generation

The current training dataset consists entirely of synthetic EIS data generated from established equivalent circuit models.

#### Synthetic Data

EIS spectra are generated using equivalent circuit models such as:
- Randles Circuit
- Rs-(Rct||Cdl)-W
- R(RC)(RCW)
- Circuits containing Constant Phase Elements (CPE)

Parameter values are sampled from physically realistic electrochemical ranges to create a diverse dataset covering a wide range of impedance responses. Noise can also be introduced into the simulated spectra to improve model robustness and better approximate practical measurement conditions.

#### Future Experimental Validation

Experimental EIS data are not yet available and are therefore not included in the current study. Once experimental measurements become available, they will be used to evaluate the model's ability to generalise beyond synthetic datasets and to assess its practical applicability.

### Machine Learning Framework

#### Input Features
- Frequency
- Real impedance (Z')
- Imaginary impedance (−Z'')

#### Models Under Evaluation
- Random Forest Regressor
- Gradient Boosting Regressor
- Multi-Layer Perceptron (MLP)
- 1D Convolutional Neural Network (CNN)
- Ensemble Learning Approaches

#### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Physical consistency of predicted parameters
- Accuracy of impedance reconstruction

## Expected Outcomes

The framework is expected to:
- Accelerate electrochemical data analysis
- Provide rapid estimation of equivalent circuit parameters
- Enable automated interpretation of EIS measurements
- Demonstrate the feasibility of machine-learning-based parameter extraction from impedance spectra
- Serve as a foundation for future validation using experimental electrochemical systems

## Project Structure

```
eis-ml-parameter-extraction/
├── README.md
├── requirements.txt
├── data_generation/
│   ├── __init__.py
│   ├── synthetic_data.py
│   └── equivalent_circuits.py
├── models/
│   ├── __init__.py
│   ├── random_forest_model.py
│   ├── mlp_model.py
│   └── ensemble_model.py
├── preprocessing/
│   ├── __init__.py
│   └── feature_engineering.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── notebooks/
│   └── training_pipeline.ipynb
└── scripts/
    └── train_model.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

```python
from data_generation.synthetic_data import generate_eis_dataset

# Generate dataset with 10000 samples
data = generate_eis_dataset(n_samples=10000, circuit_type='randles')
```

### Train Model

```python
from models.mlp_model import train_mlp_model

# Train MLP model
model, scaler = train_mlp_model(X_train, y_train, hidden_layers=(128, 64, 32))
```

### Evaluate Performance

```python
from evaluation.metrics import evaluate_model

# Evaluate on test set
metrics = evaluate_model(model, X_test, y_test, scaler)
print(f"R² Score: {metrics['r2']}")
print(f"RMSE: {metrics['rmse']}")
```

## Long-Term Vision

The long-term goal is to develop a robust and interpretable machine learning platform capable of extracting physically meaningful electrochemical information directly from EIS measurements, reducing reliance on manual fitting procedures and facilitating rapid characterisation of energy-storage materials. The current work focuses on synthetic data generation and model development, with future expansion toward validation using experimental datasets.

## License

MIT License

## Contact

Deepshikha Burman  
Masters in Physics (specialization in CMP)  


---

**Note**: This repository is under active development. Experimental EIS data will be added once available for model validation.
