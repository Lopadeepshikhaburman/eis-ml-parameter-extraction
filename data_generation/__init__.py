"""Data generation module for synthetic EIS spectra."""

from .synthetic_data import generate_eis_dataset, generate_randles_circuit
from .equivalent_circuits import RandlesCircuit, RandlesWarburgCircuit

__all__ = [
    'generate_eis_dataset',
    'generate_randles_circuit',
    'RandlesCircuit',
    'RandlesWarburgCircuit'
]
