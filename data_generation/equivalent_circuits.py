"""
Equivalent Circuit Models for EIS

This module implements various equivalent circuit models used to generate
synthetic EIS spectra. Each circuit class provides methods to compute
impedance over a frequency range.
"""

import numpy as np
from typing import Union
from scipy.constants import pi


class RandlesCircuit:
    """
    Classic Randles circuit: Rs - (Rct || Cdl)
    
    Parameters
    ----------
    Rs : float
        Solution resistance (Ohm)
    Rct : float
        Charge transfer resistance (Ohm)
    Cdl : float
        Double layer capacitance (Farad)
    """
    
    def __init__(self, Rs: float, Rct: float, Cdl: float):
        self.Rs = Rs
        self.Rct = Rct
        self.Cdl = Cdl
    
    def impedance(self, frequency: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Calculate impedance of Randles circuit.
        
        Parameters
        ----------
        frequency : float or np.ndarray
            Frequency in Hz
        
        Returns
        -------
        complex or np.ndarray
            Impedance Z = Z' + jZ''
        """
        omega = 2 * pi * frequency
        
        # Capacitor impedance: Z_C = 1 / (j * omega * C)
        Z_C = 1 / (1j * omega * self.Cdl)
        
        # Parallel combination: Z_parallel = (Rct * Z_C) / (Rct + Z_C)
        Z_parallel = (self.Rct * Z_C) / (self.Rct + Z_C)
        
        # Total impedance: Z_total = Rs + Z_parallel
        Z_total = self.Rs + Z_parallel
        
        return Z_total
    
    def __repr__(self):
        return f"RandlesCircuit(Rs={self.Rs:.3f}, Rct={self.Rct:.3f}, Cdl={self.Cdl:.3e})"


class RandlesWarburgCircuit:
    """
    Randles circuit with Warburg diffusion element: Rs - (Rct || Cdl) - W
    
    Parameters
    ----------
    Rs : float
        Solution resistance (Ohm)
    Rct : float
        Charge transfer resistance (Ohm)
    Cdl : float
        Double layer capacitance (Farad)
    sigma : float
        Warburg coefficient (Ohm s^(-1/2))
    """
    
    def __init__(self, Rs: float, Rct: float, Cdl: float, sigma: float):
        self.Rs = Rs
        self.Rct = Rct
        self.Cdl = Cdl
        self.sigma = sigma
    
    def impedance(self, frequency: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Calculate impedance of Randles + Warburg circuit.
        
        Parameters
        ----------
        frequency : float or np.ndarray
            Frequency in Hz
        
        Returns
        -------
        complex or np.ndarray
            Impedance Z = Z' + jZ''
        """
        omega = 2 * pi * frequency
        
        # Capacitor impedance
        Z_C = 1 / (1j * omega * self.Cdl)
        
        # Warburg impedance: Z_W = sigma / sqrt(j * omega) = sigma / sqrt(omega) * (1 - j) / sqrt(2)
        Z_W = self.sigma / np.sqrt(1j * omega)
        
        # Parallel combination: (Rct || Cdl)
        Z_parallel = (self.Rct * Z_C) / (self.Rct + Z_C)
        
        # Total impedance: Rs + Z_parallel + Z_W
        Z_total = self.Rs + Z_parallel + Z_W
        
        return Z_total
    
    def __repr__(self):
        return f"RandlesWarburgCircuit(Rs={self.Rs:.3f}, Rct={self.Rct:.3f}, Cdl={self.Cdl:.3e}, sigma={self.sigma:.3e})"


class ConstantPhaseElement:
    """
    Constant Phase Element (CPE) for modeling non-ideal capacitance.
    
    Impedance: Z_CPE = 1 / (Q * (j * omega)^n)
    
    Parameters
    ----------
    Q : float
        CPE coefficient (F s^(n-1))
    n : float
        CPE exponent (0 < n <= 1)
    """
    
    def __init__(self, Q: float, n: float):
        if not (0 < n <= 1):
            raise ValueError("CPE exponent n must be in range (0, 1]")
        self.Q = Q
        self.n = n
    
    def impedance(self, frequency: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Calculate CPE impedance.
        
        Parameters
        ----------
        frequency : float or np.ndarray
            Frequency in Hz
        
        Returns
        -------
        complex or np.ndarray
            Impedance Z_CPE
        """
        omega = 2 * pi * frequency
        Z_cpe = 1 / (self.Q * (1j * omega) ** self.n)
        return Z_cpe
    
    def __repr__(self):
        return f"ConstantPhaseElement(Q={self.Q:.3e}, n={self.n:.3f})"


if __name__ == "__main__":
    # Test circuits
    import matplotlib.pyplot as plt
    
    freq = np.logspace(-3, 6, 100)
    
    # Randles circuit
    randles = RandlesCircuit(Rs=1.0, Rct=100.0, Cdl=1e-4)
    Z_randles = randles.impedance(freq)
    
    # Randles + Warburg
    randles_w = RandlesWarburgCircuit(Rs=1.0, Rct=100.0, Cdl=1e-4, sigma=50.0)
    Z_randles_w = randles_w.impedance(freq)
    
    # Plot Nyquist
    plt.figure(figsize=(8, 6))
    plt.plot(Z_randles.real, -Z_randles.imag, 'b-', label='Randles')
    plt.plot(Z_randles_w.real, -Z_randles_w.imag, 'r-', label='Randles + Warburg')
    plt.xlabel("Z' (Ohm)")
    plt.xlabel("-Z'' (Ohm)")
    plt.legend()
    plt.title("Nyquist Plot Comparison")
    plt.grid(True, alpha=0.3)
    plt.show()
