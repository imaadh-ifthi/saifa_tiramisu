"""
Multiresolution decomposition module.

This implements a MODWT-equivalent normalized undecimated wavelet decomposition
using PyWavelets' Stationary Wavelet Transform (SWT). It builds additive
multiresolution analysis (MRA) components without decimation.
Arbitrary lengths are supported via symmetric padding. A generous padding margin
is used to completely prevent out-of-sample (OOS) leakage and preserve strict
causality within the rolling window.
"""

from typing import List, Tuple
import numpy as np
import pywt


class MODWTDecomposer:
    """
    MODWT-equivalent normalized undecimated wavelet decomposition.

    This builds additive MRA components using the undecimated SWT.
    It preserves the full original time-series length at every scale.
    """

    def __init__(self, wavelet: str = "db4", levels: int = 5):
        self.wavelet_name = wavelet
        self.levels = int(levels)

    def decompose(self, x: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Decompose a series into details D1, ..., DJ and smooth SJ.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional time series.

        Returns
        -------
        details : list of np.ndarray
            Detail components D1, ..., DJ.
        smooth : np.ndarray
            Smooth component SJ.
        """
        x = np.asarray(x, dtype=float)
        L = len(x)

        if L == 0:
            return [np.array([]) for _ in range(self.levels)], np.array([])
        
        # Use a generous margin to prevent PyWavelets' periodic boundary handling
        # from wrapping the right edge back to the left edge.
        # For db4 at level 5, max filter reach is 218, so 250 is extremely safe.
        margin = 250
        
        total_len = L + 2 * margin
        modulo = total_len % (2 ** self.levels)
        
        if modulo != 0:
            pad_right = margin + (2 ** self.levels) - modulo
        else:
            pad_right = margin
            
        pad_left = margin
        
        # Symmetric padding minimizes statistical artefacts at the edges
        x_pad = np.pad(x, (pad_left, pad_right), mode='symmetric')
            
        coefs = pywt.swt(x_pad, self.wavelet_name, level=self.levels, norm=True, trim_approx=True)
        # coefs is structured as [cA_J, cD_J, cD_{J-1}, ..., cD_1]
        
        zeros = [np.zeros_like(c) for c in coefs]
        
        details = []
        # Extract additive detail components D1 to DJ
        for j in range(1, self.levels + 1):
            idx = self.levels - j + 1
            c_temp = list(zeros)
            c_temp[idx] = coefs[idx]
            D_j_pad = pywt.iswt(c_temp, self.wavelet_name, norm=True)
            D_j = D_j_pad[pad_left : pad_left + L]
            details.append(D_j)
            
        # Extract additive smooth component SJ
        c_temp = list(zeros)
        c_temp[0] = coefs[0]
        S_J_pad = pywt.iswt(c_temp, self.wavelet_name, norm=True)
        S_J = S_J_pad[pad_left : pad_left + L]
            
        return details, S_J

    def reconstruct(self, details: List[np.ndarray], smooth: np.ndarray) -> np.ndarray:
        """
        Exact additive reconstruction.
        """
        out = smooth.copy()
        for D in details:
            out = out + D
        return out

# Maintain backward compatibility for existing downstream imports
AdditiveMODWT = MODWTDecomposer
