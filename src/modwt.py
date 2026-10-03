"""
Multiresolution decomposition module.

This implements a practical wavelet-style additive multiresolution decomposition
using Daubechies low-pass filters with reflection boundary handling.

The key properties required for the challenge are:

1. Decomposition into multiple investment horizons.
2. Exact additive reconstruction:
       X = D1 + D2 + ... + DJ + SJ
3. Reflection boundary treatment to reduce future information leakage
   at the right boundary during rolling out-of-sample forecasting.
"""

from typing import List, Tuple

import numpy as np
import pywt


class AdditiveMODWT:
    """
    Additive multiresolution decomposition inspired by MODWT.

    This is not the non-redundant DWT. It keeps all series at the original
    length and gives an exact additive reconstruction.
    """

    def __init__(self, wavelet: str = "db4", levels: int = 5):
        self.wavelet_name = wavelet
        self.levels = int(levels)

        wave = pywt.Wavelet(wavelet)

        # Low-pass scaling filter.
        lo = np.asarray(wave.dec_lo, dtype=float)

        # Normalize so that the filter sums to one.
        # PyWavelets' dec_lo usually sums to sqrt(2).
        lo = lo / np.sqrt(2.0)
        lo = lo / np.sum(lo)

        self.base_filter = lo

    def _pad_reflect(self, x: np.ndarray, left: int, right: int) -> np.ndarray:
        """
        Reflect padding with edge fallback if reflection width is too large.
        """
        if left == 0 and right == 0:
            return x

        try:
            return np.pad(x, (left, right), mode="reflect")
        except ValueError:
            return np.pad(x, (left, right), mode="edge")

    def _upsample_filter(self, stride: int) -> np.ndarray:
        """
        Upsample the base filter for à-trous style decomposition.
        """
        base = self.base_filter
        up = np.zeros(len(base) * stride, dtype=float)
        up[::stride] = base
        up = up / np.sum(up)
        return up

    def _smooth(self, x: np.ndarray, filter_: np.ndarray) -> np.ndarray:
        """
        Apply centered low-pass filter with reflection boundary handling.
        """
        x = np.asarray(x, dtype=float)
        L = len(filter_)

        left_pad = (L - 1) // 2
        right_pad = (L - 1) - left_pad

        padded = self._pad_reflect(x, left_pad, right_pad)

        smooth = np.convolve(padded, filter_, mode="valid")

        # Ensure exact length.
        smooth = smooth[: len(x)]

        return smooth

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

        S = x.copy()
        details = []

        for j in range(1, self.levels + 1):
            stride = 2 ** (j - 1)
            filter_j = self._upsample_filter(stride)

            S_next = self._smooth(S, filter_j)
            D_j = S - S_next

            details.append(D_j)
            S = S_next

        return details, S

    def reconstruct(self, details: List[np.ndarray], smooth: np.ndarray) -> np.ndarray:
        """
        Exact additive reconstruction.
        """
        out = smooth.copy()

        for D in details:
            out = out + D

        return out
