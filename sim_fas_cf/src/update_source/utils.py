import numpy as np
import torch 
import math 
import logging

log = logging.getLogger(__name__)

# Sinh mảng số phức ngẫu nhiên theo phân phối chuẩn phức 
def crandn(shape, generator, device, dtype=torch.complex64) -> torch.Tensor:
    """i.i.d. CN(0, 1): independent real and imaginary parts N(0, 1/2)."""
    rd = torch.float32 if dtype == torch.complex64 else torch.float64
    re = torch.randn(shape, generator=generator, device=generator.device, dtype=rd)
    im = torch.randn(shape, generator=generator, device=generator.device, dtype=rd)
    return (torch.complex(re, im) / (2.0 ** 0.5)).to(device)

#sinc(x)
def spherical_j0(x):
    return np.sinc(np.asarray(x, dtype=np.float64) / np.pi)

#sqrt for matrix 
def psd_sqrt(R: np.ndarray, tol: float = 1e-9, name: str = "R") -> np.ndarray:
    R = 0.5 * (R + R.conj().T) # remove round-off asymmetry
    eigvals, eigvecs = np.linalg.eigh(R) # R = V diag(w) V^H --> R^1/2 = V diag(w^1/2) V^H

    # A clearly negative eigenvalue means R is not PSD; a tiny one is only round-off
    scale = max(1.0, float(eigvals.max()))
    if eigvals.min() < -tol * scale:
        raise ValueError(f"{name} is not PSD (min eigenvalue {eigvals.min():.3e})")
    n_clip = int((eigvals < 0).sum())
    if n_clip:
        log.info("psd_sqrt(%s): clipped %d round-off negative eigenvalue(s), min %.3e", name, n_clip, eigvals.min())

    # R = V diag(w) V^H  ->  R^{1/2} = V diag(sqrt(w)) V^H
    sqrt_eigvals = np.sqrt(np.clip(eigvals, 0.0, None))
    return (eigvecs * sqrt_eigvals) @ eigvecs.conj().T

#Rayleigh-Sommerfeld
def rs_coefficient(d, dz, area, wavelength):
    return area * dz / d ** 2 * (1.0 / (2.0 * np.pi * d) - 1j / wavelength) * np.exp(1j * 2.0 * np.pi * d / wavelength)

#Phase-shift in each SIM layer
def phase_factors(phase: torch.Tensor, cdtype) -> torch.Tensor:
    return torch.exp(1j * phase.to(torch.float64)).to(cdtype)
