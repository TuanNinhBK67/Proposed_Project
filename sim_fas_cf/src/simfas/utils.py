"""Numerical helpers, seeding, running statistics and CSV/JSON logging."""
import csv
import json
import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------- numerics
def complex_dtype(name: str) -> torch.dtype:
    return {"complex64": torch.complex64, "complex128": torch.complex128}[name]


def real_dtype(cdtype: torch.dtype) -> torch.dtype:
    return torch.float32 if cdtype == torch.complex64 else torch.float64


def crandn(shape, generator, device, dtype=torch.complex64) -> torch.Tensor:
    """i.i.d. CN(0, 1): independent real and imaginary parts N(0, 1/2)."""
    rd = real_dtype(dtype)
    re = torch.randn(shape, generator=generator, device=generator.device, dtype=rd)
    im = torch.randn(shape, generator=generator, device=generator.device, dtype=rd)
    return (torch.complex(re, im) / (2.0 ** 0.5)).to(device)


# ---------------------------------------------------------------------- seeding
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device="cpu") -> torch.Generator:
    """Dedicated generator, e.g. for the environment (geometry, shadowing, fading, initial configuration),
    so that the scenario stream does not depend on how a policy consumes random numbers. The environment uses
    a CPU generator and moves the draws to its device: CUDA and CPU generators give different streams, and the
    held-out scenarios must be identical on every device."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed))
    return g


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if "cuda" in s and torch.cuda.is_available() and len(s["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])


# ---------------------------------------------------------------------- running statistics
class RunningMeanStd:
    """Per-feature running mean / variance (parallel update), float64."""

    def __init__(self, dim: int, device):
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.var = torch.ones(dim, dtype=torch.float64, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).double()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x: torch.Tensor, clip: float) -> torch.Tensor:
        y = (x.double() - self.mean) / torch.sqrt(self.var + 1e-8)
        return y.clamp(-clip, clip).to(x.dtype)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var = d["mean"].to(self.mean.device), d["var"].to(self.var.device)
        self.count = d["count"]


class SliceNormalizer:
    """Training-only running statistics on the feature slice [lo, hi) (the channel block); bounded features
    (sin/cos, normalised powers, binary masks) are left untouched. Frozen (update=False) at evaluation."""

    def __init__(self, lo: int, hi: int, device, clip: float):
        self.lo, self.hi, self.clip = lo, hi, clip
        self.rms = RunningMeanStd(hi - lo, device)

    def __call__(self, x: torch.Tensor, update: bool) -> torch.Tensor:
        block = x[..., self.lo:self.hi]
        if update:
            self.rms.update(block)
        return torch.cat([x[..., :self.lo], self.rms.normalize(block, self.clip), x[..., self.hi:]], dim=-1)


class ValueNorm:
    """Running mean/std of value targets; the critic regresses normalised returns."""

    def __init__(self, device, enabled=True):
        self.enabled, self.rms = enabled, RunningMeanStd(1, device)

    def update(self, x):
        if self.enabled:
            self.rms.update(x.reshape(-1, 1))

    def normalize(self, x):
        if not self.enabled:
            return x
        return ((x.double() - self.rms.mean) / torch.sqrt(self.rms.var + 1e-8)).to(x.dtype)

    def denormalize(self, x):
        if not self.enabled:
            return x
        return (x.double() * torch.sqrt(self.rms.var + 1e-8) + self.rms.mean).to(x.dtype)


# ---------------------------------------------------------------------- logging
def atomic_write(path: str, write_fn) -> None:
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{os.getpid()}{ext}"
    write_fn(tmp)
    os.replace(tmp, path)


def atomic_json(path: str, obj) -> None:
    def w(fn):
        with open(fn, "w") as f:
            json.dump(obj, f, indent=2)
    atomic_write(path, w)


class CSVLogger:
    """Appends one row per call; new columns are added by rewriting the header once."""

    def __init__(self, path: str):
        self.path, self.fields = path, None
        if os.path.exists(path):
            os.remove(path)

    def log(self, row: dict) -> None:
        if self.fields is None or any(k not in self.fields for k in row):
            old = []
            if self.fields is not None and os.path.exists(self.path):
                with open(self.path, newline="") as f:
                    old = list(csv.DictReader(f))
            self.fields = list(dict.fromkeys((self.fields or []) + list(row)))
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
                w.writerows(old)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)
