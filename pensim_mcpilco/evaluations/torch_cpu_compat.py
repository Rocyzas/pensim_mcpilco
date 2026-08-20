"""Makes GPU-trained run artifacts loadable on a CPU-only machine.

Runs trained on a CUDA box pickle tensors whose storages come back through
torch.storage._load_from_bytes, which calls torch.load() with NO map_location -- so every
unpickle of a log.pkl/monitor.pkl raises

    RuntimeError: Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False

on a laptop. Rather than route each call site through a custom Unpickler, patch that single
callable once: this also covers the plain `pkl.load(...)` calls inside vendored MC-PILCO
(MC_PILCO.load_policy_from_log / load_model_from_log), which we do not want to edit.

Import for side effect, before anything unpickles a run:

    import torch_cpu_compat  # noqa: F401

No-op when CUDA is available, so GPU runs keep their current behaviour.
"""
import io

import torch

_patched = False


def install():
    """Idempotently map CUDA storages to CPU during unpickling on CPU-only machines."""
    global _patched
    if _patched or torch.cuda.is_available():
        return
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    _patched = True


install()
