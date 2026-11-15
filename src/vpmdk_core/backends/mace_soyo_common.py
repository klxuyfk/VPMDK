"""Portable cuEquivariance construction for direct MACE-SOYO inference."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator


@contextmanager
def _naive_segmented_polynomials(polynomial_cls: Any) -> Iterator[None]:
    """Select cuEquivariance's public PyTorch method during model construction.

    Upstream MACE-SOYO hardcodes uniform_1d in some layers. Scope the override
    to explicit portable model construction; weights and installed sources stay intact.
    """
    original = polynomial_cls.__init__

    @wraps(original)
    def initialize(self: Any, polynomial: Any, *args: Any, **kwargs: Any) -> None:
        # SOYO's convolution wrapper reads uniform_1d shape metadata at
        # construction. Retain that metadata while replacing the evaluator.
        original(self, polynomial, *args, **kwargs)
        original_keys = set(self.state_dict()) if hasattr(self, "state_dict") else set()
        evaluator = getattr(self, "m", None)
        metadata = {
            name: getattr(evaluator, name)
            for name in ("buffer_num_segments", "operand_extent")
            if hasattr(evaluator, name)
        }
        if args:
            args = ("naive", *args[1:])
        else:
            kwargs["method"] = "naive"
        original(self, polynomial, *args, **kwargs)
        for name, value in metadata.items():
            setattr(self.m, name, value)
        # Naive evaluators derive polynomial constants from the descriptor.
        # New constants must not become missing learned checkpoint entries;
        # retain any buffers that were already part of the original evaluator.
        if hasattr(self.m, "named_modules"):
            for prefix, module in self.m.named_modules():
                for name in module._buffers:
                    key = ".".join(part for part in ("m", prefix, name) if part)
                    if key not in original_keys:
                        module._non_persistent_buffers_set.add(name)

    polynomial_cls.__init__ = initialize
    try:
        yield
    finally:
        polynomial_cls.__init__ = original


@contextmanager
def _legacy_cueq_linear_aliases(linear_cls: Any) -> Iterator[None]:
    """Remove only duplicate 0.6.1 registrations, retaining all model tensors.

    Older Linear registers the same evaluator under both m and fallback.
    Modern checkpoints contain the m keys only. Keep fallback as a Python
    alias so evaluation is unchanged and checkpoint loading stays strict.
    """
    original = linear_cls.__init__

    @wraps(original)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        for module in self.modules():
            fallback = module._modules.get("fallback")
            if fallback is not None and fallback is module._modules.get("m"):
                module._modules.pop("fallback")
                object.__setattr__(module, "fallback", fallback)

    linear_cls.__init__ = initialize
    try:
        yield
    finally:
        linear_cls.__init__ = original


@contextmanager
def _cueq_checkpoint_compat() -> Iterator[None]:
    """Apply the experimentally validated 0.6.1 compatibility rule only."""
    import importlib.metadata

    if importlib.metadata.version("cuequivariance-torch") == "0.6.1":
        import cuequivariance_torch as cuet

        with _legacy_cueq_linear_aliases(cuet.Linear):
            yield
    else:
        yield
