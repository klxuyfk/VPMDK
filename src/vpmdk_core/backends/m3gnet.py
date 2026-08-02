"""M3GNet and MACE backend builders."""

from __future__ import annotations

import sys
from typing import Any, Dict

from ..backend_common import ModelReference, ModelReferenceKind


def _root() -> Any:
    return sys.modules["vpmdk_core"]


def _matgl_calculator_requires_potential(calculator_cls: Any) -> bool:
    """Return whether a MatGL ASE calculator requires a potential argument."""

    root = _root()
    try:
        parameter = root.inspect.signature(calculator_cls).parameters.get("potential")
    except (TypeError, ValueError):
        parameter = None
    if parameter is not None:
        return parameter.default is root.inspect.Parameter.empty
    return getattr(calculator_cls, "__name__", "") == "PESCalculator"


def _move_module_to_device(module: Any, device: str | None) -> Any:
    """Relocate a loaded torch module onto ``device``, returning the moved object.

    Shared by the MatGL potential and MACE model movers. Some torch calculators
    (MatGL's ``PESCalculator``) take no ``device`` argument and do not move the
    model, so a ``DEVICE=cuda`` request would otherwise run on whatever device
    the checkpoint loaded onto while server status reports cuda. Moving the module
    keeps placement consistent with the reported device (and fails loudly if the
    device is unavailable, matching the other torch backends). ``.to`` returns
    ``self`` for a torch module; tolerate wrappers that return ``None`` and let a
    non-in-place ``.to`` (a distinct returned object) be repointed by the caller.
    """

    if device is None or not str(device).strip():
        # A present-but-BLANK ``DEVICE =`` resolves to "" (_resolve_device only
        # autodetects for None), and ``module.to("")`` raises
        # "Device string must not be empty". Before this relocation helper
        # existed, a blank device was simply absorbed by ASE's
        # ``Calculator.__init__(**kwargs)`` and ignored, so the calculator built
        # fine; raising here would newly abort `vpmdk serve` AFTER the model is
        # loaded (and break the one-shot run too). Treat it like an omitted
        # device: leave placement alone.
        #
        # (server.py's _DEVICE_BLANK_TO_CPU_IDENTITIES does list MATGL, so a
        # blank DEVICE is advertised as "cpu" in the resident's configuration --
        # that is about what `vpmdk status` reports and how §3.4 compares tags,
        # not about where this module is placed, which is what the guard above
        # decides. An earlier version of this comment claimed the opposite.)
        return module
    mover = getattr(module, "to", None)
    if not callable(mover):
        return module
    moved = mover(device)
    return moved if moved is not None else module


def _repoint_calculator_module(calculator: Any, attr: str, device: str | None) -> None:
    """Move ``calculator.<attr>`` onto ``device``, repointing it if ``.to`` was not in place.

    Shared by the MatGL potential (``.potential``) and MACE single-model
    (``.model``) branches. torch modules move in place, but a wrapper whose
    ``.to`` returns a new object must be written back or the calculator would keep
    the original unmoved module while status reports the requested device. A
    read-only attribute degrades to a no-op.
    """

    module = getattr(calculator, attr, None)
    if module is None:
        return
    moved = _move_module_to_device(module, device)
    if moved is not module:
        try:
            setattr(calculator, attr, moved)
        except AttributeError:
            pass


def _move_matgl_calculator_potential_to_device(calculator: Any, device: str | None) -> Any:
    """Move a path-built MatGL calculator's internal potential onto ``device``.

    The local-path compat fallbacks construct the calculator directly from a
    checkpoint path, so the loud pre-construction potential move never touched
    the potential the calculator loaded internally. Relocate that potential too
    (older MatGL calculators expose it as ``.potential``; absence is a safe
    no-op) so device placement stays consistent with reported status across
    every construction branch, and an unavailable device still fails loudly.
    """

    if device is None:
        return calculator
    _repoint_calculator_module(calculator, "potential", device)
    return calculator


def _load_matgl_potential(model_identifier: str) -> Any:
    """Load one MatGL potential while retaining the upstream failure cause.

    Device placement is applied separately by the caller (not here) so a
    device-move failure is never swallowed by the local-path load fallback.
    """

    root = _root()
    if root.MatGLLoadModel is None:
        raise RuntimeError(
            "This MatGL calculator requires a potential, but matgl.load_model "
            "is unavailable."
        )
    try:
        potential = root.MatGLLoadModel(model_identifier)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load MatGL model from {model_identifier}: {exc}"
        ) from exc
    return root._require_loaded_model(
        potential,
        backend_name="MatGL",
        model=model_identifier,
    )


_STRESS_UNIT_KEY = "stress_unit"
_MATGL_ASE_STRESS_UNIT = "eV/A3"


def _matgl_accepts_stress_unit(calculator_cls: Any) -> bool:
    """Whether ``stress_unit`` verifiably reaches a DECLARED parameter.

    matgl's legacy ``M3GNetCalculator.__init__`` takes ``**kwargs`` and forwards
    them to ``PESCalculator.__init__``, which is where ``stress_unit`` is
    declared -- so requiring the class itself to declare it is too strict and
    would leave the GPa default in place. Accepting a bare ``**kwargs`` is too
    loose for the opposite reason: an older signature would silently swallow the
    pin. So follow the MRO and pin only when some base really declares it.
    """

    root = _root()
    if root._callable_declares_parameter(calculator_cls, _STRESS_UNIT_KEY):
        return True
    if not root._callable_supports_parameter(calculator_cls, _STRESS_UNIT_KEY):
        return False
    for base in getattr(calculator_cls, "__mro__", ())[1:]:
        initializer = base.__dict__.get("__init__")
        if initializer is not None and root._callable_declares_parameter(
            initializer, _STRESS_UNIT_KEY
        ):
            return True
    return False


def _construct_matgl_calculator(model: Any, kwargs: Dict[str, Any]) -> Any:
    """Build matgl's ASE calculator with stresses pinned to eV/A^3.

    matgl >= 4 defaults ``stress_unit="GPa"`` while VPMDK (like ASE) reads
    ``atoms.get_stress()`` as eV/A^3, so leaving the default in place scales
    every reported stress and pressure by 1/ase.units.GPa (~160.2x).
    """

    root = _root()
    calculator_cls = root.M3GNetCalculator
    call_kwargs = dict(kwargs)
    if _matgl_accepts_stress_unit(calculator_cls):
        call_kwargs[_STRESS_UNIT_KEY] = _MATGL_ASE_STRESS_UNIT
    try:
        return calculator_cls(model, **call_kwargs)
    except TypeError:
        if not call_kwargs:
            raise
        pinned = (
            {_STRESS_UNIT_KEY: call_kwargs[_STRESS_UNIT_KEY]}
            if _STRESS_UNIT_KEY in call_kwargs
            else {}
        )
        if pinned == call_kwargs:
            # Only the pin was passed and it still raised: the TypeError is
            # not about optional extras, so retrying cannot help.
            raise
        # Retry dropping only the OTHER keyword arguments (e.g. a device the
        # installed signature rejects). Dropping the whole dict here silently
        # discarded the stress_unit pin too, and the calculator fell back to
        # matgl's GPa default: every stress and pressure ~160.2x too large,
        # with exit 0. The pin is only ever added when the MRO verifiably
        # declares it, so if this retry STILL raises the declaration probe
        # was wrong -- fail loudly rather than compute silently wrong
        # numbers (the bare-constructor fallback stays available for
        # signatures that never declared stress_unit).
        if pinned:
            return calculator_cls(model, **pinned)
        return calculator_cls(model)


def _construct_matgl_identifier(
    model_reference: ModelReference,
    kwargs: Dict[str, Any],
    device: str | None = None,
) -> Any:
    """Load an explicit MatGL model and preserve compatible path fallbacks."""

    root = _root()
    model_identifier = str(model_reference.value)
    if root.MatGLLoadModel is None:
        if model_reference.kind is ModelReferenceKind.LOCAL_PATH:
            return _move_matgl_calculator_potential_to_device(
                _construct_matgl_calculator(model_identifier, kwargs), device
            )
        raise RuntimeError(
            f"Unable to load MatGL MODEL {model_identifier!r}: "
            "matgl.load_model is unavailable"
        )

    try:
        potential = _load_matgl_potential(model_identifier)
    except RuntimeError as exc:
        return _construct_local_matgl_fallback(model_reference, kwargs, exc, device)

    # Move to DEVICE *outside* the load fallback: a device-placement failure
    # (e.g. DEVICE=cuda on a CPU-only host) must fail loudly, not be rerouted
    # into the local-path fallback, which would rebuild the calculator on CPU
    # while status still reports the requested device.
    potential = _move_module_to_device(potential, device)

    try:
        # Retry this same loaded potential without optional keywords before
        # considering any direct-path compatibility fallback.
        return _construct_matgl_calculator(potential, kwargs)
    except Exception as exc:
        return _construct_local_matgl_fallback(model_reference, kwargs, exc, device)


def _construct_local_matgl_fallback(
    model_reference: ModelReference,
    kwargs: Dict[str, Any],
    primary_error: Exception,
    device: str | None = None,
) -> Any:
    """Try a direct local-path constructor without masking the first error."""

    if model_reference.kind is not ModelReferenceKind.LOCAL_PATH:
        # Registry names are opaque identifiers. Never replace one with the
        # VPMDK default or pass it through a local-checkpoint-only recovery.
        raise primary_error
    model_identifier = str(model_reference.value)
    try:
        calculator = _construct_matgl_calculator(model_identifier, kwargs)
    except Exception:
        # Some MatGL releases accept paths that matgl.load_model() does not,
        # and vice versa. Prefer the loaded-potential path's diagnostic when
        # neither API can construct the requested model.
        raise primary_error
    # The path constructor may have dropped the device kwarg (TypeError retry);
    # move its internally-loaded potential so this branch keeps the same
    # fail-loud device guarantee as the loaded-potential path.
    return _move_matgl_calculator_potential_to_device(calculator, device)


def _move_mace_models_to_device(calculator: Any, device: str | None) -> Any:
    """Move a MACE calculator's loaded model(s) onto ``device`` (best-effort).

    Used only when MACECalculator has no ``device`` parameter, so a requested
    DEVICE is honored rather than silently left on the default device. MACE
    exposes loaded models as ``.models`` (a list) or ``.model``; absence is a
    safe no-op, and an unavailable device propagates loudly from ``.to``.
    """

    if device is None:
        return calculator
    models = getattr(calculator, "models", None)
    if models is not None:
        for index, model in enumerate(list(models)):
            moved = _move_module_to_device(model, device)
            if moved is not model:
                # Repoint a non-in-place .to result, as the matgl sibling does.
                try:
                    models[index] = moved
                except (TypeError, IndexError):
                    pass
        return calculator
    _repoint_calculator_module(calculator, "model", device)
    return calculator


def _build_mace_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create a MACE calculator with optional ``MODEL`` override."""

    root = _root()
    if root.MACECalculator is None:
        raise RuntimeError("MACECalculator not available. Install mace-torch and dependencies.")

    model_reference = root._resolve_backend_model_reference(
        "MACE", bcar_tags.get("MODEL")
    )
    device = root._resolve_device(bcar_tags.get("DEVICE"))

    kwargs = {}
    # Deliberately the LOOSE check: unlike TACE's fidelity selector (physics) or
    # MatRIS's converter (which has an explicit override fallback that an early
    # return would skip), forwarding DEVICE through a ``**kwargs`` signature is a
    # supported wrapper pattern here -- test_mace_existing_explicit_model_is_forwarded
    # pins it -- and a build that declares neither `device` nor ``**kwargs`` still
    # gets the explicit _move_mace_models_to_device relocation below.
    device_forwarded = device is not None and root._callable_supports_parameter(
        root.MACECalculator, "device"
    )
    if device_forwarded:
        kwargs["device"] = device
    if model_reference.kind is ModelReferenceKind.LOCAL_PATH:
        calculator = root.MACECalculator(model_reference.value, **kwargs)
    else:
        calculator = root.MACECalculator(**kwargs)
    if device is not None and not device_forwarded:
        # MACECalculator has no device parameter on this build; relocate its
        # loaded model(s) so placement matches the reported device.
        _move_mace_models_to_device(calculator, device)
    return calculator


def _load_legacy_m3gnet_potential(model_path: str | None) -> Any:
    """Load the requested legacy potential without substituting its default."""

    root = _root()
    potential_cls = root.LegacyM3GNetPotential
    model_cls = root.LegacyM3GNet
    if potential_cls is None:
        raise RuntimeError("Legacy M3GNet Potential is unavailable.")

    if model_path:
        load_errors: list[Exception] = []
        from_checkpoint = getattr(potential_cls, "from_checkpoint", None)
        if callable(from_checkpoint):
            try:
                return root._require_loaded_model(
                    from_checkpoint(model_path),
                    backend_name="legacy M3GNet",
                    model=model_path,
                )
            except Exception as exc:
                load_errors.append(exc)
        if model_cls is not None:
            try:
                model = root._require_loaded_model(
                    model_cls.load(model_path),
                    backend_name="legacy M3GNet",
                    model=model_path,
                )
                return root._require_loaded_model(
                    potential_cls(model),
                    backend_name="legacy M3GNet potential",
                    model=model_path,
                )
            except Exception as exc:
                load_errors.append(exc)
        error = RuntimeError(
            f"Unable to load requested legacy M3GNet MODEL: {model_path}"
        )
        if load_errors:
            raise error from load_errors[-1]
        raise error

    if model_cls is None:
        raise RuntimeError("Legacy M3GNet model loader is unavailable.")
    model = root._require_loaded_model(
        model_cls.load(),
        backend_name="legacy M3GNet",
        model=None,
    )
    return root._require_loaded_model(
        potential_cls(model),
        backend_name="legacy M3GNet potential",
        model=None,
    )


def _build_m3gnet_calculator(bcar_tags: Dict[str, str]):
    """Create a MatGL or legacy M3GNet calculator based on availability."""

    root = _root()
    if root.M3GNetCalculator is None:
        raise RuntimeError("M3GNetCalculator not available. Install matgl or m3gnet.")

    device = root._resolve_device(bcar_tags.get("DEVICE"))

    if not root._USING_LEGACY_M3GNET:
        model_reference = root._resolve_backend_model_reference(
            "MATGL", bcar_tags.get("MODEL")
        )
        kwargs = {"device": device} if device is not None else {}
        if model_reference.explicit:
            return _construct_matgl_identifier(model_reference, kwargs, device)
        if _matgl_calculator_requires_potential(root.M3GNetCalculator):
            potential = _load_matgl_potential(str(model_reference.value))
            potential = _move_module_to_device(potential, device)
            return _construct_matgl_calculator(potential, kwargs)
        try:
            calculator = root.M3GNetCalculator(**kwargs)
        except TypeError:
            calculator = root.M3GNetCalculator()
        # This branch's calculator loads its own default potential; move it so a
        # requested DEVICE is honored (and fails loudly if unavailable) rather
        # than silently left on the loader's default device.
        return _move_matgl_calculator_potential_to_device(calculator, device)

    model_reference = root._resolve_backend_model_reference(
        "M3GNET", bcar_tags.get("MODEL")
    )
    potential = _load_legacy_m3gnet_potential(model_reference.value)

    if device is not None:
        try:
            return root.M3GNetCalculator(potential=potential, device=device)
        except TypeError:
            pass

    return root.M3GNetCalculator(potential=potential)
