import importlib


class PluginLoadError(Exception):
    pass


def import_object(spec: str) -> object:
    """Resolve ``package.module:attribute`` to an object."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise PluginLoadError(f"expected 'module:attribute', got {spec!r}")
    try:
        target: object = importlib.import_module(module_name)
        for part in attribute.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError) as exc:
        raise PluginLoadError(f"cannot import {spec!r}: {exc}") from exc
    return target


def materialize(target: object) -> object:
    """Instantiate classes and call zero-argument factories; return instances unchanged."""
    if isinstance(target, type):
        return target()
    if callable(target) and not hasattr(target, "name"):
        return target()
    return target
