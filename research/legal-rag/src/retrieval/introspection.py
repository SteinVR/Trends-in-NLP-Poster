"""Shared lightweight introspection helpers for retrieval modules."""

from __future__ import annotations

import inspect
from typing import Any


def callable_accepts_keyword(func: Any, keyword: str) -> bool:
    """Return whether a callable accepts a keyword argument.

    Args:
        func: Callable object to inspect.
        keyword: Keyword name to check.

    Returns:
        ``True`` when the callable accepts ``keyword`` explicitly or via ``**kwargs``.
    """

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def metadata_field(container: Any, field_name: str, default: Any) -> Any:
    """Read one metadata field from dict-like or object-like containers.

    Args:
        container: Source object or mapping that may contain the field.
        field_name: Metadata field to fetch.
        default: Value to return when field is absent.

    Returns:
        Field value when present, otherwise ``default``.
    """

    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(field_name, default)
    return getattr(container, field_name, default)
