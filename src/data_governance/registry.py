"""Registry of data-source connector *classes* (not instances).

Connectors register themselves at import time:

    from src.data_governance.registry import register

    @register
    class MyConnector(DataSourceConnector):
        META = ConnectorMeta(name="my", host="...", ...)
"""
from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type] = {}


def register(cls):
    """Class decorator. Idempotent."""
    name = getattr(cls, "META", None) and cls.META.name
    if not name:
        raise ValueError(f"{cls.__name__} has no META.name; cannot register")
    if name in REGISTRY:
        logger.debug("connector %s already registered", name)
        return cls
    REGISTRY[name] = cls
    return cls


def list_sources() -> list[dict]:
    """Return a JSON-friendly snapshot of every registered connector."""
    out = []
    for name, cls in REGISTRY.items():
        m = cls.META
        out.append({
            "name":          m.name,
            "host":          m.host,
            "priority":      m.priority,
            "requires_auth": m.requires_auth,
            "category":      m.category,
            "description":   m.description,
        })
    return sorted(out, key=lambda x: (x["priority"], x["name"]))


__all__ = ["REGISTRY", "register", "list_sources"]
