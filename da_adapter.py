"""Compatibility re-exports for legacy imports."""

from dem.models.da_adapter import DAAdapter, DAAdapterConfig, TokenResidualMLP, l2_normalize

__all__ = [
    "DAAdapter",
    "DAAdapterConfig",
    "TokenResidualMLP",
    "l2_normalize",
]
