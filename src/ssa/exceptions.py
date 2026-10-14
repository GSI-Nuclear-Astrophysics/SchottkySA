"""Named input and fitting errors, retaining their built-in exception bases."""

from __future__ import annotations


class SSAError(Exception):
    """Base class for all SchottkySA-specific errors. Not raised directly."""


class InputFormatError(ValueError, SSAError):
    """Malformed, too short, or shape-mismatched ``(x, y)`` input data."""


class TemplateConstructionError(ValueError, SSAError):
    """The empirical peak template could not be built from the given region."""


class FitConfigurationError(ValueError, SSAError):
    """Invalid fit configuration: bounds, ``mu_bounds``, ``min_separation``, ..."""


class FitConstraintViolationError(RuntimeError, SSAError):
    """The optimizer returned a result that violates a hard constraint
    (e.g. the requested minimum component separation)."""
