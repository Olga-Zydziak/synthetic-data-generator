"""Custom exception hierarchy for fraudforge."""

from __future__ import annotations

__all__ = [
    "FraudForgeError",
    "ConfigurationError",
    "GenerationError",
    "WriterError",
    "SynthesizerError",
]


class FraudForgeError(Exception):
    """Base class for all fraudforge errors."""


class ConfigurationError(FraudForgeError):
    """Raised when configuration validation fails."""


class GenerationError(FraudForgeError):
    """Raised when generation logic fails."""


class WriterError(FraudForgeError):
    """Raised when streaming writers encounter an issue."""


class SynthesizerError(FraudForgeError):
    """Raised for optional synthesizer backend errors."""


class MissingExtraError(SynthesizerError):
    """Raised when an optional dependency is unavailable."""

    def __init__(self, extra: str) -> None:
        message = (
            f"Optional dependency for '{extra}' synthesizer is not installed. "
            f"Install: pip install fraudforge[{extra}]"
        )
        super().__init__(message)
        self.extra = extra


__all__.append("MissingExtraError")
