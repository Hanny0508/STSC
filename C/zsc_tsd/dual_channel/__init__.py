"""Two-host authenticated RTP/Opus experiment orchestration."""

from .protocol import (
    DualChannelError,
    decode_checks,
    read_control_environment,
)

__all__ = ["DualChannelError", "decode_checks", "read_control_environment"]
