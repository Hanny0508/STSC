"""SIP signaling status for the paper artifact.

The validated two-host experiment uses RTP/UDP with Opus over Tailscale.  It
does not implement SIP signaling, so this module fails explicitly instead of
silently labeling an RTP command as a SIP call.
"""


class SIPNotImplementedError(NotImplementedError):
    pass


def create_sip_call(*_args, **_kwargs):
    raise SIPNotImplementedError(
        "SIP signaling is not part of the validated ZSC-TSD artifact; "
        "use zsc_tsd.transport.rtp_opus for the RTP/Opus media channel"
    )
