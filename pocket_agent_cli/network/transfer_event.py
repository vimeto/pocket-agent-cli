"""Data model for network transfer events."""

from pydantic import BaseModel


class TransferEvent(BaseModel):
    """Record of a single simulated network transfer.

    Each transfer through the NetworkSimulator produces one of these events,
    which is logged for later analysis. The fields capture all components of
    the simulated delay so we can decompose latency in our paper figures.
    """

    timestamp: float
    direction: str  # "upload" or "download"
    payload_bytes: int
    simulated_rtt_ms: float  # actual RTT used (base + jitter)
    bandwidth_delay_ms: float  # delay from bandwidth limit
    retransmit: bool  # whether packet loss triggered a retransmit
    total_delay_ms: float  # total simulated delay
    radio_state_transition: bool  # whether this triggered a new radio state
