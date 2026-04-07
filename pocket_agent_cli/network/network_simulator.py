"""Core network simulation for hybrid/cloud deployment experiments.

Simulates network transfer delays at the application level, including:
- Round-trip time (RTT) with configurable base and jitter
- Bandwidth-limited transfer delay
- Packet loss with retransmission
- Cellular radio state transitions (via RadioStateModel)

All delays use actual sleep (time.sleep / asyncio.sleep) so they affect
wall-clock time and energy measurements in experiments.
"""

import asyncio
import random
import time
from typing import Dict, List

from pydantic import BaseModel, Field

from .radio_model import RadioStateModel
from .transfer_event import TransferEvent


class NetworkConfig(BaseModel):
    """Configuration for simulated network conditions.

    Each preset represents a realistic network scenario. The parameters are
    based on measurement studies of mobile network performance.
    """

    name: str  # e.g., "wifi", "5g", "4g", "poor_cellular"
    rtt_ms: float  # base round-trip time in milliseconds
    jitter_ms: float = 0.0  # RTT jitter (gaussian std dev in ms)
    packet_loss_rate: float = 0.0  # probability of loss (triggers retransmit)
    bandwidth_mbps: float = 1000.0  # bandwidth limit in Mbps
    radio_tail_energy_j: float = 0.0  # energy cost per radio state transition


NETWORK_PRESETS: Dict[str, NetworkConfig] = {
    "local": NetworkConfig(
        name="local",
        rtt_ms=0,
        jitter_ms=0,
        packet_loss_rate=0.0,
        bandwidth_mbps=10000.0,
        radio_tail_energy_j=0.0,
    ),
    "lan": NetworkConfig(
        name="lan",
        rtt_ms=1,
        jitter_ms=0.5,
        packet_loss_rate=0.0,
        bandwidth_mbps=1000.0,
        radio_tail_energy_j=0.0,
    ),
    "wifi": NetworkConfig(
        name="wifi",
        rtt_ms=20,
        jitter_ms=5,
        packet_loss_rate=0.001,
        bandwidth_mbps=50.0,
        radio_tail_energy_j=0.0,
    ),
    "5g": NetworkConfig(
        name="5g",
        rtt_ms=40,
        jitter_ms=15,
        packet_loss_rate=0.001,
        bandwidth_mbps=100.0,
        radio_tail_energy_j=0.3,
    ),
    "4g": NetworkConfig(
        name="4g",
        rtt_ms=80,
        jitter_ms=30,
        packet_loss_rate=0.005,
        bandwidth_mbps=20.0,
        radio_tail_energy_j=0.5,
    ),
    "poor_cellular": NetworkConfig(
        name="poor_cellular",
        rtt_ms=200,
        jitter_ms=100,
        packet_loss_rate=0.02,
        bandwidth_mbps=2.0,
        radio_tail_energy_j=0.8,
    ),
    "edge_case": NetworkConfig(
        name="edge_case",
        rtt_ms=500,
        jitter_ms=200,
        packet_loss_rate=0.05,
        bandwidth_mbps=0.5,
        radio_tail_energy_j=1.0,
    ),
    # ── Bandwidth-focused presets for bandwidth × RTT sweep ──────────
    # 4G RTT (~80ms) at different bandwidth levels
    "4g_1mbps": NetworkConfig(
        name="4g_1mbps",
        rtt_ms=80,
        jitter_ms=30,
        packet_loss_rate=0.005,
        bandwidth_mbps=1.0,
        radio_tail_energy_j=0.5,
    ),
    "4g_5mbps": NetworkConfig(
        name="4g_5mbps",
        rtt_ms=80,
        jitter_ms=30,
        packet_loss_rate=0.005,
        bandwidth_mbps=5.0,
        radio_tail_energy_j=0.5,
    ),
    "4g_20mbps": NetworkConfig(
        name="4g_20mbps",
        rtt_ms=80,
        jitter_ms=30,
        packet_loss_rate=0.005,
        bandwidth_mbps=20.0,
        radio_tail_energy_j=0.5,
    ),
    # WiFi RTT (~20ms) at different bandwidth levels
    "wifi_10mbps": NetworkConfig(
        name="wifi_10mbps",
        rtt_ms=20,
        jitter_ms=5,
        packet_loss_rate=0.001,
        bandwidth_mbps=10.0,
        radio_tail_energy_j=0.0,
    ),
    "wifi_50mbps": NetworkConfig(
        name="wifi_50mbps",
        rtt_ms=20,
        jitter_ms=5,
        packet_loss_rate=0.001,
        bandwidth_mbps=50.0,
        radio_tail_energy_j=0.0,
    ),
}


class NetworkSimulator:
    """Simulates network transfer delays for hybrid/cloud architectures.

    Each call to simulate_transfer() computes the total delay from RTT, jitter,
    bandwidth, and packet loss, then actually sleeps for that duration. All
    transfers are logged for post-experiment analysis.
    """

    def __init__(self, config: NetworkConfig, seed: int | None = None) -> None:
        """Initialize the simulator.

        Args:
            config: Network conditions to simulate.
            seed: Optional random seed for reproducibility.
        """
        self.config = config
        self.transfer_log: List[TransferEvent] = []
        self.radio_model = RadioStateModel()
        self._rng = random.Random(seed)

    def _compute_transfer(self, payload_bytes: int, direction: str) -> TransferEvent:
        """Compute transfer delay components without sleeping.

        Args:
            payload_bytes: Size of the payload in bytes.
            direction: "upload" or "download".

        Returns:
            A TransferEvent with all delay components.
        """
        timestamp = time.time()

        # 1. RTT with jitter (gaussian, clamped to non-negative)
        jitter = self._rng.gauss(0, self.config.jitter_ms) if self.config.jitter_ms > 0 else 0.0
        simulated_rtt_ms = max(0.0, self.config.rtt_ms + jitter)

        # 2. Bandwidth delay: time to transfer payload at given bandwidth
        # bandwidth_mbps -> bytes per ms = (bandwidth_mbps * 1e6) / 8 / 1000 = bandwidth_mbps * 125
        if self.config.bandwidth_mbps > 0:
            bytes_per_ms = self.config.bandwidth_mbps * 125.0  # Mbps to bytes/ms
            bandwidth_delay_ms = payload_bytes / bytes_per_ms
        else:
            bandwidth_delay_ms = 0.0

        # 3. Packet loss -> retransmission (adds one extra RTT)
        retransmit = self._rng.random() < self.config.packet_loss_rate
        retransmit_delay_ms = simulated_rtt_ms if retransmit else 0.0

        # 4. Total delay
        total_delay_ms = simulated_rtt_ms + bandwidth_delay_ms + retransmit_delay_ms

        # 5. Radio state transition
        transition_info = self.radio_model.on_transfer(timestamp, self.config)
        radio_state_transition = transition_info["triggered_transition"]

        event = TransferEvent(
            timestamp=timestamp,
            direction=direction,
            payload_bytes=payload_bytes,
            simulated_rtt_ms=simulated_rtt_ms,
            bandwidth_delay_ms=bandwidth_delay_ms,
            retransmit=retransmit,
            total_delay_ms=total_delay_ms,
            radio_state_transition=radio_state_transition,
        )
        self.transfer_log.append(event)
        return event

    async def simulate_transfer(
        self, payload_bytes: int, direction: str = "upload"
    ) -> TransferEvent:
        """Simulate a network transfer with latency, jitter, loss, and bandwidth.

        Actually sleeps for the simulated duration using asyncio.sleep.

        Args:
            payload_bytes: Size of the payload in bytes.
            direction: "upload" or "download".

        Returns:
            A TransferEvent with timing details.
        """
        event = self._compute_transfer(payload_bytes, direction)
        if event.total_delay_ms > 0:
            await asyncio.sleep(event.total_delay_ms / 1000.0)
        return event

    def simulate_transfer_sync(
        self, payload_bytes: int, direction: str = "upload"
    ) -> TransferEvent:
        """Synchronous version using time.sleep.

        Args:
            payload_bytes: Size of the payload in bytes.
            direction: "upload" or "download".

        Returns:
            A TransferEvent with timing details.
        """
        event = self._compute_transfer(payload_bytes, direction)
        if event.total_delay_ms > 0:
            time.sleep(event.total_delay_ms / 1000.0)
        return event

    def get_transfer_log(self) -> List[TransferEvent]:
        """Return all logged transfers for analysis."""
        return list(self.transfer_log)

    def get_summary(self) -> Dict:
        """Summary stats: total transfers, bytes, time, simulated radio events.

        Returns:
            Dict with aggregate statistics over all logged transfers.
        """
        if not self.transfer_log:
            return {
                "total_transfers": 0,
                "total_bytes": 0,
                "total_delay_ms": 0.0,
                "total_uploads": 0,
                "total_downloads": 0,
                "total_retransmits": 0,
                "mean_delay_ms": 0.0,
                "radio_transitions": 0,
                "radio_energy": self.radio_model.get_energy_summary(),
                "network_preset": self.config.name,
            }

        total_bytes = sum(e.payload_bytes for e in self.transfer_log)
        total_delay = sum(e.total_delay_ms for e in self.transfer_log)
        uploads = sum(1 for e in self.transfer_log if e.direction == "upload")
        downloads = sum(1 for e in self.transfer_log if e.direction == "download")
        retransmits = sum(1 for e in self.transfer_log if e.retransmit)

        return {
            "total_transfers": len(self.transfer_log),
            "total_bytes": total_bytes,
            "total_delay_ms": total_delay,
            "total_uploads": uploads,
            "total_downloads": downloads,
            "total_retransmits": retransmits,
            "mean_delay_ms": total_delay / len(self.transfer_log),
            "radio_transitions": self.radio_model.get_energy_summary()["total_transitions"],
            "radio_energy": self.radio_model.get_energy_summary(),
            "network_preset": self.config.name,
        }

    def reset(self) -> None:
        """Clear transfer log and reset radio model."""
        self.transfer_log = []
        self.radio_model.reset()
