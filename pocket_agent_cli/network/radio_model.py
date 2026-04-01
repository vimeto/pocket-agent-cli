"""Simple cellular radio RRC state model for energy analysis.

Models the Radio Resource Control (RRC) state machine used in LTE/5G networks.
This is important for understanding the energy cost of agentic LLM traffic,
which has long gaps (10-30s of on-device compute) between short bursts
(1-5KB tool call exchanges), causing repeated IDLE->CONNECTED transitions
with tail energy waste.

References:
- 3GPP TS 36.331 (RRC protocol specification)
- Huang et al., "A Close Examination of Performance and Power Characteristics
  of 4G LTE Networks", MobiSys 2012
"""

import time
from typing import Dict, List

from .transfer_event import TransferEvent


class RadioStateModel:
    """Models cellular radio RRC state transitions for energy analysis.

    States: IDLE -> CONNECTED -> TAIL -> IDLE
    - IDLE: low power, no radio activity
    - CONNECTED: high power, actively transferring
    - TAIL: medium power, radio stays active after transfer (typically 5-12s)

    The tail timer is the key parameter: after each transfer, the radio stays
    in a high-power state for TAIL_TIMER_S seconds before returning to IDLE.
    If a new transfer arrives during the tail period, no state transition
    occurs (saving transition energy). If the gap exceeds the tail timer,
    the radio must transition from IDLE to CONNECTED again.
    """

    TAIL_TIMER_S: float = 7.0  # seconds radio stays in high-power state after last transfer

    def __init__(self) -> None:
        self.state: str = "IDLE"
        self.last_transfer_time: float = 0.0
        self.state_transitions: List[Dict] = []
        self.total_tail_energy_j: float = 0.0

    def on_transfer(self, timestamp: float, config: "NetworkConfig") -> Dict:
        """Update state on a new transfer. Returns state transition info.

        Args:
            timestamp: Current wall-clock time (or simulated time).
            config: Network config with radio_tail_energy_j parameter.

        Returns:
            Dict with transition details: from_state, to_state, triggered_transition,
            tail_energy_wasted_j.
        """
        from .network_simulator import NetworkConfig  # avoid circular import

        previous_state = self.state
        triggered_transition = False
        tail_energy_wasted_j = 0.0

        if self.state == "IDLE":
            # Must transition to CONNECTED — costs energy
            self.state = "CONNECTED"
            triggered_transition = True
        elif self.state == "CONNECTED":
            # Already connected, no transition needed
            pass
        elif self.state == "TAIL":
            # Check if tail timer has expired
            time_since_last = timestamp - self.last_transfer_time
            if time_since_last > self.TAIL_TIMER_S:
                # Tail expired, was in IDLE, must reconnect
                # The tail energy from the previous session was wasted
                tail_energy_wasted_j = config.radio_tail_energy_j
                self.total_tail_energy_j += tail_energy_wasted_j
                self.state = "CONNECTED"
                triggered_transition = True
            else:
                # Still within tail period — stays connected, no extra cost
                self.state = "CONNECTED"
                triggered_transition = False

        self.last_transfer_time = timestamp

        transition_info = {
            "timestamp": timestamp,
            "from_state": previous_state,
            "to_state": self.state,
            "triggered_transition": triggered_transition,
            "tail_energy_wasted_j": tail_energy_wasted_j,
        }
        self.state_transitions.append(transition_info)

        return transition_info

    def update_idle_check(self, current_time: float) -> None:
        """Check if the radio should transition to TAIL or IDLE based on elapsed time.

        Call this to advance the state machine between transfers.

        Args:
            current_time: Current wall-clock time.
        """
        if self.state == "CONNECTED":
            time_since_last = current_time - self.last_transfer_time
            if time_since_last > 0:
                # After a transfer completes, radio enters TAIL state
                self.state = "TAIL"

        if self.state == "TAIL":
            time_since_last = current_time - self.last_transfer_time
            if time_since_last > self.TAIL_TIMER_S:
                self.state = "IDLE"

    def get_energy_summary(self) -> Dict:
        """Compute total radio energy from state transitions.

        Returns:
            Dict with total_transitions, total_tail_energy_j, and transitions list.
        """
        num_transitions = sum(
            1 for t in self.state_transitions if t["triggered_transition"]
        )
        return {
            "total_transitions": num_transitions,
            "total_tail_energy_j": self.total_tail_energy_j,
            "transition_count": len(self.state_transitions),
            "transitions": self.state_transitions,
        }

    def reset(self) -> None:
        """Reset the radio model to initial state."""
        self.state = "IDLE"
        self.last_transfer_time = 0.0
        self.state_transitions = []
        self.total_tail_energy_j = 0.0
