"""Network simulation layer for Pocket Agent deployment architecture experiments.

Provides application-level simulation of network latency, jitter, packet loss,
and bandwidth constraints for comparing on-device vs hybrid vs cloud LLM
agent deployment under varying network conditions.
"""

from .transfer_event import TransferEvent
from .network_simulator import NetworkConfig, NetworkSimulator, NETWORK_PRESETS
from .radio_model import RadioStateModel
from .deployment_architectures import DeploymentArchitecture, HybridArchitectureWrapper

__all__ = [
    "TransferEvent",
    "NetworkConfig",
    "NetworkSimulator",
    "NETWORK_PRESETS",
    "RadioStateModel",
    "DeploymentArchitecture",
    "HybridArchitectureWrapper",
]
