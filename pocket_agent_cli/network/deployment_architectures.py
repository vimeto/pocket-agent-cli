"""Deployment architecture definitions and hybrid wrapper.

Defines the three deployment modes studied in the Pocket Agent paper:
1. FULLY_LOCAL — inference and tools on device
2. HYBRID — inference on cloud, tools on device
3. FULLY_CLOUD — inference and tools on cloud

The HybridArchitectureWrapper intercepts inference calls to add simulated
network delays, modeling the cost of shipping prompts to and responses from
a cloud inference endpoint.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .network_simulator import NetworkSimulator


class DeploymentArchitecture(str, Enum):
    """The three deployment modes compared in the paper."""

    FULLY_LOCAL = "local"  # inference + tools on device
    HYBRID = "hybrid"  # inference on cloud, tools on device
    FULLY_CLOUD = "cloud"  # inference + tools on cloud


class HybridArchitectureWrapper:
    """Wraps an inference service to simulate hybrid deployment.

    In hybrid mode, each agentic iteration involves:
    1. Upload prompt/observation to cloud (simulated delay)
    2. Cloud runs inference (actual inference via the wrapped service)
    3. Download response from cloud (simulated delay)
    4. Execute tools locally (no network delay)
    5. Repeat

    This wrapper intercepts generate_with_tools() calls and adds
    network simulation around them, without modifying the underlying
    inference service.
    """

    def __init__(
        self,
        inference_service: Any,
        network_simulator: NetworkSimulator,
    ) -> None:
        """Initialize the hybrid wrapper.

        Args:
            inference_service: Any object with a generate_with_tools() method
                (e.g., InferenceService or MLXInferenceService).
            network_simulator: Configured NetworkSimulator instance.
        """
        self.service = inference_service
        self.network = network_simulator

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Wrapper that adds network delays around inference.

        Simulates uploading the prompt to a cloud endpoint, running inference,
        and downloading the response — all with the configured network conditions.

        Args:
            messages: List of chat messages.
            tools: List of tool definitions.
            **kwargs: Passed through to the underlying service.

        Returns:
            Tuple of (response_text, tool_calls, metrics) — same as the
            underlying service, but with network metrics added.
        """
        # Calculate upload payload size (estimate from messages)
        upload_bytes = self._estimate_payload(messages)

        # Simulate upload to cloud
        self.network.simulate_transfer_sync(upload_bytes, "upload")

        # Run actual inference (this is the real compute)
        response_text, tool_calls, metrics = self.service.generate_with_tools(
            messages, tools, **kwargs
        )

        # Simulate download from cloud
        download_bytes = len(response_text.encode("utf-8")) if response_text else 0
        # Include tool call data in download size
        if tool_calls:
            tool_calls_str = json.dumps(tool_calls)
            download_bytes += len(tool_calls_str.encode("utf-8"))
        self.network.simulate_transfer_sync(download_bytes, "download")

        # Add network metrics to the result
        metrics["network"] = self.network.get_summary()

        return response_text, tool_calls, metrics

    @staticmethod
    def _estimate_payload(messages: List[Dict[str, str]]) -> int:
        """Estimate the byte size of a message payload.

        This is a rough estimate — in practice, the actual serialized size
        depends on the API format, but for simulation purposes a UTF-8
        encoding of the JSON is close enough.

        Args:
            messages: List of chat messages.

        Returns:
            Estimated payload size in bytes.
        """
        payload_str = json.dumps(messages)
        return len(payload_str.encode("utf-8"))
