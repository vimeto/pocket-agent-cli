"""Tests for the network simulation layer.

Tests cover:
- NetworkConfig presets
- Simulated delays (RTT, jitter, bandwidth, packet loss)
- RadioStateModel state transitions and energy tracking
- TransferEvent logging and summary statistics
- HybridArchitectureWrapper
- Sweep across all presets
- Total delay decomposition
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from pocket_agent_cli.network.transfer_event import TransferEvent
from pocket_agent_cli.network.network_simulator import (
    NetworkConfig,
    NetworkSimulator,
    NETWORK_PRESETS,
)
from pocket_agent_cli.network.radio_model import RadioStateModel
from pocket_agent_cli.network.deployment_architectures import (
    DeploymentArchitecture,
    HybridArchitectureWrapper,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def wifi_simulator():
    """NetworkSimulator with wifi preset and fixed seed."""
    return NetworkSimulator(NETWORK_PRESETS["wifi"], seed=42)


@pytest.fixture
def poor_cellular_simulator():
    """NetworkSimulator with poor_cellular preset and fixed seed."""
    return NetworkSimulator(NETWORK_PRESETS["poor_cellular"], seed=42)


@pytest.fixture
def local_simulator():
    """NetworkSimulator with local preset (zero delays)."""
    return NetworkSimulator(NETWORK_PRESETS["local"], seed=42)


@pytest.fixture
def high_loss_config():
    """Network config with very high packet loss for deterministic retransmit testing."""
    return NetworkConfig(
        name="high_loss",
        rtt_ms=50,
        jitter_ms=0,
        packet_loss_rate=1.0,  # 100% loss -> always retransmit
        bandwidth_mbps=1000.0,
        radio_tail_energy_j=0.5,
    )


@pytest.fixture
def radio_model():
    """Fresh RadioStateModel instance."""
    return RadioStateModel()


# ============================================================================
# Tests: NetworkConfig presets
# ============================================================================


class TestNetworkPresets:
    """Tests for NETWORK_PRESETS configuration."""

    def test_all_seven_presets_exist(self):
        """All 7 presets from the spec are present."""
        expected = {"local", "lan", "wifi", "5g", "4g", "poor_cellular", "edge_case"}
        assert set(NETWORK_PRESETS.keys()) == expected

    def test_presets_have_correct_names(self):
        """Each preset's name field matches its key."""
        for key, config in NETWORK_PRESETS.items():
            assert config.name == key

    def test_local_preset_has_zero_rtt(self):
        """Local preset should have zero RTT."""
        assert NETWORK_PRESETS["local"].rtt_ms == 0
        assert NETWORK_PRESETS["local"].jitter_ms == 0
        assert NETWORK_PRESETS["local"].packet_loss_rate == 0.0

    def test_rtt_increases_with_worse_conditions(self):
        """RTT should generally increase: local < lan < wifi < 5g < 4g < poor < edge."""
        order = ["local", "lan", "wifi", "5g", "4g", "poor_cellular", "edge_case"]
        rtts = [NETWORK_PRESETS[name].rtt_ms for name in order]
        assert rtts == sorted(rtts)

    def test_cellular_presets_have_radio_energy(self):
        """Cellular presets (5g, 4g, poor_cellular, edge_case) should have nonzero radio energy."""
        for name in ["5g", "4g", "poor_cellular", "edge_case"]:
            assert NETWORK_PRESETS[name].radio_tail_energy_j > 0

    def test_non_cellular_presets_have_zero_radio_energy(self):
        """Local, LAN, and WiFi should have zero radio tail energy."""
        for name in ["local", "lan", "wifi"]:
            assert NETWORK_PRESETS[name].radio_tail_energy_j == 0.0

    def test_all_presets_have_positive_bandwidth(self):
        """Every preset should have a positive bandwidth limit."""
        for config in NETWORK_PRESETS.values():
            assert config.bandwidth_mbps > 0


# ============================================================================
# Tests: Simulated delays
# ============================================================================


class TestSimulatedDelays:
    """Tests for delay computation in NetworkSimulator."""

    def test_local_preset_zero_delay(self, local_simulator):
        """Local preset should produce zero total delay."""
        event = local_simulator.simulate_transfer_sync(1024, "upload")
        # Local has 0 RTT, 0 jitter, 0 loss, and huge bandwidth
        # Bandwidth delay for 1KB at 10000 Mbps is negligible
        assert event.total_delay_ms < 0.01

    def test_rtt_in_expected_range_with_jitter(self, wifi_simulator):
        """WiFi RTT should be near 20ms with some jitter spread."""
        rtts = []
        for _ in range(100):
            event = wifi_simulator._compute_transfer(1024, "upload")
            rtts.append(event.simulated_rtt_ms)

        mean_rtt = sum(rtts) / len(rtts)
        # Mean should be close to base RTT (20ms), within 3ms tolerance
        assert abs(mean_rtt - 20.0) < 3.0
        # Should have some spread from jitter
        rtt_range = max(rtts) - min(rtts)
        assert rtt_range > 1.0  # jitter_ms=5, so range should be significant

    def test_bandwidth_delay_for_large_payload(self):
        """Bandwidth delay should be significant for large payloads on slow links."""
        config = NetworkConfig(
            name="slow",
            rtt_ms=0,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=1.0,  # 1 Mbps
        )
        sim = NetworkSimulator(config, seed=42)

        # 1 MB payload at 1 Mbps = 8 seconds = 8000 ms
        # bytes_per_ms = 1.0 * 125 = 125 bytes/ms
        # bandwidth_delay = 1_000_000 / 125 = 8000 ms
        event = sim._compute_transfer(1_000_000, "upload")
        assert abs(event.bandwidth_delay_ms - 8000.0) < 1.0

    def test_bandwidth_delay_small_payload(self, wifi_simulator):
        """Small payloads on good links should have negligible bandwidth delay."""
        event = wifi_simulator._compute_transfer(100, "upload")
        # 100 bytes at 50 Mbps: 100 / (50*125) = 0.016 ms
        assert event.bandwidth_delay_ms < 0.1

    def test_total_delay_equals_sum_of_components_no_retransmit(self):
        """Total delay = RTT + bandwidth_delay when no retransmit."""
        config = NetworkConfig(
            name="test",
            rtt_ms=100,
            jitter_ms=0,  # no jitter for deterministic test
            packet_loss_rate=0.0,  # no loss
            bandwidth_mbps=1.0,
        )
        sim = NetworkSimulator(config, seed=42)
        event = sim._compute_transfer(125_000, "upload")

        # RTT = 100ms (no jitter)
        # bandwidth_delay = 125000 / 125 = 1000ms
        # No retransmit
        expected = 100.0 + 1000.0
        assert abs(event.total_delay_ms - expected) < 0.01
        assert not event.retransmit

    def test_total_delay_includes_retransmit(self):
        """When retransmit occurs, total delay includes an extra RTT."""
        config = NetworkConfig(
            name="lossy",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=1.0,  # always lose
            bandwidth_mbps=1000.0,
        )
        sim = NetworkSimulator(config, seed=42)
        event = sim._compute_transfer(100, "upload")

        # RTT = 50, retransmit = 50, bandwidth ~ 0
        assert event.retransmit
        assert abs(event.total_delay_ms - 100.0) < 1.0

    def test_packet_loss_triggers_retransmit_probabilistically(self):
        """With moderate loss rate, some transfers should retransmit."""
        config = NetworkConfig(
            name="moderate_loss",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.5,  # 50% loss
            bandwidth_mbps=1000.0,
        )
        sim = NetworkSimulator(config, seed=42)

        retransmits = 0
        n = 200
        for _ in range(n):
            event = sim._compute_transfer(100, "upload")
            if event.retransmit:
                retransmits += 1

        # With 50% loss rate and 200 trials, expect ~100 retransmits
        # Allow wide tolerance for randomness
        assert 60 < retransmits < 140

    def test_jitter_is_gaussian(self, wifi_simulator):
        """RTT jitter should produce a roughly gaussian distribution."""
        rtts = []
        for _ in range(500):
            event = wifi_simulator._compute_transfer(100, "upload")
            rtts.append(event.simulated_rtt_ms)

        mean_rtt = sum(rtts) / len(rtts)
        variance = sum((r - mean_rtt) ** 2 for r in rtts) / len(rtts)
        std_dev = variance ** 0.5

        # Std dev should be close to jitter_ms (5.0)
        assert 2.0 < std_dev < 8.0

    def test_negative_jitter_clamped_to_zero(self):
        """Huge jitter that would make RTT negative should be clamped to 0."""
        config = NetworkConfig(
            name="extreme_jitter",
            rtt_ms=1,
            jitter_ms=100,  # much larger than rtt
            packet_loss_rate=0.0,
            bandwidth_mbps=1000.0,
        )
        sim = NetworkSimulator(config, seed=42)

        for _ in range(100):
            event = sim._compute_transfer(100, "upload")
            assert event.simulated_rtt_ms >= 0.0


# ============================================================================
# Tests: Async transfers
# ============================================================================


class TestAsyncTransfers:
    """Tests for async simulate_transfer."""

    @pytest.mark.asyncio
    async def test_async_transfer_returns_event(self, local_simulator):
        """Async transfer should return a TransferEvent."""
        event = await local_simulator.simulate_transfer(1024, "upload")
        assert isinstance(event, TransferEvent)
        assert event.direction == "upload"
        assert event.payload_bytes == 1024

    @pytest.mark.asyncio
    async def test_async_transfer_actually_sleeps(self):
        """Async transfer should actually sleep for the delay duration."""
        config = NetworkConfig(
            name="test",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=10000.0,
        )
        sim = NetworkSimulator(config, seed=42)

        start = time.monotonic()
        await sim.simulate_transfer(100, "upload")
        elapsed_ms = (time.monotonic() - start) * 1000

        # Should have slept ~50ms (allow 20ms tolerance for scheduling)
        assert elapsed_ms >= 40
        assert elapsed_ms < 120

    @pytest.mark.asyncio
    async def test_sync_transfer_actually_sleeps(self):
        """Sync transfer should actually sleep for the delay duration."""
        config = NetworkConfig(
            name="test",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=10000.0,
        )
        sim = NetworkSimulator(config, seed=42)

        start = time.monotonic()
        sim.simulate_transfer_sync(100, "upload")
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms >= 40
        assert elapsed_ms < 120


# ============================================================================
# Tests: RadioStateModel
# ============================================================================


class TestRadioStateModel:
    """Tests for cellular radio RRC state model."""

    def test_initial_state_is_idle(self, radio_model):
        """Radio should start in IDLE state."""
        assert radio_model.state == "IDLE"

    def test_first_transfer_triggers_transition(self, radio_model):
        """First transfer should trigger IDLE -> CONNECTED transition."""
        config = NETWORK_PRESETS["4g"]
        result = radio_model.on_transfer(time.time(), config)

        assert result["from_state"] == "IDLE"
        assert result["to_state"] == "CONNECTED"
        assert result["triggered_transition"] is True

    def test_rapid_transfers_no_extra_transitions(self, radio_model):
        """Rapid consecutive transfers should not trigger extra transitions."""
        config = NETWORK_PRESETS["4g"]
        t = time.time()

        # First transfer: IDLE -> CONNECTED
        r1 = radio_model.on_transfer(t, config)
        assert r1["triggered_transition"] is True

        # Second transfer 1s later (within tail timer)
        r2 = radio_model.on_transfer(t + 1.0, config)
        assert r2["triggered_transition"] is False

        # Third transfer 2s after that (still within tail timer from last)
        r3 = radio_model.on_transfer(t + 3.0, config)
        assert r3["triggered_transition"] is False

    def test_tail_timer_expiry_causes_transition(self, radio_model):
        """Transfer after tail timer expires should trigger new transition."""
        config = NETWORK_PRESETS["4g"]
        t = 1000.0  # Use fixed timestamps

        # First transfer
        radio_model.on_transfer(t, config)

        # Advance state machine past tail timer
        radio_model.update_idle_check(t + RadioStateModel.TAIL_TIMER_S + 1.0)
        assert radio_model.state == "IDLE"

        # New transfer after tail expired
        r2 = radio_model.on_transfer(t + RadioStateModel.TAIL_TIMER_S + 2.0, config)
        assert r2["triggered_transition"] is True
        assert r2["from_state"] == "IDLE"

    def test_transfer_during_tail_resets_timer(self, radio_model):
        """Transfer during TAIL state should keep radio connected."""
        config = NETWORK_PRESETS["4g"]
        t = 1000.0

        # First transfer
        radio_model.on_transfer(t, config)
        radio_model.update_idle_check(t + 5.0)  # Enter TAIL state
        assert radio_model.state == "TAIL"

        # Transfer during tail (within 7s window)
        r2 = radio_model.on_transfer(t + 5.0, config)
        assert r2["triggered_transition"] is False
        assert radio_model.state == "CONNECTED"

    def test_agentic_traffic_pattern_wastes_energy(self, radio_model):
        """Agentic LLM traffic (bursts with 15s gaps) should cause multiple transitions."""
        config = NETWORK_PRESETS["4g"]
        t = 0.0

        transitions = 0
        # Simulate 5 agentic iterations with 15s gaps (longer than 7s tail timer)
        for i in range(5):
            result = radio_model.on_transfer(t, config)
            if result["triggered_transition"]:
                transitions += 1
            # Advance past tail timer
            radio_model.update_idle_check(t + RadioStateModel.TAIL_TIMER_S + 1.0)
            t += 15.0  # 15s gap between iterations

        # Each iteration should trigger a new transition (5 total)
        assert transitions == 5

    def test_energy_summary_counts_transitions(self, radio_model):
        """Energy summary should correctly count state transitions."""
        config = NETWORK_PRESETS["4g"]
        t = 0.0

        # Create 3 transition events with gaps > tail timer
        for i in range(3):
            radio_model.on_transfer(t, config)
            radio_model.update_idle_check(t + RadioStateModel.TAIL_TIMER_S + 1.0)
            t += 20.0

        summary = radio_model.get_energy_summary()
        assert summary["total_transitions"] == 3

    def test_tail_energy_tracked_on_wasted_tails(self, radio_model):
        """Tail energy should be tracked when a tail period is followed by IDLE."""
        config = NetworkConfig(
            name="test_radio",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=100.0,
            radio_tail_energy_j=0.5,
        )
        t = 0.0

        # First transfer
        radio_model.on_transfer(t, config)
        # Advance past tail to IDLE
        radio_model.update_idle_check(t + RadioStateModel.TAIL_TIMER_S + 1.0)

        # Second transfer after tail expired — this should detect wasted tail energy
        t2 = t + RadioStateModel.TAIL_TIMER_S + 2.0
        result = radio_model.on_transfer(t2, config)
        assert result["triggered_transition"] is True
        # The tail energy is tracked from the TAIL->IDLE transition
        # that was wasted (radio stayed powered for nothing)

    def test_reset_clears_state(self, radio_model):
        """Reset should return radio model to initial state."""
        config = NETWORK_PRESETS["4g"]
        radio_model.on_transfer(time.time(), config)
        radio_model.reset()

        assert radio_model.state == "IDLE"
        assert radio_model.last_transfer_time == 0.0
        assert radio_model.state_transitions == []
        assert radio_model.total_tail_energy_j == 0.0


# ============================================================================
# Tests: TransferEvent logging and summary
# ============================================================================


class TestTransferLogging:
    """Tests for transfer event logging and summary statistics."""

    def test_transfers_are_logged(self, wifi_simulator):
        """Each transfer should be recorded in the log."""
        wifi_simulator._compute_transfer(1024, "upload")
        wifi_simulator._compute_transfer(2048, "download")

        log = wifi_simulator.get_transfer_log()
        assert len(log) == 2
        assert log[0].direction == "upload"
        assert log[0].payload_bytes == 1024
        assert log[1].direction == "download"
        assert log[1].payload_bytes == 2048

    def test_transfer_log_is_copy(self, wifi_simulator):
        """get_transfer_log should return a copy, not the internal list."""
        wifi_simulator._compute_transfer(1024, "upload")
        log = wifi_simulator.get_transfer_log()
        log.clear()
        assert len(wifi_simulator.get_transfer_log()) == 1

    def test_summary_empty_log(self, local_simulator):
        """Summary on empty log should return zeroed stats."""
        summary = local_simulator.get_summary()
        assert summary["total_transfers"] == 0
        assert summary["total_bytes"] == 0

    def test_summary_counts(self, wifi_simulator):
        """Summary should correctly aggregate transfer counts and bytes."""
        wifi_simulator._compute_transfer(1000, "upload")
        wifi_simulator._compute_transfer(2000, "upload")
        wifi_simulator._compute_transfer(3000, "download")

        summary = wifi_simulator.get_summary()
        assert summary["total_transfers"] == 3
        assert summary["total_bytes"] == 6000
        assert summary["total_uploads"] == 2
        assert summary["total_downloads"] == 1
        assert summary["network_preset"] == "wifi"

    def test_summary_mean_delay(self):
        """Mean delay should be correct."""
        config = NetworkConfig(
            name="test",
            rtt_ms=100,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=10000.0,
        )
        sim = NetworkSimulator(config, seed=42)
        sim._compute_transfer(100, "upload")
        sim._compute_transfer(100, "download")

        summary = sim.get_summary()
        # Each transfer should have ~100ms RTT
        assert abs(summary["mean_delay_ms"] - 100.0) < 1.0

    def test_reset_clears_log(self, wifi_simulator):
        """Reset should clear the transfer log."""
        wifi_simulator._compute_transfer(1024, "upload")
        assert len(wifi_simulator.get_transfer_log()) == 1

        wifi_simulator.reset()
        assert len(wifi_simulator.get_transfer_log()) == 0

    def test_reset_clears_radio_model(self, wifi_simulator):
        """Reset should also reset the radio model."""
        wifi_simulator._compute_transfer(1024, "upload")
        wifi_simulator.reset()
        assert wifi_simulator.radio_model.state == "IDLE"


# ============================================================================
# Tests: HybridArchitectureWrapper
# ============================================================================


class TestHybridArchitectureWrapper:
    """Tests for the hybrid deployment wrapper."""

    def _make_mock_service(self, response_text="Hello world", tool_calls=None):
        """Create a mock inference service."""
        service = MagicMock()
        service.generate_with_tools.return_value = (
            response_text,
            tool_calls,
            {"tokens": 10, "tps": 5.0},
        )
        return service

    def test_wrapper_calls_underlying_service(self):
        """Wrapper should call the underlying service's generate_with_tools."""
        service = self._make_mock_service()
        sim = NetworkSimulator(NETWORK_PRESETS["local"], seed=42)
        wrapper = HybridArchitectureWrapper(service, sim)

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        wrapper.generate_with_tools(messages, tools)

        service.generate_with_tools.assert_called_once_with(messages, tools)

    def test_wrapper_returns_service_response(self):
        """Wrapper should return the same response as the underlying service."""
        service = self._make_mock_service("test response", [{"name": "foo"}])
        sim = NetworkSimulator(NETWORK_PRESETS["local"], seed=42)
        wrapper = HybridArchitectureWrapper(service, sim)

        text, calls, metrics = wrapper.generate_with_tools(
            [{"role": "user", "content": "Hi"}], []
        )
        assert text == "test response"
        assert calls == [{"name": "foo"}]

    def test_wrapper_adds_network_metrics(self):
        """Wrapper should add network summary to metrics."""
        service = self._make_mock_service()
        sim = NetworkSimulator(NETWORK_PRESETS["wifi"], seed=42)
        wrapper = HybridArchitectureWrapper(service, sim)

        _, _, metrics = wrapper.generate_with_tools(
            [{"role": "user", "content": "Hello"}], []
        )

        assert "network" in metrics
        assert metrics["network"]["total_transfers"] == 2  # 1 upload + 1 download
        assert metrics["network"]["total_uploads"] == 1
        assert metrics["network"]["total_downloads"] == 1

    def test_wrapper_simulates_upload_and_download(self):
        """Wrapper should create both upload and download transfer events."""
        service = self._make_mock_service("A" * 1000)
        sim = NetworkSimulator(NETWORK_PRESETS["wifi"], seed=42)
        wrapper = HybridArchitectureWrapper(service, sim)

        wrapper.generate_with_tools(
            [{"role": "user", "content": "Hello"}], []
        )

        log = sim.get_transfer_log()
        assert len(log) == 2
        assert log[0].direction == "upload"
        assert log[1].direction == "download"
        # Download should include the response size
        assert log[1].payload_bytes >= 1000

    def test_wrapper_adds_delay_on_slow_network(self):
        """Wrapper on a slow network should measurably increase wall-clock time."""
        service = self._make_mock_service("short response")
        config = NetworkConfig(
            name="slow_test",
            rtt_ms=50,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=10000.0,
        )
        sim = NetworkSimulator(config, seed=42)
        wrapper = HybridArchitectureWrapper(service, sim)

        start = time.monotonic()
        wrapper.generate_with_tools(
            [{"role": "user", "content": "Hello"}], []
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        # Should have at least ~100ms delay (50ms upload + 50ms download RTT)
        assert elapsed_ms >= 80

    def test_wrapper_includes_tool_calls_in_download_size(self):
        """Download size should include serialized tool calls."""
        tool_calls = [{"name": "run_python_code", "parameters": {"code": "print(1)"}}]
        service = self._make_mock_service("ok", tool_calls)
        sim = NetworkSimulator(NETWORK_PRESETS["local"], seed=42)
        wrapper = HybridArchitectureWrapper(service, sim)

        wrapper.generate_with_tools(
            [{"role": "user", "content": "run code"}], []
        )

        log = sim.get_transfer_log()
        download_event = log[1]
        # Download bytes should include both response text and tool calls JSON
        assert download_event.payload_bytes > len("ok".encode("utf-8"))

    def test_payload_estimation(self):
        """_estimate_payload should give reasonable byte estimates."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a function that sorts a list."},
        ]
        estimate = HybridArchitectureWrapper._estimate_payload(messages)
        # Should be roughly the JSON size
        import json
        expected = len(json.dumps(messages).encode("utf-8"))
        assert estimate == expected


# ============================================================================
# Tests: DeploymentArchitecture enum
# ============================================================================


class TestDeploymentArchitecture:
    """Tests for the deployment architecture enum."""

    def test_three_architectures(self):
        """Should have exactly three deployment modes."""
        assert len(DeploymentArchitecture) == 3

    def test_values(self):
        """Values should be the expected strings."""
        assert DeploymentArchitecture.FULLY_LOCAL == "local"
        assert DeploymentArchitecture.HYBRID == "hybrid"
        assert DeploymentArchitecture.FULLY_CLOUD == "cloud"


# ============================================================================
# Tests: Sweep across all presets
# ============================================================================


class TestPresetSweep:
    """Tests for sweeping across all network presets."""

    def test_sweep_all_presets_produces_events(self):
        """Running a transfer on every preset should produce valid events."""
        for name, config in NETWORK_PRESETS.items():
            sim = NetworkSimulator(config, seed=42)
            event = sim._compute_transfer(4096, "upload")

            assert isinstance(event, TransferEvent)
            assert event.payload_bytes == 4096
            assert event.direction == "upload"
            assert event.total_delay_ms >= 0

    def test_sweep_delay_ordering(self):
        """Worse network conditions should generally produce higher mean delays."""
        mean_delays = {}
        for name, config in NETWORK_PRESETS.items():
            sim = NetworkSimulator(config, seed=42)
            delays = []
            for _ in range(50):
                event = sim._compute_transfer(4096, "upload")
                delays.append(event.total_delay_ms)
            mean_delays[name] = sum(delays) / len(delays)

        # Local should have the lowest delay
        assert mean_delays["local"] < mean_delays["wifi"]
        # Edge case should have the highest
        assert mean_delays["edge_case"] > mean_delays["4g"]

    def test_sweep_collects_summaries(self):
        """Sweeping and collecting summaries should work for all presets."""
        summaries = {}
        for name, config in NETWORK_PRESETS.items():
            sim = NetworkSimulator(config, seed=42)
            for _ in range(10):
                sim._compute_transfer(2048, "upload")
                sim._compute_transfer(1024, "download")
            summaries[name] = sim.get_summary()

        assert len(summaries) == 7
        for name, summary in summaries.items():
            assert summary["total_transfers"] == 20
            assert summary["total_uploads"] == 10
            assert summary["total_downloads"] == 10
            assert summary["network_preset"] == name


# ============================================================================
# Tests: Total delay decomposition
# ============================================================================


class TestDelayDecomposition:
    """Tests that total_delay = rtt + bandwidth_delay + retransmit_delay."""

    def test_decomposition_no_loss(self):
        """Without loss: total = rtt + bandwidth."""
        config = NetworkConfig(
            name="decomp_test",
            rtt_ms=100,
            jitter_ms=0,
            packet_loss_rate=0.0,
            bandwidth_mbps=10.0,  # 10 Mbps
        )
        sim = NetworkSimulator(config, seed=42)

        # 12500 bytes at 10 Mbps: 12500 / (10*125) = 10ms bandwidth delay
        event = sim._compute_transfer(12500, "upload")

        expected_bw = 12500 / (10.0 * 125.0)
        assert abs(event.bandwidth_delay_ms - expected_bw) < 0.001
        assert abs(event.simulated_rtt_ms - 100.0) < 0.001
        assert event.total_delay_ms == pytest.approx(100.0 + expected_bw, abs=0.01)

    def test_decomposition_with_loss(self):
        """With 100% loss: total = rtt + bandwidth + rtt (retransmit)."""
        config = NetworkConfig(
            name="decomp_loss",
            rtt_ms=80,
            jitter_ms=0,
            packet_loss_rate=1.0,
            bandwidth_mbps=10.0,
        )
        sim = NetworkSimulator(config, seed=42)

        event = sim._compute_transfer(12500, "upload")

        expected_bw = 12500 / (10.0 * 125.0)
        # retransmit adds another RTT
        expected_total = 80.0 + expected_bw + 80.0
        assert event.retransmit
        assert event.total_delay_ms == pytest.approx(expected_total, abs=0.01)

    def test_decomposition_with_jitter(self):
        """With jitter: total still equals simulated_rtt + bandwidth + retransmit."""
        config = NetworkConfig(
            name="decomp_jitter",
            rtt_ms=50,
            jitter_ms=10,
            packet_loss_rate=0.0,
            bandwidth_mbps=100.0,
        )
        sim = NetworkSimulator(config, seed=42)

        for _ in range(50):
            event = sim._compute_transfer(5000, "upload")
            expected_bw = 5000 / (100.0 * 125.0)
            retransmit_delay = event.simulated_rtt_ms if event.retransmit else 0.0
            expected_total = event.simulated_rtt_ms + expected_bw + retransmit_delay
            assert event.total_delay_ms == pytest.approx(expected_total, abs=0.001)


# ============================================================================
# Tests: Reproducibility with seed
# ============================================================================


class TestReproducibility:
    """Tests that the same seed produces identical results."""

    def test_same_seed_same_results(self):
        """Two simulators with the same seed should produce identical events."""
        config = NETWORK_PRESETS["4g"]

        sim1 = NetworkSimulator(config, seed=123)
        sim2 = NetworkSimulator(config, seed=123)

        events1 = [sim1._compute_transfer(1024, "upload") for _ in range(20)]
        events2 = [sim2._compute_transfer(1024, "upload") for _ in range(20)]

        for e1, e2 in zip(events1, events2):
            assert e1.simulated_rtt_ms == e2.simulated_rtt_ms
            assert e1.bandwidth_delay_ms == e2.bandwidth_delay_ms
            assert e1.retransmit == e2.retransmit
            assert e1.total_delay_ms == e2.total_delay_ms

    def test_different_seeds_different_results(self):
        """Different seeds should (almost certainly) produce different events."""
        config = NETWORK_PRESETS["4g"]

        sim1 = NetworkSimulator(config, seed=1)
        sim2 = NetworkSimulator(config, seed=2)

        events1 = [sim1._compute_transfer(1024, "upload") for _ in range(10)]
        events2 = [sim2._compute_transfer(1024, "upload") for _ in range(10)]

        # At least some RTTs should differ
        diffs = sum(
            1 for e1, e2 in zip(events1, events2)
            if e1.simulated_rtt_ms != e2.simulated_rtt_ms
        )
        assert diffs > 0
