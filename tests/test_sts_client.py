"""Unit tests for ``StsClient`` (Feetech STS3215) against a simulated bus.

What these pin down:
- the force limit goes to register 48 (torque limit), scaled 2.5 mA per unit,
  and register 44 (goal time on STS, goal current on HLS) only ever gets 0;
- offset calibration lands the reading on the requested raw target, keeps the
  motor still, restores torque, and works whichever sign convention the
  firmware uses for the homing offset;
- probing and scanning tell STS and HLS servos apart by model number.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import orca_core.hardware.feetech_client as feetech_client_module
import orca_core.hardware.sts_client as sts_client_module
from orca_core.constants import FEETECH_STS, SUPPORTED_MOTOR_TYPES
from orca_core.hardware.feetech_client import COMM_SUCCESS, FeetechClient
from orca_core.hardware.mock_sts_client import MockStsClient
from orca_core.hardware.motor_client import ServoGains
from orca_core.hardware.motor_factory import mock_motor_client_class, motor_client_class
from orca_core.hardware.sts_client import (
    StsClient,
    _decode_homing_offset,
    _encode_homing_offset,
    _wrap_offset,
)
from orca_core.hardware.sts_registers import STS

COMM_FAIL = -1001
STS3215_MODEL = 777
HLS3915_MODEL = 6922


class SimulatedStsBus:
    """A few STS3215 servos as register memory.

    Present position is computed from a hidden encoder angle and the homing
    offset register, as the servo firmware does. ``offset_sign`` chooses the
    convention: +1 reads ``encoder - offset``, -1 reads ``encoder + offset``.
    """

    def __init__(self, encoders: dict[int, int], offset_sign: int = 1,
                 model: int = STS3215_MODEL):
        self.encoders = dict(encoders)
        self.offset_sign = offset_sign
        self.model = model
        self.regs: dict[int, dict[int, int]] = {
            mid: {STS.TORQUE_ENABLE: 1, STS.LOCK: 1, STS.KP: 32, STS.KD: 32, STS.KI: 0}
            for mid in encoders
        }
        self.offsets = {mid: 0 for mid in encoders}  # signed, decoded
        self.writes: list[tuple] = []         # (motor_id, address, value) per single write
        self.sync_writes: list[tuple] = []    # (start_address, {motor_id: bytes})
        self.dead: set[int] = set()

    # -- what the client calls ------------------------------------------------
    def present(self, mid: int) -> int:
        return (self.encoders[mid] - self.offset_sign * self.offsets[mid]) % 4096

    def ping(self, mid):
        if mid not in self.regs or mid in self.dead:
            return 0, COMM_FAIL, 0
        return self.model, COMM_SUCCESS, 0

    def read1ByteTxRx(self, mid, address):
        if mid not in self.regs or mid in self.dead:
            return 0, COMM_FAIL, 0
        return self.regs[mid].get(address, 0), COMM_SUCCESS, 0

    def read2ByteTxRx(self, mid, address):
        if mid not in self.regs or mid in self.dead:
            return 0, COMM_FAIL, 0
        if address == STS.PRESENT_POSITION:
            return self.present(mid), COMM_SUCCESS, 0
        if address == STS.HOMING_OFFSET:
            return _encode_homing_offset(self.offsets[mid]), COMM_SUCCESS, 0
        return self.regs[mid].get(address, 0), COMM_SUCCESS, 0

    def write1ByteTxRx(self, mid, address, value):
        if mid not in self.regs or mid in self.dead:
            return COMM_FAIL, 0
        self.writes.append((mid, address, value))
        self.regs[mid][address] = value
        return COMM_SUCCESS, 0

    def write2ByteTxRx(self, mid, address, value):
        if mid not in self.regs or mid in self.dead:
            return COMM_FAIL, 0
        self.writes.append((mid, address, value))
        if address == STS.HOMING_OFFSET:
            assert self.regs[mid][STS.LOCK] == 0, "EEPROM written while locked"
            self.offsets[mid] = _decode_homing_offset(value)
        else:
            self.regs[mid][address] = value
        return COMM_SUCCESS, 0

    def syncWriteTxOnly(self, start_address, data_length, param, param_length):
        per_motor = {}
        for i in range(0, param_length, 1 + data_length):
            per_motor[param[i]] = list(param[i + 1:i + 1 + data_length])
        self.sync_writes.append((start_address, per_motor))
        return COMM_SUCCESS

    # -- byte helpers the client and SDK groups use ---------------------------
    def scs_tohost(self, raw, bit):
        return raw

    def scs_toscs(self, raw, bit):
        return raw

    def scs_lobyte(self, word):
        return word & 0xFF

    def scs_hibyte(self, word):
        return (word >> 8) & 0xFF


def _client_on(bus: SimulatedStsBus) -> StsClient:
    """An StsClient wired to the simulated bus, as if connect() had run."""
    client = StsClient(list(bus.encoders), port="sim")
    client.port_handler = SimpleNamespace(is_open=True, is_using=False, ser=None,
                                          closePort=lambda: None)
    client.packet_handler = bus
    client._connected = True
    return client


def _word(data: list[int]) -> int:
    return data[0] | (data[1] << 8)


# ---------------------------------------------------------------- registration

def test_sts_family_is_registered():
    assert FEETECH_STS in SUPPORTED_MOTOR_TYPES
    assert motor_client_class(FEETECH_STS) is StsClient
    assert mock_motor_client_class(FEETECH_STS) is MockStsClient


def test_mock_copies_sts_family_attributes():
    for name in ("motor_type", "current_scale_ma", "max_current_ma",
                 "default_max_current_ma", "default_calibration_current_ma",
                 "config_registers"):
        assert getattr(MockStsClient, name) == getattr(StsClient, name)


def test_sts_config_registers_leave_out_the_hls_shadow_id():
    # Register 7 is Return Delay on STS; offering it as "shadow ID" would corrupt it.
    keys = {entry.key for entry in StsClient.config_registers}
    assert "secondary_id" not in keys
    assert all(entry.address != 7 for entry in StsClient.config_registers)


# ---------------------------------------------------------------- homing offset encoding

@pytest.mark.parametrize("value", [0, 1, 100, 2047, -1, -100, -2047])
def test_homing_offset_round_trips(value):
    assert _decode_homing_offset(_encode_homing_offset(value)) == value


def test_negative_offset_sets_bit_11():
    assert _encode_homing_offset(-100) == 100 | (1 << 11)


@pytest.mark.parametrize("raw, wrapped", [(0, 0), (2047, 2047), (2048, -2047),
                                          (4095, -1), (-4095, 1), (-2047, -2047)])
def test_wrap_offset_stays_encodable(raw, wrapped):
    assert _wrap_offset(raw) == wrapped
    assert abs(_wrap_offset(raw)) <= STS.HOMING_OFFSET_MAX


# ---------------------------------------------------------------- force limit

def test_desired_current_goes_to_torque_limit_register():
    bus = SimulatedStsBus({1: 2048, 2: 2048})
    client = _client_on(bus)

    client.write_desired_current([1, 2], np.array([1250.0, 500.0]))

    (address, per_motor), = bus.sync_writes
    assert address == STS.TORQUE_LIMIT
    assert _word(per_motor[1]) == 500   # 1250 mA / 2.5 = 50.0 %
    assert _word(per_motor[2]) == 200   # 500 mA  / 2.5 = 20.0 %


def test_desired_current_is_clamped_to_full_torque():
    bus = SimulatedStsBus({1: 2048})
    client = _client_on(bus)

    client.write_desired_current([1], np.array([9000.0]))

    (_, per_motor), = bus.sync_writes
    assert _word(per_motor[1]) == STS.TORQUE_LIMIT_MAX_RAW


def test_profile_write_never_puts_a_current_into_goal_time():
    bus = SimulatedStsBus({1: 2048, 2: 2048})
    client = _client_on(bus)
    client._current_limit_raw = {1: 500, 2: 200}

    client._write_profile_params([1, 2])

    writes = dict(bus.sync_writes)
    assert set(writes) == {STS.ACC, STS.GOAL_TIME, STS.TORQUE_LIMIT}
    for mid in (1, 2):
        assert writes[STS.GOAL_TIME][mid][:2] == [0, 0]       # goal time 0
    assert _word(writes[STS.TORQUE_LIMIT][1]) == 500
    assert _word(writes[STS.TORQUE_LIMIT][2]) == 200


def test_hold_and_release_use_torque_limit():
    bus = SimulatedStsBus({1: 2048})
    client = _client_on(bus)
    client._current_limit_raw[1] = 400

    assert client.hold_motor_off(1)
    assert bus.regs[1][STS.TORQUE_LIMIT] == 0
    assert client.release_motor(1)
    assert bus.regs[1][STS.TORQUE_LIMIT] == 400
    assert all(address != STS.GOAL_TIME for _, address, _ in bus.writes)


def test_current_limits_report_full_torque_without_bus_traffic():
    bus = SimulatedStsBus({1: 2048, 2: 2048})
    client = _client_on(bus)

    assert client.read_current_limits() == {1: 2500.0, 2: 2500.0}
    assert bus.writes == [] and bus.sync_writes == []


# ---------------------------------------------------------------- gains

def test_gains_are_read_from_eeprom_registers():
    bus = SimulatedStsBus({1: 2048})
    bus.regs[1].update({STS.KP: 20, STS.KD: 10, STS.KI: 3})
    client = _client_on(bus)

    assert client.read_servo_gains([1]) == {1: ServoGains(kp=20, ki=3, kd=10)}


def test_gain_writes_are_refused():
    client = _client_on(SimulatedStsBus({1: 2048}))
    with pytest.raises(ValueError, match="EEPROM"):
        client.write_servo_gains({1: ServoGains(kp=10)})


# ---------------------------------------------------------------- offset calibration

def _target(upper: bool) -> int:
    # Same raw targets as the HLS client: the read frame is inverted.
    flipped = (not upper) if sts_client_module.POSITION_DIRECTION < 0 else upper
    return 3595 if flipped else 500


@pytest.mark.parametrize("offset_sign", [1, -1], ids=["encoder-minus-offset", "encoder-plus-offset"])
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("encoder", [10, 1500, 2048, 4000])
def test_offset_calibration_lands_on_target(monkeypatch, offset_sign, upper, encoder):
    monkeypatch.setattr(sts_client_module, "EEPROM_SETTLE_S", 0.0)
    bus = SimulatedStsBus({1: encoder}, offset_sign=offset_sign)
    client = _client_on(bus)

    assert client.calibrate_offset(1, upper=upper)

    target = _target(upper)
    assert abs(bus.present(1) - target) <= sts_client_module.OFFSET_VERIFY_TOLERANCE_STEPS
    # Goal set to where it now reads, so re-enabling torque does not move it.
    assert bus.regs[1][STS.GOAL_POSITION] == bus.present(1)
    assert bus.regs[1][STS.TORQUE_ENABLE] == 1   # torque restored
    assert bus.regs[1][STS.LOCK] == 1            # EEPROM re-locked


def test_offset_calibration_keeps_torque_off_when_it_was_off(monkeypatch):
    monkeypatch.setattr(sts_client_module, "EEPROM_SETTLE_S", 0.0)
    bus = SimulatedStsBus({1: 1200})
    bus.regs[1][STS.TORQUE_ENABLE] = 0
    client = _client_on(bus)

    assert client.calibrate_offset(1, upper=True)
    assert bus.regs[1][STS.TORQUE_ENABLE] == 0


def test_offset_calibration_turns_torque_off_before_touching_the_offset(monkeypatch):
    monkeypatch.setattr(sts_client_module, "EEPROM_SETTLE_S", 0.0)
    bus = SimulatedStsBus({1: 1200})
    client = _client_on(bus)

    client.calibrate_offset(1, upper=True)

    first_offset_write = next(i for i, w in enumerate(bus.writes) if w[1] == STS.HOMING_OFFSET)
    torque_off = bus.writes.index((1, STS.TORQUE_ENABLE, 0))
    assert torque_off < first_offset_write


def test_offset_calibration_fails_cleanly_on_a_silent_motor():
    bus = SimulatedStsBus({1: 1200})
    bus.dead.add(1)
    client = _client_on(bus)

    assert client.calibrate_offset(1, upper=True) is False


def test_offset_calibration_restores_the_old_offset_when_nothing_works(monkeypatch):
    monkeypatch.setattr(sts_client_module, "EEPROM_SETTLE_S", 0.0)
    bus = SimulatedStsBus({1: 1200})
    bus.offsets[1] = 37
    client = _client_on(bus)
    # A servo that ignores the offset entirely: no candidate can verify.
    bus.present = lambda mid: 1200

    assert client.calibrate_offset(1, upper=True) is False
    assert bus.offsets[1] == 37


# ---------------------------------------------------------------- telling STS from HLS

class _FakePort:
    def __init__(self, port):
        self.baudrate = None

    def openPort(self):
        return True

    def closePort(self):
        pass


def _probe_with_model(monkeypatch, module, client_cls, model):
    monkeypatch.setattr(module, "PortHandler", _FakePort)
    monkeypatch.setattr(
        module, "HLSPacketHandler",
        lambda handler: SimpleNamespace(ping=lambda mid: (model, COMM_SUCCESS, 0)))
    return client_cls.probe("sim", 1_000_000, [1, 2])


def test_sts_probe_accepts_sts_and_rejects_hls(monkeypatch):
    assert _probe_with_model(monkeypatch, sts_client_module, StsClient, STS3215_MODEL)
    assert not _probe_with_model(monkeypatch, sts_client_module, StsClient, HLS3915_MODEL)


def test_hls_probe_no_longer_claims_sts_servos(monkeypatch):
    assert _probe_with_model(monkeypatch, feetech_client_module, FeetechClient, HLS3915_MODEL)
    assert not _probe_with_model(monkeypatch, feetech_client_module, FeetechClient, STS3215_MODEL)


def test_sts_scan_keeps_only_sts_servos(monkeypatch):
    found = [
        {"id": 1, "baud_rate": 1_000_000, "model_name": "STS3215"},
        {"id": 2, "baud_rate": 1_000_000, "model_name": "HLS3915"},
    ]
    monkeypatch.setattr(FeetechClient, "scan_for_motors", lambda self, *a, **k: found)
    client = StsClient([1], port="sim")
    assert [m["id"] for m in client.scan_for_motors("sim", (1, 2))] == [1]


# ---------------------------------------------------------------- whole stack, no hardware

def test_mock_hand_with_sts_motors_runs_end_to_end(tmp_path):
    """A config that says ``motor_type: feetech_sts`` drives the whole ORCA stack."""
    import logging
    import shutil
    from pathlib import Path

    import yaml

    import orca_core
    from orca_core import load_hand

    model_dir = Path(orca_core.__file__).parent / "models" / "v2" / "orcahand-right"
    config = tmp_path / "config.yaml"
    shutil.copy(model_dir / "config.yaml", config)
    with open(config, "a") as f:
        f.write("\nmotor_type: feetech_sts\n")

    # The mock's synthesized limits sit at [0, span], outside the Feetech read
    # frame (which is <= 0), for the HLS mock as much as this one. Give the hand
    # limits inside that frame, as a real calibration on these motors would.
    cfg = yaml.safe_load(config.read_text())
    ratio = 0.007
    limits, ratios = {}, {}
    for joint, motor in cfg["joint_to_motor_map"].items():
        lo, hi = cfg["joint_roms"][joint]
        limits[abs(motor)] = [-(hi - lo) * ratio, 0.0]
        ratios[abs(motor)] = ratio
    (tmp_path / "calibration.yaml").write_text(yaml.safe_dump({
        "motor_limits": limits, "joint_to_motor_ratios": ratios,
        "calibrated": True, "wrist_calibrated": True,
    }))

    logging.disable(logging.WARNING)
    try:
        hand = load_hand(config_path=str(config), mock=True)
        hand.connect()
        try:
            assert isinstance(hand.motor_client, MockStsClient)
            hand.init_joints(move_to_neutral=False)
            hand.set_joint_positions({"index_mcp": 50.0})
            assert hand.get_joint_position().data["index_mcp"] == pytest.approx(50.0, abs=0.5)
        finally:
            hand.disconnect()
    finally:
        logging.disable(logging.NOTSET)
