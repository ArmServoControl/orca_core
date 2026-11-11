# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================

"""Feetech STS3215 舵机的「传话员」（MotorClient）。

整体思路：
    STS3215 和 ORCA 已经支持的 Feetech HLS 舵机是同一家公司、同一种通信协议，
    寄存器地址也几乎一样。所以这里不从零写，而是「继承」ORCA 的 FeetechClient：
    连接、读位置、写位置、开关扭矩、扫描、改 ID 这些都直接沿用父类，
    只重写 STS 和 HLS 不一样的地方：

    1. 限力方式：HLS 用 44 号「目标电流」，STS 用 48 号「扭矩限制」。
       ORCA 上层只会说「把电流上限设成 X mA」，所以我们把 mA 换算成扭矩百分比。
    2. 44 号寄存器：在 STS 上是「目标时间」，必须写 0，不能把电流值写进去。
    3. PID 增益：STS 的在 EEPROM（21-23 号），不适合频繁写，所以只读不写。
    4. 零点标定：STS 没有 HLS 那条「偏移标定」指令，改成写 31 号零点偏移寄存器。
    5. 识别：只认 STS 型号编号，避免把 HLS 手当成 STS 手（反之亦然）。

上层（OrcaHand）完全不知道底下是 STS 还是 HLS，这就是分层的好处。
"""

import logging  # 打印日志（警告、错误）
import time  # 用来短暂等待（time.sleep）
from typing import Optional, Sequence  # 类型标注，让代码更好读

import numpy as np  # 数组运算，ORCA 里位置、电流都用 numpy 数组

from ..constants import FEETECH_STS  # 这个舵机家族的名字 "feetech_sts"
from .config_registers import ConfigRegister  # 「操作员可改的设置」的描述格式
from .feetech import COMM_SUCCESS, PortHandler  # SDK：通信成功的代号、串口对象
from .feetech_client import (  # 从 ORCA 的 HLS 驱动里借用现成的东西：
    DEFAULT_ACC,  #   默认加速度
    DEFAULT_SPEED,  #   默认速度
    FEETECH_BAUD_RATE_MAP,  #   波特率 → 寄存器代码 对照表（STS 和 HLS 一样）
    POSITION_DIRECTION,  #   方向约定（Feetech 读数取反，和 Dynamixel 对齐）
    FeetechClient,  #   父类：我们要继承的 HLS 驱动
    HLSPacketHandler,  #   打包/解包数据的工具（协议和 STS 一样）
)
from .motor_client import ServoGains  # 「增益」的数据格式：kp / ki / kd
from .sts_registers import STS, STS_MODEL_NAMES, STS_MODEL_NUMBERS  # 第一步写的地址表

# 零点标定后，读数和目标值最多允许差几格才算成功（4 格 ≈ 0.35°）
OFFSET_VERIFY_TOLERANCE_STEPS = 4
# 写完 EEPROM 后等多久再读（秒）。EEPROM 写入需要一点时间
EEPROM_SETTLE_S = 0.02


# 操作员可以在 ORCA 的工具里改的设置。和 HLS 版的区别：
# 去掉了「影子 ID」（STS 的 7 号寄存器是回复延迟，不是影子 ID，写错会出事）。
STS_CONFIG_REGISTERS = (
    ConfigRegister(
        key="id", label="ID", address=STS.ID, size=1, eeprom=True,  # 5 号：舵机 ID
        minimum=0, maximum=253, reidentifies=True,                   # 改完 ID 舵机马上用新 ID 回话
        note="the motor answers at the new id immediately"),
    ConfigRegister(
        key="baud_rate", label="Baud rate", address=STS.BAUD_RATE, size=1,  # 6 号：波特率
        eeprom=True, reidentifies=True,
        choices={index: f"{rate:,}".replace(",", " ")               # 显示成 "1 000 000" 这样的选项
                 for rate, index in FEETECH_BAUD_RATE_MAP.items()},
        note="indices run the opposite way to Dynamixel: 0 is the fastest"),
    ConfigRegister(
        key="operating_mode", label="Operating mode", address=STS.MODE,  # 33 号：工作模式
        size=1, eeprom=True,
        choices={0: "position", 1: "speed (closed loop)",
                 2: "speed (open loop)", 3: "step"},
        note="position here is torque-limited by register 48"),
    ConfigRegister(
        key="temperature_limit", label="Temperature limit",          # 13 号：过热保护温度
        address=STS.MAX_TEMPERATURE, size=1, eeprom=True,
        unit="degC", minimum=0, maximum=100,
        note="the shutdown threshold, not a warning level"),
)


def _encode_homing_offset(offset: int) -> int:
    """把一个有正负的偏移量，编码成 STS 的「符号-数值」格式（第 11 位是符号）。

    例：+100 → 100；-100 → 100 + 2048 = 2148。
    """
    magnitude = min(abs(int(offset)), STS.HOMING_OFFSET_MAX)  # 取绝对值，最大 2047
    if offset < 0:                                            # 负数：
        return magnitude | (1 << STS.HOMING_OFFSET_SIGN_BIT)  #   把第 11 位置 1 表示负号
    return magnitude                                          # 正数：原样返回


def _decode_homing_offset(raw: int) -> int:
    """反过来：把寄存器读到的「符号-数值」格式还原成有正负的整数。"""
    magnitude = raw & STS.HOMING_OFFSET_MAX                   # 低 11 位是数值
    if raw & (1 << STS.HOMING_OFFSET_SIGN_BIT):               # 第 11 位是 1：
        return -magnitude                                     #   负数
    return magnitude                                          # 否则是正数


def _wrap_offset(offset: int) -> int:
    """把偏移量折回 -2047…2047 的范围内（一圈 4096 格，转一圈等于没转）。"""
    half = STS.STEPS_PER_REV // 2                             # 2048
    wrapped = ((int(offset) + half) % STS.STEPS_PER_REV) - half  # 折到 -2048…2047
    return max(wrapped, -STS.HOMING_OFFSET_MAX)               # -2048 编码不了，改成 -2047（差 1 格可忽略）


class StsClient(FeetechClient):
    """Feetech STS 系列（STS3215）舵机驱动，继承 ORCA 的 Feetech HLS 驱动。"""

    # ---------------- 家族描述：上层通过这些属性了解这种舵机 ----------------
    motor_type = FEETECH_STS            # 家族名字，config.yaml 里写 motor_type: feetech_sts
    config_registers = STS_CONFIG_REGISTERS  # 用上面去掉了「影子 ID」的设置表
    servo_gain_max = STS.GAIN_MAX       # PID 增益寄存器最大值 254
    # 空载转速 52 rpm 换算成 rad/s：rpm × 2π / 60
    no_load_speed_rad_s = STS.NO_LOAD_RPM * 2.0 * np.pi / 60.0

    # ---- 「电流」的含义：ORCA 上层用 mA 说话，STS 只有扭矩限制（0-1000 = 0-100%）----
    # 扭矩 ∝ 电流，堵转电流 2500 mA 对应 100% 扭矩，所以每 1 个寄存器单位 = 2.5 mA。
    current_scale_ma = STS.STALL_CURRENT_MA / STS.TORQUE_LIMIT_MAX_RAW  # = 2.5 mA / 单位
    max_current_ma = STS.STALL_CURRENT_MA                               # 最多 2500 mA（100%）
    # 平时运行的默认上限：1250 mA = 50% 扭矩 ≈ 9.75 kg·cm。
    # 低于 80% 的过载保护线，不会因为握东西太用力就自动卸力。
    default_max_current_ma = 1250
    # 标定时顶限位用的力：500 mA = 20% 扭矩 ≈ 3.9 kg·cm，足够把关节推到头，又不会拉坏肌腱。
    default_calibration_current_ma = 500

    def __init__(
        self,
        motor_ids: Sequence[int],          # 这条总线上所有舵机的 ID
        port: str = '/dev/ttyUSB0',        # 串口名，Windows 上是 COM5 这样的
        baudrate: int = 1000000,           # 波特率，STS3215 出厂是 1M
    ):
        super().__init__(motor_ids, port, baudrate)  # 先让父类把所有通用的东西准备好
        # 父类按 HLS 的满量程（2047）初始化了限力值，这里改成 STS 的满量程 1000
        self._current_limit_raw = {mid: STS.TORQUE_LIMIT_MAX_RAW for mid in self.motor_ids}
        self._current_ceiling_raw = {mid: STS.TORQUE_LIMIT_MAX_RAW for mid in self.motor_ids}

    # ======================== 1. 限力：扭矩限制（48 号） ========================

    def read_current_limits(self) -> "dict[int, float | None]":
        """每个舵机的限力上限（mA）。

        HLS 会去读舵机里的「保护电流」作为上限；STS 的扭矩限制本身就是 0-100%，
        上限固定是 100%，所以不需要读总线（connect() 时父类会调用这里）。
        """
        self.check_connected()                                   # 没连接就报错
        for motor_id in self.motor_ids:                          # 对每个舵机：
            self._current_ceiling_raw[motor_id] = STS.TORQUE_LIMIT_MAX_RAW  # 上限 = 1000
        return {mid: self._current_ceiling_ma(mid) for mid in self.motor_ids}  # 换算成 mA 返回

    def _current_ceiling_ma(self, motor_id: int) -> "float | None":
        """某个舵机的上限，单位 mA（寄存器值 × 2.5）。"""
        ceiling_raw = self._current_ceiling_raw.get(motor_id, STS.TORQUE_LIMIT_MAX_RAW)
        return ceiling_raw * self.current_scale_ma

    def _current_bytes(self, motor_id: int) -> "list[int]":
        """某个舵机当前的扭矩限制，拆成两个字节（低字节在前）。"""
        return self._word_bytes(
            self._current_limit_raw.get(motor_id, STS.TORQUE_LIMIT_MAX_RAW))

    def write_desired_current(
        self,
        motor_ids: Sequence[int],      # 要设置的舵机
        currents: np.ndarray,          # 每个舵机的上限，单位 mA
    ) -> None:
        """设置限力。上层说 mA，我们换算成扭矩限制写进 48 号寄存器。"""
        self.check_connected()                                   # 没连接就报错
        # 父类的工具函数：检查数值合法（不能负数）、超过上限就截断、换算成寄存器值
        plan = self._goal_current_plan(motor_ids, currents)

        with self._bus_lock:                                     # 占用总线，防止别的线程插话
            sync_write = self._sync_write(STS.TORQUE_LIMIT, 2)   # 准备一个「同步写 48 号、2 字节」的包
            sync_write.clearParam()                              # 清空上次留下的内容
            failed_ids = []                                      # 记录写失败的舵机
            for motor_id, raw in plan.items():                   # 对每个舵机：
                self._current_limit_raw[motor_id] = raw          #   记住这个值（重启后要恢复）
                if not sync_write.addParam(motor_id, self._current_bytes(motor_id)):
                    failed_ids.append(motor_id)                  #   加不进包就记为失败
            if sync_write.txPacket() != COMM_SUCCESS:            # 一次性发给所有舵机
                self._flush_input_buffer()                       # 失败就清掉残留数据
                failed_ids = list(plan)                          # 整包失败 = 全部失败
            sync_write.clearParam()                              # 用完清空

        if failed_ids:                                           # 有失败的就报错
            logging.error('Torque limit not written for %d motor(s): %s',
                          len(failed_ids), str(failed_ids))

    def hold_motor_off(self, motor_id: int) -> bool:
        """让某个舵机「不出力」（比如过热时让它冷却）：扭矩限制写 0。"""
        self.check_connected()
        with self._bus_lock:
            result, _ = self.packet_handler.write2ByteTxRx(
                int(motor_id), STS.TORQUE_LIMIT, 0)              # 48 号写 0 = 0% 扭矩
        if result != COMM_SUCCESS:                               # 写失败：
            logging.error("Motor %d: could not set its torque limit to zero", motor_id)
            return False                                         #   告诉上层没成功
        self._held_off.add(int(motor_id))                        # 记下它处于「不出力」状态
        return True

    def release_motor(self, motor_id: int) -> bool:
        """恢复某个舵机原来的限力值（hold_motor_off 的反操作）。"""
        self.check_connected()
        raw = self._current_limit_raw.get(int(motor_id), STS.TORQUE_LIMIT_MAX_RAW)  # 原来的值
        with self._bus_lock:
            result, _ = self.packet_handler.write2ByteTxRx(
                int(motor_id), STS.TORQUE_LIMIT, int(raw))       # 写回 48 号
        if result != COMM_SUCCESS:
            logging.error("Motor %d: torque limit not restored; it will not "
                          "move until something writes it again", motor_id)
            return False
        self._held_off.discard(int(motor_id))                    # 不再是「不出力」状态
        return True

    # ======================== 2. 运动参数（加速度、速度、限力） ========================

    def _write_profile_params(self, motor_ids: Sequence[int]) -> None:
        """一次性给多个舵机写：加速度（41）、目标时间=0 + 目标速度（44-47）、扭矩限制（48）。

        父类 HLS 版会把电流值写进 44 号；在 STS 上 44 号是「目标时间」，
        所以这里重写：44 号固定写 0，限力单独写 48 号。
        """
        acc_write = self._sync_write(STS.ACC, 1)                 # 包 1：41 号加速度，1 字节
        acc_write.clearParam()
        time_speed_write = self._sync_write(STS.GOAL_TIME, 4)    # 包 2：44-47 号，时间 + 速度，4 字节
        time_speed_write.clearParam()
        torque_write = self._sync_write(STS.TORQUE_LIMIT, 2)     # 包 3：48 号扭矩限制，2 字节
        torque_write.clearParam()
        for motor_id in motor_ids:                               # 对每个舵机，往三个包里各加一份数据
            acc_write.addParam(motor_id, [self._motor_acc.get(motor_id, DEFAULT_ACC)])
            time_speed_write.addParam(motor_id, [0, 0] + self._speed_bytes(motor_id))  # 时间=0，再接速度
            torque_write.addParam(motor_id, self._current_bytes(motor_id))
        for writer, name in ((acc_write, 'acceleration'),       # 依次发出三个包
                             (time_speed_write, 'time/speed'),
                             (torque_write, 'torque limit')):
            if writer.txPacket() != COMM_SUCCESS:                # 发送失败就清缓冲并报错
                self._flush_input_buffer()
                logging.error('Sync write of the motion-profile %s failed', name)
            writer.clearParam()

    def _restore_ram_after_reboot(self, motor_id: int) -> None:
        """舵机重启后 SRAM 会恢复出厂值，把我们设过的限力、速度、加速度写回去。"""
        motor_id = int(motor_id)
        failures = []                                            # 记录没写回去的寄存器
        for address, value in (                                  # 先写 2 字节的：
            (STS.TORQUE_LIMIT, self._current_limit_raw.get(motor_id, STS.TORQUE_LIMIT_MAX_RAW)),  # 限力最重要，先写
            (STS.GOAL_TIME, 0),                                  # 目标时间清零
            (STS.GOAL_SPEED, self._motor_speed.get(motor_id, DEFAULT_SPEED)),
        ):
            result, _ = self.packet_handler.write2ByteTxRx(motor_id, address, int(value))
            if result != COMM_SUCCESS:
                failures.append(address)
        result, _ = self.packet_handler.write1ByteTxRx(          # 再写 1 字节的加速度
            motor_id, STS.ACC, int(self._motor_acc.get(motor_id, DEFAULT_ACC)))
        if result != COMM_SUCCESS:
            failures.append(STS.ACC)
        if failures:                                             # 有没写回的就报错：可能在不限力状态下运行
            logging.error("Motor %d: RAM register(s) %s not restored after its reboot; "
                          "it may be running without its torque limit.",
                          motor_id, ", ".join(str(a) for a in failures))

    # ======================== 3. PID 增益（21-23 号，EEPROM） ========================

    def read_servo_gains(
        self, motor_ids: "Sequence[int]"
    ) -> "dict[int, ServoGains | None]":
        """读每个舵机的位置环增益。STS 存在 EEPROM 21（P）、22（D）、23（I）。"""
        out: "dict[int, ServoGains | None]" = {}                 # 结果：舵机 ID → 增益
        with self._bus_lock:
            for motor_id in motor_ids:
                motor_id = int(motor_id)
                values = []                                      # 依次读 P、D、I
                for address in (STS.KP, STS.KD, STS.KI):
                    value, result, error = self.packet_handler.read1ByteTxRx(motor_id, address)
                    if result != COMM_SUCCESS or error != 0:     # 任何一个读失败：
                        values = []                              #   整个舵机记为读不到
                        break
                    values.append(int(value))
                if len(values) != 3:                             # 没读全
                    self._flush_input_buffer()
                    out[motor_id] = None                         # 返回 None 表示「不知道」
                    continue
                kp, kd, ki = values                              # 注意寄存器顺序是 P、D、I
                out[motor_id] = ServoGains(kp=kp, ki=ki, kd=kd)
        return out

    def write_servo_gains(self, gains: "dict[int, ServoGains]") -> None:
        """不支持在程序里改增益。

        原因：STS 的增益在 EEPROM，EEPROM 写入次数有寿命限制，而 ORCA 每次启动
        都可能重新设一遍增益。需要调增益时，用 Feetech FD 上位机改一次即可。
        """
        raise ValueError(
            f"{type(self).__name__} keeps its PID gains in EEPROM (registers "
            "21-23); set them once with Feetech FD instead of on every bring-up")

    # ======================== 4. 零点标定（31 号零点偏移） ========================

    def _read_raw_word(self, motor_id: int, address: int) -> "int | None":
        """读一个 2 字节寄存器的原始值；舵机没回话就返回 None。"""
        value, result, _ = self.packet_handler.read2ByteTxRx(motor_id, address)
        if result != COMM_SUCCESS:
            self._flush_input_buffer()
            return None
        return int(value)

    def _read_raw_position(self, motor_id: int) -> "int | None":
        """读当前位置的原始格数（0-4095）。"""
        raw = self._read_raw_word(motor_id, STS.PRESENT_POSITION)
        if raw is None:
            return None
        return int(self.packet_handler.scs_tohost(raw, 15))     # 处理第 15 位符号位

    def _write_homing_offset(self, motor_id: int, offset: int) -> bool:
        """把零点偏移写进 EEPROM（31 号）：解锁 → 写 → 上锁。"""
        self._unlock_eeprom(motor_id)                            # LOCK=0，允许写 EEPROM
        result, _ = self.packet_handler.write2ByteTxRx(
            motor_id, STS.HOMING_OFFSET, _encode_homing_offset(offset))
        self._lock_eeprom(motor_id)                              # LOCK=1，锁上
        time.sleep(EEPROM_SETTLE_S)                              # 等 EEPROM 写完
        return result == COMM_SUCCESS

    def calibrate_offset(self, motor_id: int, upper: bool = True) -> bool:
        """把「当前物理位置」重新定义成 3595 格（upper）或 500 格（lower）。

        为什么要这样做：STS 的位置是 0-4095 一圈。如果关节的行程恰好跨过 0/4095
        这个点，读数会突然从 4095 跳到 0，翻译公式就全乱了。ORCA 的标定会在顶到
        一端限位时调用这里，把这一端放到 500 或 3595，整个行程就落在中间。

        做法：改 31 号零点偏移。舵机读数 = 编码器原始值 − 零点偏移，
        所以想让读数变成目标值 T，新偏移 = 旧偏移 + (当前读数 − T)。
        万一这款固件的符号约定相反，就自动试另一个方向；两个都不对就恢复原样并返回 False。
        """
        self.check_connected()                                   # 没连接就报错
        motor_id = int(motor_id)

        if POSITION_DIRECTION < 0:                               # Feetech 的读数在 ORCA 里是取反的，
            upper = not upper                                    #   所以「上限」对应的是小的原始格数
        target = 3595 if upper else 500                          # 目标原始格数（和 HLS 版一致）

        with self._bus_lock:                                     # 整个过程占住总线
            # 第 1 步：记下扭矩原来是开还是关，然后关掉，防止改偏移瞬间舵机猛转
            torque_was_on, result, _ = self.packet_handler.read1ByteTxRx(
                motor_id, STS.TORQUE_ENABLE)
            if result != COMM_SUCCESS:                           # 读不到就放弃
                self._flush_input_buffer()
                logging.error('Motor %d: could not read torque state for offset calibration',
                              motor_id)
                return False
            self.packet_handler.write1ByteTxRx(motor_id, STS.TORQUE_ENABLE, 0)

            # 第 2 步：读当前位置和当前偏移
            position = self._read_raw_position(motor_id)        # 当前读数 P
            old_raw = self._read_raw_word(motor_id, STS.HOMING_OFFSET)  # 当前偏移（编码后）
            if position is None or old_raw is None:            # 有一个读不到就放弃
                logging.error('Motor %d: could not read position/offset', motor_id)
                self._restore_torque(motor_id, torque_was_on)
                return False
            old_offset = _decode_homing_offset(old_raw)         # 解码成有正负的数

            # 第 3 步：算新偏移，写进去，读回来检查；不对就换一个方向再试
            delta = position - target                           # 读数要往下挪多少
            for candidate in (old_offset + delta, old_offset - delta):  # 先试正常约定，再试相反约定
                new_offset = _wrap_offset(candidate)            # 折回 -2047…2047
                if not self._write_homing_offset(motor_id, new_offset):
                    continue                                    # 没写进去，试下一个
                now = self._read_raw_position(motor_id)        # 读回新的读数
                if now is not None and abs(now - target) <= OFFSET_VERIFY_TOLERANCE_STEPS:
                    # 成功！把目标位置设成当前位置，再恢复扭矩时舵机就不会动
                    self.packet_handler.write2ByteTxRx(motor_id, STS.GOAL_POSITION, now)
                    self._goal_positions[motor_id] = self._decode_position(now)
                    self._restore_torque(motor_id, torque_was_on)
                    logging.info('Motor %d offset calibrated: position now reads %d',
                                 motor_id, now)
                    return True

            # 第 4 步：两个方向都不对，把旧偏移写回去，告诉上层失败了
            self._write_homing_offset(motor_id, old_offset)
            self._restore_torque(motor_id, torque_was_on)
            logging.error('Motor %d: offset calibration did not take; position '
                          'frame left unchanged', motor_id)
            return False

    def _restore_torque(self, motor_id: int, torque_was_on: int) -> None:
        """如果扭矩原来是开着的，就重新打开。"""
        if torque_was_on:
            self.packet_handler.write1ByteTxRx(motor_id, STS.TORQUE_ENABLE, 1)

    # ======================== 5. 识别：只认 STS 型号 ========================

    def no_load_speed_rad_s_for(self, motor_id: int) -> float:
        """某个舵机的空载转速。这条总线上都是 STS3215，所以直接用家族的值。"""
        return self.no_load_speed_rad_s

    @staticmethod
    def probe(
        port: str,                      # 要试的串口
        baudrate: int,                  # 要试的波特率
        motor_ids: Sequence[int],       # 配置里的舵机 ID
    ) -> bool:
        """自动识别用：试着 ping 第一个和最后一个舵机，回话且型号是 STS 才返回 True。

        只看「有没有回话」是不够的：HLS 舵机也会回话。必须检查型号编号，
        否则 ORCA 可能用 STS 的方式去驱动 HLS 手。
        """
        ids = list(motor_ids)
        if not ids:                                             # 没有舵机就不用试
            return False
        sample = [ids[0]] if len(ids) == 1 else [ids[0], ids[-1]]  # 只试头尾两个，速度快

        handler = PortHandler(port)                             # 打开一个临时串口
        handler.baudrate = baudrate
        try:
            if not handler.openPort():
                return False
            packet = HLSPacketHandler(handler)                  # 协议和 HLS 一样，直接用
            for motor_id in sample:
                model, comm, _ = packet.ping(motor_id)          # ping：舵机会回自己的型号编号
                if comm == COMM_SUCCESS and model in STS_MODEL_NUMBERS:
                    return True                                 # 是 STS，识别成功
            return False
        finally:                                                # 无论成功失败都要关串口
            try:
                handler.closePort()
            except Exception:
                pass

    def scan_for_motors(
        self,
        port: str,
        id_range: tuple,
        baud_rates: Optional[list] = None,
    ) -> list:
        """扫描总线，只返回 STS 型号的舵机（扫描本身由父类完成）。"""
        found = super().scan_for_motors(port, id_range, baud_rates)  # 父类扫出所有 Feetech 舵机
        return [m for m in found if m.get('model_name') in STS_MODEL_NAMES]  # 只留 STS
