# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================

"""Feetech STS 系列（STS3215）寄存器地址表。

这个文件只做一件事：把「寄存器名字」对应到「地址数字」，并写清楚单位。
地址来源：ORCA 仓库自带的 Feetech SDK（orca_core/hardware/feetech/sms_sts.py）
和 Feetech 官方 STS 系列内存表。

和 HLS 系列（feetech_registers.py）的对照：
- 绝大多数地址完全相同（ID=5、模式=33、目标位置=42、当前位置=56 ……），
  所以 StsClient 可以继承 ORCA 现成的 FeetechClient。
- 真正不同的只有三处，StsClient 里重写的就是这三处：
    44 号：HLS 是「目标电流」，STS 是「目标时间」。
    48 号：两者都叫「扭矩限制」，但 STS 用它来限力（STS 没有目标电流寄存器）。
    21-23 号：STS 的位置环 P/D/I 增益在 EEPROM；HLS 的在 SRAM 50-52 号。
- 另外 STS 的零点偏移（31 号）是「符号-数值」编码，第 11 位是符号位。
"""


class STS:
    # ---------------- EEPROM 区：断电后保留，改之前要先解锁（LOCK 写 0）----------------
    MODEL = 3                 # 型号编号（2 字节，只读）。STS3215 读出来是 777
    ID = 5                    # 舵机 ID，0-253
    BAUD_RATE = 6             # 波特率代码：0=1M, 1=500k, 2=250k ...（和 HLS 一样）
    RETURN_DELAY = 7          # 回复延迟，单位 2 µs（HLS 的 7 号是「影子 ID」，这里不一样）
    MIN_ANGLE_LIMIT = 9       # 位置下限（2 字节）
    MAX_ANGLE_LIMIT = 11      # 位置上限（2 字节）
    MAX_TEMPERATURE = 13      # 过热保护温度，°C
    UNLOAD_CONDITIONS = 19    # 哪些故障会让舵机自动卸力（位掩码）
    KP = 21                   # 位置环 P 增益（出厂 32）
    KD = 22                   # 位置环 D 增益（出厂 32）
    KI = 23                   # 位置环 I 增益（出厂 0）
    PROTECTION_CURRENT = 28   # 过流保护阈值（2 字节）
    HOMING_OFFSET = 31        # 零点偏移（2 字节），第 11 位是符号位
    MODE = 33                 # 工作模式：0 位置，1 闭环速度，2 开环速度，3 步进

    # ---------------- SRAM 区：断电后丢失，可以随时读写 ----------------
    TORQUE_ENABLE = 40        # 扭矩开关：0 关，1 开（写 128 = 把当前位置设为中位 2048）
    ACC = 41                  # 加速度，单位 8.7°/s²，0 = 最大
    GOAL_POSITION = 42        # 目标位置（2 字节），0-4095
    GOAL_TIME = 44            # 目标时间（2 字节）。HLS 的这个地址是目标电流！STS 上我们永远写 0
    GOAL_SPEED = 46           # 目标速度（2 字节），单位 0.732 rpm
    TORQUE_LIMIT = 48         # 扭矩限制（2 字节），0-1000，单位 0.1%。STS 靠它限力
    LOCK = 55                 # EEPROM 锁：0 = 允许写入并断电保存，1 = 锁上

    # ---------------- SRAM 只读区：舵机的实时状态 ----------------
    PRESENT_POSITION = 56     # 当前位置（2 字节），0-4095
    PRESENT_SPEED = 58        # 当前速度（2 字节），第 15 位是符号位
    PRESENT_LOAD = 60         # 当前负载（2 字节），单位 0.1%
    PRESENT_VOLTAGE = 62      # 当前电压，单位 0.1 V
    PRESENT_TEMPERATURE = 63  # 当前温度，°C
    MOVING = 66               # 是否在运动：1 是，0 否
    PRESENT_CURRENT = 69      # 当前电流（2 字节），单位 6.5 mA，第 15 位是符号位

    # ---------------- 换算常数 ----------------
    GAIN_MAX = 254            # P/D/I 寄存器是 1 字节，最大 254
    TORQUE_LIMIT_MAX_RAW = 1000   # 扭矩限制满量程 = 1000 = 100%
    HOMING_OFFSET_SIGN_BIT = 11   # 零点偏移的符号位在第 11 位
    HOMING_OFFSET_MAX = 2047      # 零点偏移的最大绝对值（11 位能表示的最大数）
    STEPS_PER_REV = 4096          # 一圈 4096 格

    # 数据手册（STS3215-C001，7.4 V）：堵转电流 2.5 A，空载转速 52 rpm。
    # 扭矩和电流成正比（Kt = 7.8 kg·cm/A），所以「扭矩限制 100%」≈「电流上限 2500 mA」。
    STALL_CURRENT_MA = 2500.0
    NO_LOAD_RPM = 52.0


# 型号编号 → 名字。舵机 ping 回复里带着型号编号，用来区分 STS 和 HLS。
# 777 是 STS3215 的型号编号（和 LeRobot 项目里的一致）。拿到实物后用
# team/sts_scan.py 读一下 model 那一列确认；如果不是 777，就把读到的数字加到这里。
STS_MODELS: dict[int, str] = {
    777: "STS3215",
}
STS_MODEL_NUMBERS = frozenset(STS_MODELS)          # 只要编号，方便用 in 判断
STS_MODEL_NAMES = frozenset(STS_MODELS.values())   # 只要名字，方便过滤扫描结果
