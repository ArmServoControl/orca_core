# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================

"""STS3215 的「假舵机」：没有硬件时用来跑代码和测试。

它继承 HLS 的假舵机，只把「家族描述」换成 StsClient 的，
这样 load_hand(mock=True) 加载 motor_type: feetech_sts 的手时，行为和真 STS 一致。
"""

from .mock_feetech_client import MockFeetechClient  # HLS 的假舵机，大部分行为相同
from .motor_client import ServoGains  # 增益的数据格式
from .sts_client import StsClient  # 真的 STS 驱动，用来复制家族属性


class MockStsClient(MockFeetechClient):
    """模拟 STS3215：家族属性（电流换算、默认限力、型号等）全部取自 StsClient。"""

    real_client = StsClient                          # 告诉基类：照着 StsClient 复制家族属性
    # STS3215 出厂的增益：P=32，D=32，I=0
    default_servo_gains = ServoGains(kp=32, ki=0, kd=32)

    def write_servo_gains(self, gains) -> None:
        """和真舵机一样拒绝写增益（STS 的增益在 EEPROM）。"""
        self.check_connected()
        StsClient.write_servo_gains(self, gains)     # 直接调用真驱动的同名方法，抛出同样的错误
